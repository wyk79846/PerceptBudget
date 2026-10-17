#!/usr/bin/env python3
"""Paired synchronous LIBERO-10 screen for the paper's SEAM settings.

Each rollout blocks during inference, then executes the first K actions of the
new H-action chunk. In the SEAM condition, the previous chunk's unexecuted tail
is supplied as the aligned prior (no asynchronous delay/cropping is involved).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import raw_libero_state
from perceptbudget.model import validate_actions
from scripts import eval_q as base

HORIZON = 50
DENOISING_STEPS = 10
EXECUTE_K = 10
SEAM_M = 20
SEAM_LAMBDA = 0.1
CONTROL_HZ = base.CONTROL_FREQ_HZ
SEED = base.SEED
INIT_STATE_INDEX = 0
MAX_STEPS_BY_SUITE = {"libero_10": 520}


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def get_tasks(task_ids):
    benchmark, _ = base._libero_api()
    registry = benchmark.get_benchmark_dict()
    suite_name = "libero_10"
    if suite_name not in registry:
        raise RuntimeError("LIBERO registry has no libero_10 suite")
    suite = registry[suite_name]()
    if suite.n_tasks != 10:
        raise RuntimeError(f"libero_10 contains {suite.n_tasks}, expected 10 tasks")
    tasks = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) <= INIT_STATE_INDEX:
            raise RuntimeError(f"task {task_id} has no init state {INIT_STATE_INDEX}")
        bddl_path = Path(suite.get_task_bddl_file_path(task_id))
        if not bddl_path.is_file():
            raise FileNotFoundError(str(bddl_path))
        tasks.append({
            "task_id": task_id,
            "suite": suite_name,
            "suite_task_id": task_id,
            "task_name": task.name,
            "instruction": task.language,
            "bddl_path": bddl_path,
            "init_state": initial_states[INIT_STATE_INDEX],
            "max_steps": MAX_STEPS_BY_SUITE[suite_name],
        })
    return tasks


def make_env(env_class, task):
    env = base._make_env(env_class, task)
    # LIBERO randomizes fixture poses on hard reset. The robot state vector
    # does not contain those poses, so freeze each task's first reset layout
    # before running the paired baseline/SEAM episodes.
    env._pb_lock_fixtures = True
    return env


def raw_chunk(model, horizon):
    import torch
    raw = getattr(model._pipe, "_noise_out", None)
    if raw is None or tuple(raw.shape) != (horizon, 32) or not torch.isfinite(raw).all():
        raise RuntimeError("FlashRT did not expose finite raw Hx32 actions for SEAM")
    return raw.float().cpu().numpy().astype("float32", copy=True)


def write_markdown(run_dir, summary):
    settings = summary.get("settings", {})
    rows = [
        "# SEAM 同步小规模测试",
        "",
        "- 状态：`{}`".format(summary.get("status")),
        "- 参数：H={}，N={}，K={}，M={}，λ={}".format(
            settings.get("H"), settings.get("N"), settings.get("K"),
            settings.get("M"), settings.get("lambda")),
        "- 评测：LIBERO-10，每任务每条件一次；seed={}，init_state_index={}".format(
            settings.get("seed"), settings.get("init_state_index")),
        "- 状态输入：{}；精度：{}".format(
            settings.get("state_enabled"), settings.get("precision")),
        "",
        "| task | baseline | SEAM | steps (base/SEAM) | inference median ms (base/SEAM) | 初态/画面/fixture 匹配 | 首 chunk 相同 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in summary.get("pairs", []):
        base_ms = pair.get("baseline_inference_median_seconds")
        seam_ms = pair.get("seam_inference_median_seconds")
        rows.append("| {} `{}` | {} | {} | {}/{} | {}/{} | {} | {} |".format(
            pair["task_id"], pair["task_name"],
            "成功" if pair["baseline_success"] else "失败/超时",
            "成功" if pair["seam_success"] else "失败/超时",
            pair["baseline_steps"], pair["seam_steps"],
            "—" if base_ms is None else "{:.1f}".format(base_ms * 1000),
            "—" if seam_ms is None else "{:.1f}".format(seam_ms * 1000),
            "{}/{}/{}".format(
                pair["initial_state_max_abs_delta"] == 0,
                pair["initial_frames_match"], pair["fixture_layout_match"]),
            pair["first_chunk_match"],
        ))
    aggregate = summary.get("aggregate")
    if aggregate:
        rows.extend((
            "",
            "- 汇总成功率：baseline {}/{}；SEAM {}/{}。".format(
                aggregate["baseline_successes"], aggregate["paired_tasks"],
                aggregate["seam_successes"], aggregate["paired_tasks"]),
            "- 共同配对初态：{} / {} 帧相同；首 chunk 相同 {} / {}。".format(
                aggregate["initial_frame_matches"], aggregate["paired_tasks"],
                aggregate["first_chunk_matches"], aggregate["paired_tasks"]),
        ))
    if summary.get("error"):
        rows.extend(("", "## Error", "", "```text", summary["error"], "```"))
    rows.extend(("", "结果目录：`{}`".format(run_dir), ""))
    (Path(run_dir) / "summary.md").write_text("\n".join(rows), encoding="utf-8")


def run_episode(*, env, task, condition, model, adapter, stats, config,
                np, torch, trace, order_index):
    base._seed_process(SEED, np, torch)
    observation, frequency = base._prepare_env_episode(env, task, np)
    if abs(frequency - CONTROL_HZ) > 1e-9:
        raise RuntimeError(f"LIBERO control frequency is {frequency}, expected {CONTROL_HZ}")
    snapshot = base._make_snapshot(observation, stats, config["state_enabled"])
    initial_state = np.asarray(snapshot["raw_state"], dtype=np.float64)
    initial_frame_hash = digest_arrays([np.ascontiguousarray(image) for image in snapshot["images"]])
    initial_state_hash = hashlib.sha256(initial_state.tobytes()).hexdigest()
    fixture_layout_hash = hashlib.sha256(json.dumps(
        env._pb_fixture_layout, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    records = []
    prior_raw = None
    control_steps = 0
    success = False
    env_done = False
    deadline_misses = 0
    inference_seconds = []
    episode_start = time.monotonic()
    deadline = None
    terminal_status = "timeout"

    while control_steps < task["max_steps"]:
        snapshot = base._make_snapshot(observation, stats, config["state_enabled"])
        request = {}
        infer_start = time.monotonic()
        if condition == "seam" and prior_raw is not None:
            actions = adapter.predict(
                snapshot, task["instruction"], prior_raw[EXECUTE_K:],
                executed=EXECUTE_K, delay_estimate=0, request=request,
            )
            current_raw = np.asarray(request["actions_raw"], dtype=np.float32)
            if current_raw.shape != (HORIZON, 32) or not np.isfinite(current_raw).all():
                raise RuntimeError("SEAM output is not finite Hx32")
            seam_meta = request.get("seam_guidance", {})
        else:
            actions = base._infer_actions(
                model, snapshot, task["instruction"], HORIZON, request,
            )
            current_raw = raw_chunk(model, HORIZON)
            seam_meta = None
        inference_seconds.append(time.monotonic() - infer_start)
        actions = validate_actions(actions, HORIZON)
        if condition == "seam" and prior_raw is not None:
            expected_window = min(SEAM_M, HORIZON - EXECUTE_K)
            if seam_meta.get("guided_start") != 0 or seam_meta.get("guided_window") != expected_window:
                raise RuntimeError(f"Unexpected synchronous SEAM window: {seam_meta}")

        chunk_index = len(records)
        action_count = min(EXECUTE_K, task["max_steps"] - control_steps)
        raw_hash = hashlib.sha256(current_raw.tobytes()).hexdigest()
        record = {
            "chunk_index": chunk_index,
            "observation_step": control_steps,
            "inference_seconds": inference_seconds[-1],
            "raw_chunk_sha256": raw_hash,
            "actions_to_execute": action_count,
            "seam_guidance": seam_meta,
        }
        records.append(record)
        trace.write(json.dumps({
            "event": "chunk_prediction", "task_id": task["task_id"],
            "condition": condition, **record,
        }, ensure_ascii=False) + "\n")
        trace.flush()

        # Synchronous scheduling: no environment actions run while the model
        # predicts. Resume the same nominal 20 Hz control cadence afterwards.
        if deadline is None:
            deadline = time.monotonic()
        for offset in range(action_count):
            step = control_steps
            action = np.asarray(actions[offset], dtype=np.float32).copy()
            before_state = np.asarray(raw_libero_state(observation), dtype=np.float64)
            result = env.step(action)
            if not isinstance(result, tuple) or len(result) != 4:
                raise RuntimeError("LIBERO env.step must return (obs, reward, done, info)")
            observation, reward, env_done, _info = result
            control_steps += 1
            success = bool(env.check_success())
            terminal_status = "success" if success else (
                "failure" if env_done else "timeout")
            after_state = np.asarray(raw_libero_state(observation), dtype=np.float64)
            trace.write(json.dumps({
                "event": "control_step", "task_id": task["task_id"],
                "condition": condition, "control_step": step,
                "executed_action": action.tolist(),
                "raw_state": before_state.tolist(),
                "next_raw_state": after_state.tolist(),
                "reward": float(reward), "env_done": bool(env_done),
                "env_success": success,
            }, ensure_ascii=False) + "\n")
            trace.flush()
            if success or env_done or control_steps >= task["max_steps"]:
                break

            deadline += 1.0 / frequency
            now = time.monotonic()
            if now > deadline:
                deadline_misses += 1
                deadline = now
            else:
                time.sleep(deadline - now)

        prior_raw = current_raw
        if success or env_done or control_steps >= task["max_steps"]:
            break

    stop = time.monotonic()
    return {
        "task_id": task["task_id"], "suite": task["suite"],
        "task_name": task["task_name"], "instruction": task["instruction"],
        "condition": condition, "order_index": order_index,
        "seed": SEED, "init_state_index": INIT_STATE_INDEX,
        "status": terminal_status, "success": success,
        "control_steps": control_steps,
        "simulated_seconds": control_steps / frequency,
        "real_seconds": stop - episode_start,
        "deadline_misses": deadline_misses,
        "inference_count": len(inference_seconds),
        "inference_seconds": inference_seconds,
        "inference_mean_seconds": float(np.mean(inference_seconds)) if inference_seconds else None,
        "inference_median_seconds": float(np.median(inference_seconds)) if inference_seconds else None,
        "initial_raw_state": initial_state.tolist(),
        "initial_raw_state_sha256": initial_state_hash,
        "initial_frames_sha256": initial_frame_hash,
        "fixture_layout_sha256": fixture_layout_hash,
        "first_chunk_sha256": records[0]["raw_chunk_sha256"] if records else None,
        "chunks": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)),
                        help="LIBERO-10 task ids; default: all 10 once per condition")
    parser.add_argument("--flashrt-root", type=Path,
                        default=Path("/home/wyk/workspace/FlashRT-seam-native"))
    args = parser.parse_args(argv)
    if len(set(args.task_ids)) != len(args.task_ids) or any(i < 0 or i >= 10 for i in args.task_ids):
        parser.error("task ids must be unique integers from 0 through 9")

    os.environ.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0", PYOPENGL_PLATFORM="egl")
    flashrt_root = args.flashrt_root.expanduser().resolve()
    config = load_config(repo_root=ROOT)
    config.update(
        flashrt_root=str(flashrt_root),
        H=HORIZON, N=DENOISING_STEPS, candidate_q=[EXECUTE_K],
        execution_mode="seam_synchronous_quality",
        seam={"H": HORIZON, "N": DENOISING_STEPS, "K": EXECUTE_K,
              "M": SEAM_M, "lambda": SEAM_LAMBDA,
              "prior": "previous normalized raw action chunk after first K actions",
              "schedule": "pause during inference; execute first K; repeat"},
    )
    config["screen"] = {
        **config.get("screen", {}), "suites": ["libero_10"],
        "task_ids": list(args.task_ids), "trials_per_task": 1,
        "seed": SEED, "init_state_index": INIT_STATE_INDEX,
    }
    run_dir = create_run(config, "M1", "seam-sync-h50-k10", repo_root=ROOT)
    print("run_dir=" + str(run_dir), flush=True)
    summary = {
        "status": "running", "purpose": "paired synchronous SEAM screen",
        "settings": {"H": HORIZON, "N": DENOISING_STEPS, "K": EXECUTE_K,
                     "M": SEAM_M, "lambda": SEAM_LAMBDA,
                     "control_hz": CONTROL_HZ, "seed": SEED,
                     "init_state_index": INIT_STATE_INDEX,
                     "fixture_layout_locked": True,
                     "state_enabled": config["state_enabled"],
                     "precision": config["precision"]},
        "suite": "libero_10", "task_ids": list(args.task_ids),
        "episodes": [], "pairs": [],
        "limitations": [
            "每任务每条件一次，仅为小规模筛查，不是论文规模复现。",
            "SEAM 与基线均使用本项目当前模型/状态输入配置；仅衔接算法不同。",
            "同步推理期间暂停仿真；不测异步延迟收益。",
        ],
        "episodes_expected": len(args.task_ids) * 2,
        "source_hashes": {},
    }
    source_files = [
        ROOT / "scripts/eval_seam_sync.py",
        ROOT / "perceptbudget/seam_rollout.py",
        flashrt_root / "flash_rt/subgraphs/pi05/seam_vls.py",
        flashrt_root / "flash_rt/models/pi05/pipeline_rtx.py",
        flashrt_root / "csrc/kernels/elementwise.cu",
    ]
    for path in source_files:
        if path.is_file():
            summary["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    np = torch = model = adapter = None
    env = None
    try:
        summary["egl_cleanup"] = base._patch_egl_cleanup()
        torch, cuda_report = base._check_cuda_and_native(config)
        summary["cuda"] = cuda_report
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats
        from perceptbudget.seam_rollout import SEAMGuidanceAdapter
        from flash_rt.subgraphs.pi05.context_action import enable as enable_context

        stats = load_libero_norm_stats(config["norm_stats_path"])
        tasks = get_tasks(args.task_ids)
        summary["tasks"] = [{k: task[k] for k in
                              ("task_id", "task_name", "instruction", "max_steps")}
                             for task in tasks]
        load_started = time.monotonic()
        model = base._load_and_check_model(config, torch)
        summary["model"] = {
            "loaded_once": True, "load_seconds": time.monotonic() - load_started,
            "horizon": model._pipe.chunk_size,
            "denoising_steps": model._pipe._num_steps,
            "flashrt_root": config["flashrt_root"],
        }
        enable_context(model)
        _, env_class = base._libero_api()

        # Prime the first task's prompt and capture FlashRT's context graph
        # before attaching the optional SEAM decoder graph.
        first_task = tasks[0]
        env = make_env(env_class, first_task)
        obs, _ = base._prepare_env_episode(env, first_task, np)
        warm_snapshot = base._make_snapshot(obs, stats, config["state_enabled"])
        base._seed_process(SEED, np, torch)
        base._infer_actions(model, warm_snapshot, first_task["instruction"], HORIZON, {})
        adapter = SEAMGuidanceAdapter(
            model, torch, horizon=HORIZON, num_steps=DENOISING_STEPS,
            strength=SEAM_LAMBDA, max_window=SEAM_M,
        )
        summary["seam_runtime"] = {
            "implementation": "FlashRT isolated native fused CUDA VLS graph",
            "no_op_parity": adapter.output_parity,
            "graph_capture_seconds": adapter.capture_seconds,
            "expected_synchronous_guidance": {
                "executed_prefix": EXECUTE_K, "delay_steps": 0,
                "guided_start": 0, "guided_window": SEAM_M,
                "old_tail_length": HORIZON - EXECUTE_K,
            },
        }
        base._write_json_atomic(run_dir / "summary.json", summary)

        with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
            for task_index, task in enumerate(tasks):
                if task_index or task["task_id"] != first_task["task_id"]:
                    if env is not None:
                        env.close()
                    env = make_env(env_class, task)
                    obs, _ = base._prepare_env_episode(env, task, np)
                    warm_snapshot = base._make_snapshot(obs, stats, config["state_enabled"])
                    base._seed_process(SEED, np, torch)
                    base._infer_actions(model, warm_snapshot, task["instruction"], HORIZON, {})

                # Alternate order across tasks to reduce warm-start/order bias.
                conditions = ["baseline", "seam"]
                if task_index % 2:
                    conditions.reverse()
                episodes_by_condition = {}
                for order_index, condition in enumerate(conditions):
                    episode = run_episode(
                        env=env, task=task, condition=condition,
                        model=model, adapter=adapter, stats=stats, config=config,
                        np=np, torch=torch, trace=trace, order_index=order_index,
                    )
                    episodes_by_condition[condition] = episode
                    summary["episodes"].append(episode)
                    base._write_json_atomic(run_dir / "summary.json", summary)
                    print("task={} condition={} status={} steps={} real={:.1f}s".format(
                        task["task_id"], condition, episode["status"],
                        episode["control_steps"], episode["real_seconds"]), flush=True)

                baseline = episodes_by_condition["baseline"]
                seam = episodes_by_condition["seam"]
                state_delta = max(abs(a-b) for a, b in zip(
                    baseline["initial_raw_state"], seam["initial_raw_state"]))
                pair = {
                    "task_id": task["task_id"], "task_name": task["task_name"],
                    "baseline_success": baseline["success"],
                    "seam_success": seam["success"],
                    "baseline_steps": baseline["control_steps"],
                    "seam_steps": seam["control_steps"],
                    "initial_state_max_abs_delta": state_delta,
                    "initial_frames_match": baseline["initial_frames_sha256"] == seam["initial_frames_sha256"],
                    "fixture_layout_match": baseline["fixture_layout_sha256"] == seam["fixture_layout_sha256"],
                    "first_chunk_match": baseline["first_chunk_sha256"] == seam["first_chunk_sha256"],
                    "baseline_inference_median_seconds": baseline["inference_median_seconds"],
                    "seam_inference_median_seconds": seam["inference_median_seconds"],
                }
                summary["pairs"].append(pair)
                trace.write(json.dumps({"event": "paired_result", **pair}, ensure_ascii=False) + "\n")
                trace.flush()
                base._write_json_atomic(run_dir / "summary.json", summary)
                if state_delta > 1e-8 or not pair["initial_frames_match"]:
                    summary["pairing_warning"] = "initial conditions did not exactly match"

        total = len(summary["pairs"])
        summary["aggregate"] = {
            "paired_tasks": total,
            "baseline_successes": sum(p["baseline_success"] for p in summary["pairs"]),
            "seam_successes": sum(p["seam_success"] for p in summary["pairs"]),
            "baseline_success_rate": (sum(p["baseline_success"] for p in summary["pairs"]) / total) if total else None,
            "seam_success_rate": (sum(p["seam_success"] for p in summary["pairs"]) / total) if total else None,
            "first_chunk_matches": sum(p["first_chunk_match"] for p in summary["pairs"]),
            "initial_frame_matches": sum(p["initial_frames_match"] for p in summary["pairs"]),
            "fixture_layout_matches": sum(p["fixture_layout_match"] for p in summary["pairs"]),
        }
        summary["status"] = "completed"
        base._save_run_status(run_dir, "completed", completed_episodes=len(summary["episodes"]))
    except BaseException as exc:
        summary.update(status="invalid", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
        base._save_run_status(run_dir, "invalid", error=summary["error"])
        raise
    finally:
        if env is not None:
            env.close()
        base._write_json_atomic(run_dir / "summary.json", summary)
        write_markdown(run_dir, summary)
    print("summary=" + str(run_dir / "summary.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
