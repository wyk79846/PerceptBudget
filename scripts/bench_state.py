#!/usr/bin/env python3
"""Compare paired synchronous inference latency with and without LIBERO state."""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import load_libero_norm_stats, prepare_libero_observation
from perceptbudget.model import validate_actions
from scripts import eval_q

SEED, WARMUPS, PAIRS = 7, 2, 5

def _graph_info(pipe, with_state):
    active = pipe.pipeline
    return {
        "routing": "fixed_state_pipeline" if with_state else "per_length_prompt_cache",
        "frontend_state_prompt_mode": getattr(pipe, "_state_prompt_mode", None),
        "prompt_length": getattr(pipe, "current_prompt_len", None),
        "pipeline_max_prompt_len": getattr(active, "max_prompt_len", None),
        "pipeline_fixed_shape": getattr(active, "_fixed_shape", None),
        "frontend_graph_recorded": bool(getattr(pipe, "graph_recorded", False)),
        "pipeline_graph_present": getattr(active, "_graph", None) is not None,
        "chunk_size": getattr(active, "chunk_size", None),
        "num_views": getattr(active, "num_views", None),
    }

def _predict(model, snapshot, prompt, with_state, torch):
    kwargs = {"prompt": prompt}
    if with_state:
        kwargs["state"] = snapshot["state"]
    torch.cuda.synchronize()
    started = time.perf_counter()
    actions = model.predict(snapshot["images"], **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    actions = validate_actions(actions, model._pipe.chunk_size)
    return elapsed_ms, list(actions.shape)

def _save_summary(run_dir, summary):
    eval_q._write_json_atomic(run_dir / "summary.json", summary)
    rows = [
        "# State 推理延迟对比", "", f"- status: `{summary['status']}`",
        f"- run directory: `{run_dir}`",
        f"- task: {summary.get('task_name', 'pending')} (`{summary.get('task_id', 0)}`), init 0, settle 10",
        f"- sync diagnostic; H={summary['config_snapshot']['H']}, N=10, BF16, two views; action shape: `{summary.get('action_shape', 'pending')}`",
        "- timed: `predict()` through CUDA sync; excludes load, input prep, and warmup graph build.",
        "- seed 7 is reset before each call; state/no-state calls alternate.", "",
        "| pair | state ms | no-state ms | added ms with state | added % vs no-state |",
        "|---:|---:|---:|---:|---:|",
    ]
    for p in summary.get("pairs", []):
        rows.append(f"| {p['pair']} | {p['state_ms']:.3f} | {p['no_state_ms']:.3f} | {p['delta_ms']:+.3f} | {p['delta_percent']:+.2f}% |")
    stats = summary.get("statistics")
    if stats:
        rows.extend(("", "| path | mean ms | median ms | raw ms |", "|---|---:|---:|---|"))
        for key, label in (("state", "state"), ("no_state", "no-state")):
            item = stats[key]
            raw = ", ".join("{:.3f}".format(x) for x in item["raw_ms"])
            rows.append(f"| {label} | {item['mean_ms']:.3f} | {item['median_ms']:.3f} | {raw} |")
        rows.append(f"\nMean delta: {stats['delta_mean_ms']:+.3f} ms ({stats['delta_mean_percent']:+.2f}%); median paired: {stats['delta_median_ms']:+.3f} ms ({stats['delta_median_percent']:+.2f}%).")
    if summary.get("graph_paths"):
        rows.extend(("", "## Actual frontend graph paths", "", "```json",
                     json.dumps(summary["graph_paths"], ensure_ascii=False, indent=2), "```"))
    if summary.get("error"):
        rows.extend(("", "## Error", "", "```text", summary["error"], "```"))
    rows.extend(("", "## Config snapshot", "", "```json",
                 json.dumps(summary.get("config_snapshot", {}), ensure_ascii=False, indent=2), "```"))
    (run_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

def _validate_config(config):
    if ((config.get("model_id"), config.get("N"), config.get("precision"))
            != ("pi05_libero_pytorch", 10, "bfloat16")
            or config.get("H") not in (10, 20)
            or type(config.get("state_enabled")) is not bool):
        raise ValueError("要求 pi05_libero_pytorch、H=10/20、N=10、BF16 和布尔 state_enabled")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json")
    parser.add_argument("--horizon", type=int, choices=(10, 20),
                        help="Override H for this diagnostic only; config.json is unchanged")
    args = parser.parse_args(argv)
    run_dir = summary = None
    try:
        config = load_config(args.config, repo_root=PROJECT_ROOT)
        base_config = json.loads(json.dumps(config))
        if args.horizon is not None:
            config["H"] = args.horizon
            config["candidate_q"] = [q for q in config["candidate_q"] if q <= args.horizon]
        _validate_config(config)
        norm_stats = load_libero_norm_stats(config["norm_stats_path"])
        state_stats = norm_stats.get("state", {})
        if any(len(state_stats.get(k, [])) != 8 for k in ("q01", "q99")):
            raise ValueError("checkpoint state q01/q99 必须恰为 8 维")

        os.environ.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0", PYOPENGL_PLATFORM="egl")
        egl_cleanup = eval_q._patch_egl_cleanup()
        run_dir = create_run(config, "M1", "state-latency", repo_root=PROJECT_ROOT)
        summary = {
            "schema_version": 1, "status": "running", "purpose": "state-latency",
            "run_mode": "synchronous_diagnostic", "run_dir": str(run_dir),
            "config_snapshot": config, "state_enabled_config_unchanged": config["state_enabled"],
            "base_config_snapshot": base_config,
            "diagnostic_horizon_override": args.horizon,
            "task_q_evaluation_started": False,
            "task_id": 0, "init_state_index": 0, "settle_steps": 10,
            "control_freq_hz": 20, "seed": SEED, "warmups_per_path": WARMUPS,
            "measured_pairs": PAIRS,
            "state_protocol": "8D checkpoint q01/q99 normalization; extra prompt tokens",
            "normalization": "(x-q01)/(q99-q01+1e-6)*2-1",
            "timing_boundary": "wall time around predict() through torch.cuda.synchronize",
            "excluded_from_measurement": ["model load", "input preparation", "first graph build in warmup"],
            "egl_environment": {k: os.environ[k] for k in (
                "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "PYOPENGL_PLATFORM")},
            "egl_cleanup": egl_cleanup, "warmup_ms": {"state": [], "no_state": []}, "pairs": [],
        }
        eval_q._save_run_status(run_dir, "running", run_mode=summary["run_mode"])
        _save_summary(run_dir, summary)
        import numpy as np
        task = eval_q._load_tasks([0])[0]
        _, env_class = eval_q._libero_api()
        env = eval_q._make_env(env_class, task)
        try:
            observation, actual_frequency = eval_q._prepare_env_episode(env, task, np)
            snapshot = prepare_libero_observation(observation, norm_stats, state_enabled=True)
        finally:
            env.close()
        summary.update(
            task_name=task["task_name"], instruction=task["instruction"], control_freq_actual_hz=actual_frequency,
            observation_shapes={"agentview_image": list(observation["agentview_image"].shape),
                "robot0_eye_in_hand_image": list(observation["robot0_eye_in_hand_image"].shape),
                "model_images": [list(image.shape) for image in snapshot["images"]],
                "normalized_state": list(snapshot["state"].shape)},
        )
        torch, cuda_report = eval_q._check_cuda_and_native(config)
        load_started = time.perf_counter()
        model = eval_q._load_and_check_model(config, torch)
        pipe = model._pipe
        if getattr(pipe, "_state_prompt_mode", None) != "fixed":
            raise RuntimeError("实际 FlashRT state_prompt_mode 不是 fixed")
        summary["model"] = {
            **cuda_report, "loaded_once": True, "load_seconds": time.perf_counter() - load_started,
            "H": config["H"], "N": config["N"], "precision": config["precision"],
            "num_views": getattr(pipe, "num_views", 2), "state_prompt_mode": pipe._state_prompt_mode,
            "full_bf16_confirmed": True,
        }

        for with_state, label in ((True, "state"), (False, "no_state")):
            for _ in range(WARMUPS):
                eval_q._seed_process(SEED, np, torch)
                elapsed, _ = _predict(model, snapshot, task["instruction"], with_state, torch)
                summary["warmup_ms"][label].append(elapsed)
            summary.setdefault("graph_paths", {})[label] = _graph_info(pipe, with_state)
        _save_summary(run_dir, summary)
        for pair in range(1, PAIRS + 1):
            values = {}
            for with_state, label in ((True, "state"), (False, "no_state")):
                target = snapshot["state"] if with_state else None
                if model._prompt_state_equal(model._current_prompt_state, target):
                    raise RuntimeError("交替测量未触发 state prompt 更新")
                eval_q._seed_process(SEED, np, torch)
                elapsed, shape = _predict(model, snapshot, task["instruction"], with_state, torch)
                values[label] = elapsed
                summary["action_shape"] = shape
            delta = values["state"] - values["no_state"]
            summary["pairs"].append({
                "pair": pair, "state_ms": values["state"], "no_state_ms": values["no_state"],
                "delta_ms": delta, "delta_percent": delta / values["no_state"] * 100.0,
                "same_seed": SEED,
            })
            _save_summary(run_dir, summary)

        state_ms, no_state_ms = ([p["state_ms"] for p in summary["pairs"]], [p["no_state_ms"] for p in summary["pairs"]])
        stats = {
            "state": {"raw_ms": state_ms, "mean_ms": statistics.mean(state_ms), "median_ms": statistics.median(state_ms)},
            "no_state": {"raw_ms": no_state_ms, "mean_ms": statistics.mean(no_state_ms), "median_ms": statistics.median(no_state_ms)},
        }
        delta = stats["state"]["mean_ms"] - stats["no_state"]["mean_ms"]
        stats.update(
            delta_mean_ms=delta, delta_mean_percent=delta / stats["no_state"]["mean_ms"] * 100.0,
            delta_median_ms=statistics.median(p["delta_ms"] for p in summary["pairs"]),
            delta_median_percent=statistics.median(p["delta_percent"] for p in summary["pairs"]),
        )
        summary.update(status="completed", statistics=stats)
        _save_summary(run_dir, summary)
        eval_q._save_run_status(run_dir, "completed", run_mode=summary["run_mode"])
        print("完成：" + str(run_dir), flush=True)
        return 0
    except Exception as exc:
        if run_dir is not None and summary is not None:
            summary.update(status="invalid", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            try:
                _save_summary(run_dir, summary)
                eval_q._save_run_status(run_dir, "invalid", error=summary["error"])
            except Exception:
                traceback.print_exc()
        traceback.print_exc()
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
