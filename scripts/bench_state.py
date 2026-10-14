#!/usr/bin/env python3
"""Measure synchronous inference latency for no-state and state input paths."""

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

SEED, WARMUPS, MEASUREMENTS = 7, 2, 5

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

def _predict(model, snapshot, prompt, state, torch, *, measure=True):
    kwargs = {"prompt": prompt}
    if state is not None:
        kwargs["state"] = state
    torch.cuda.synchronize()
    started = time.perf_counter() if measure else None
    actions = model.predict(snapshot["images"], **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 if measure else None
    actions = validate_actions(actions, model._pipe.chunk_size)
    return elapsed_ms, list(actions.shape)

def _save_summary(run_dir, summary):
    eval_q._write_json_atomic(run_dir / "summary.json", summary)
    rows = [
        "# State 推理延迟对比", "", f"- status: `{summary['status']}`",
        f"- run directory: `{run_dir}`",
        f"- task: {summary.get('task_name', 'pending')} (`{summary.get('task_id', 0)}`), init 0, settle 10",
        f"- sync diagnostic; H={summary['config_snapshot']['H']}, N=10, BF16, two views; action shape: `{summary.get('action_shape', 'pending')}`",
        "- order: `no_state` → `fixed_state` → `changing_state`; each path runs contiguously.",
        "- each path has 2 unclocked warmups followed by 5 measured calls; seed 7 resets before every call.",
        "- timed boundary: `predict()` through CUDA sync; excludes model load, input preparation, and all warmups.",
        "- `fixed_state` reuses one observed normalized 8D state. `changing_state` uses fixed images and copies that normalized state, adding `index × 0.02` to the first component without clipping; all other components retain their observed values. This is synthetic input, not an actual robot state sequence or a closed-loop run.", "",
        "## Path measurements", "",
        "| path | warmups | measured calls | mean ms | median ms | raw ms |",
        "|---|---:|---:|---:|---:|---|",
    ]
    stats = summary.get("statistics")
    if stats:
        for key, label in (("no_state", "no_state"), ("fixed_state", "fixed_state"),
                           ("changing_state", "changing_state")):
            item = stats[key]
            raw = ", ".join("{:.3f}".format(x) for x in item["raw_ms"])
            rows.append(f"| {label} | {WARMUPS} | {len(item['raw_ms'])} | {item['mean_ms']:.3f} | {item['median_ms']:.3f} | {raw} |")
        differences = summary.get("mean_differences", {})
        rows.extend(("", "## Differences of path means", ""))
        for key, label, denominator in (
            ("fixed_state_minus_no_state", "fixed_state − no_state", "no_state"),
            ("changing_state_minus_fixed_state", "changing_state − fixed_state", "fixed_state"),
        ):
            item = differences.get(key)
            if item:
                rows.append(
                    "- {}: {:+.3f} ms ({:+.2f}% vs {} mean).".format(
                        label, item["mean_ms"], item["mean_percent_vs_baseline"], denominator
                    )
                )
        rows.append("- Paths were measured sequentially and are not paired; no per-call deltas are reported.")
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
            "schema_version": 2, "status": "running", "purpose": "state-latency",
            "run_mode": "synchronous_diagnostic", "run_dir": str(run_dir),
            "config_snapshot": config, "state_enabled_config_unchanged": config["state_enabled"],
            "base_config_snapshot": base_config,
            "diagnostic_horizon_override": args.horizon,
            "task_q_evaluation_started": False,
            "task_id": 0, "init_state_index": 0, "settle_steps": 10,
            "control_freq_hz": 20, "seed": SEED, "warmups_per_path": WARMUPS,
            "measurements_per_path": MEASUREMENTS,
            "path_order": ["no_state", "fixed_state", "changing_state"],
            "state_protocol": "8D checkpoint q01/q99 normalization; extra prompt tokens",
            "normalization": "(x-q01)/(q99-q01+1e-6)*2-1",
            "timing_boundary": "wall time around predict() through torch.cuda.synchronize",
            "excluded_from_measurement": ["model load", "input preparation", "all unclocked warmups"],
            "comparison_note": "Paths are measured sequentially, not as paired calls; only differences of path means are computed.",
            "egl_environment": {k: os.environ[k] for k in (
                "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "PYOPENGL_PLATFORM")},
            "egl_cleanup": egl_cleanup,
            "measurements": {"no_state": [], "fixed_state": [], "changing_state": []},
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

        if snapshot["state"].shape != (8,):
            raise ValueError("changing_state 要求原始 observation 对应 8D normalized state")
        synthetic_base = snapshot["state"].copy()
        synthetic_states = []
        for index in range(WARMUPS + MEASUREMENTS):
            state = synthetic_base.copy()
            state[0] = synthetic_base[0] + index * 0.02
            synthetic_states.append(state)
        if any(np.array_equal(left, right)
               for left, right in zip(synthetic_states, synthetic_states[1:])):
            raise RuntimeError("changing_state 合成 state 必须在每次调用间发生变化")
        summary["changing_state_protocol"] = {
            "description": "synthetic perturbations of one observed normalized state with fixed images; not an actual robot state sequence and not a closed-loop run",
            "synthetic_base": "copy of snapshot['state']",
            "first_component_formula": "base[0] + index * 0.02; index spans all warmups and measurements; no clipping",
            "first_component_step": 0.02,
            "warmup_first_components": [float(state[0]) for state in synthetic_states[:WARMUPS]],
            "measured_first_components": [float(state[0]) for state in synthetic_states[WARMUPS:]],
            "other_components": "copied unchanged from snapshot['state']",
        }
        summary["fixed_state_protocol"] = {
            "description": "same normalized state from the single prepared observation on every call",
            "first_component": float(snapshot["state"][0]),
        }

        paths = (
            ("no_state", [None] * WARMUPS, [None] * MEASUREMENTS),
            ("fixed_state", [snapshot["state"]] * WARMUPS,
             [snapshot["state"]] * MEASUREMENTS),
            ("changing_state", synthetic_states[:WARMUPS], synthetic_states[WARMUPS:]),
        )
        for label, warmup_states, measured_states in paths:
            for state in warmup_states:
                eval_q._seed_process(SEED, np, torch)
                _predict(model, snapshot, task["instruction"], state, torch, measure=False)
            summary.setdefault("graph_paths", {})[label] = _graph_info(pipe, warmup_states[0] is not None)
            for sample, state in enumerate(measured_states, start=1):
                eval_q._seed_process(SEED, np, torch)
                elapsed, shape = _predict(model, snapshot, task["instruction"], state, torch)
                summary["action_shape"] = shape
                summary["measurements"][label].append({
                    "sample": sample, "elapsed_ms": elapsed, "action_shape": shape,
                    **({"state_first_component": float(state[0])} if state is not None else {}),
                })
            _save_summary(run_dir, summary)

        stats = {}
        for label, samples in summary["measurements"].items():
            raw_ms = [sample["elapsed_ms"] for sample in samples]
            stats[label] = {
                "raw_ms": raw_ms,
                "mean_ms": statistics.mean(raw_ms),
                "median_ms": statistics.median(raw_ms),
            }
        fixed_vs_no_state = stats["fixed_state"]["mean_ms"] - stats["no_state"]["mean_ms"]
        changing_vs_fixed = stats["changing_state"]["mean_ms"] - stats["fixed_state"]["mean_ms"]
        summary.update(
            status="completed", statistics=stats,
            mean_differences={
                "fixed_state_minus_no_state": {
                    "mean_ms": fixed_vs_no_state,
                    "mean_percent_vs_baseline": fixed_vs_no_state / stats["no_state"]["mean_ms"] * 100.0,
                },
                "changing_state_minus_fixed_state": {
                    "mean_ms": changing_vs_fixed,
                    "mean_percent_vs_baseline": changing_vs_fixed / stats["fixed_state"]["mean_ms"] * 100.0,
                },
            },
        )
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
