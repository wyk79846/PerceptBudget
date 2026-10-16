#!/usr/bin/env python3
"""Run fixed-q LIBERO quality/async evaluation or inspect the host environment."""

import argparse
import importlib
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perceptbudget.control import AsyncActionController, SynchronousActionController, SUPPORTED_Q
from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import (
    load_libero_norm_stats,
    prepare_libero_observation,
)
from perceptbudget.model import load_model, validate_actions
from perceptbudget.power import PowerMonitor


CONTROL_FREQ_HZ = 20
MAX_STEPS = 220
SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300}
SETTLE_STEPS = 10
RENDER_GPU_DEVICE_ID = -1
INIT_STATE_INDEX = 0
SEED = 7
POWER_POLL_SECONDS = 0.1
PACE_SLEEP_SLICE_SECONDS = 0.002
BOOTSTRAP_SLEEP_SECONDS = 0.002
REQUIRED_NATIVE_MODULES = (
    "flash_rt.flash_rt_kernels",
    "flash_rt.flash_rt_fa2",
)
STATE_PROTOCOL_NOTE = (
    "state_enabled=true 时，8维 checkpoint q01/q99 归一化 state 经 FlashRT "
    "state 参数进入额外 prompt tokens；OpenPI pi05_libero 的 "
    "discrete_state_input=False，pi05 没有 continuous state projection。"
    "因此这是额外 state-prompt 实验，不代表 OpenPI 官方默认 state 输入。"
)
EGL_CLEANUP_REASON = (
    "Matches FlashRT examples/thor/eval_libero.py: repeated env.close() may corrupt "
    "CUDA memory on Jetson unified-memory systems; destructor patch is process-local."
)


def _write_json_atomic(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + "-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("无法写入 JSON 的值：" + type(value).__name__)


def _trace(stream, event, **fields):
    record = {"event": event, **fields}
    stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    stream.flush()


def _save_run_status(run_dir, status, **fields):
    path = Path(run_dir) / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(status=status, **fields)
    _write_json_atomic(path, record)


def _save_summary(run_dir, summary):
    _write_json_atomic(Path(run_dir) / "summary.json", summary)
    rows = [
        "# LIBERO 固定 q " + ("同步质量评测" if summary.get("execution_mode") == "synchronous_quality" else "异步评测"),
        "",
        "- run status: `{}`".format(summary.get("status")),
        "- mode: `{}`".format(summary.get("mode")),
        "- episodes: {} / {}".format(
            len(summary.get("episodes", [])), summary.get("expected_episodes")
        ),
        "- state config: `{}`".format(summary.get("state_enabled")),
        "- state protocol: {}".format(STATE_PROTOCOL_NOTE),
        "- run directory: `{}`".format(run_dir),
        "",
        "| task | q | result | steps | simulated s | real s | infer duty | power coverage | covered mJ | total mJ |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for episode in summary.get("episodes", []):
        power = episode.get("power") or {}
        rows.append(
            "| {} `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                episode.get("task_id", "?"),
                episode.get("task_name", "unknown"),
                episode.get("q", "?"),
                episode.get("status", "unknown"),
                episode.get("control_steps", 0),
                _fmt(episode.get("simulated_control_seconds")),
                _fmt(episode.get("real_task_seconds")),
                _fmt((episode.get("inference") or {}).get("inference_duty_fraction")),
                _fmt(power.get("coverage_fraction")),
                _fmt(power.get("covered_energy_mj")),
                _fmt(power.get("total_energy_mj")),
            )
        )
    if summary.get("error"):
        rows.extend(("", "## Error", "", "```text", summary["error"], "```"))
    Path(run_dir, "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fmt(value):
    return "—" if value is None else "{:.4f}".format(value)


def _plan(config, smoke):
    if config.get("model_id") != "pi05_libero_pytorch":
        raise ValueError("当前入口只支持 pi05_libero_pytorch")
    if config.get("H") != 20 or config.get("N") != 10:
        raise ValueError("当前固定评测要求 config H=20、N=10")
    if config.get("precision") != "bfloat16":
        raise ValueError("当前固定评测要求 config precision=bfloat16")
    if type(config.get("state_enabled")) is not bool:
        raise ValueError("config.state_enabled 必须由用户选为 true 或 false")

    if smoke:
        task_ids, q_values, mode = [0], [5], "smoke"
    else:
        screen = config.get("screen")
        if not isinstance(screen, dict):
            raise ValueError("config.screen 必须是对象")
        task_ids = screen.get("task_ids")
        q_values = config.get("candidate_q")
        suites = screen.get("suites", [screen.get("suite", "libero_spatial")])
        if (not isinstance(suites, list) or not suites
                or len(set(suites)) != len(suites)
                or any(s not in ("libero_spatial", "libero_object", "libero_goal") for s in suites)):
            raise ValueError("screen.suites 需为不重复的 Spatial/Object/Goal 集合列表")
        legacy = suites == ["libero_spatial"] and task_ids == list(range(9))
        if not legacy and task_ids != list(range(10 * len(suites))):
            raise ValueError("正式 screen 的 task_ids 需完整覆盖所选集合的全局任务编号")
        if screen.get("trials_per_task") != 1:
            raise ValueError("本轮每任务每q仅测一次")
        synchronous = config.get("execution_mode") == "synchronous_quality"
        if not isinstance(q_values, list) or not q_values:
            raise ValueError("candidate_q 必须为非空列表")
        mode = "synchronous_quality" if synchronous else "formal_screen"
    if any(type(task_id) is not int or task_id < 0 for task_id in task_ids):
        raise ValueError("task_ids 必须是非负整数")
    if len(set(q_values)) != len(q_values) or any(
        type(q) is not int or q not in SUPPORTED_Q or q > config["H"]
        for q in q_values
    ):
        raise ValueError("candidate_q 必须是 controller 支持且不大于 H 的唯一整数")
    if not smoke and mode != "synchronous_quality" and set(q_values) != set(SUPPORTED_Q):
        raise ValueError("正式 screen 的 q 集合必须为 {}".format(SUPPORTED_Q))
    return task_ids, q_values, mode


def _libero_api():
    # LIBERO / robosuite are intentionally imported only by an explicit main path.
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, OffScreenRenderEnv


def _patch_egl_cleanup():
    patched = []
    errors = []
    try:
        import robosuite.renderers.context.egl_context as egl
        egl.EGLGLContext.free = lambda self: None
        egl.EGLGLContext.__del__ = lambda self: None
        patched.extend(("EGLGLContext.free", "EGLGLContext.__del__"))
    except Exception as exc:
        errors.append("egl_context: {}: {}".format(type(exc).__name__, exc))
    try:
        import robosuite.utils.binding_utils as binding_utils
        binding_utils.MjRenderContext.__del__ = lambda self: None
        patched.append("MjRenderContext.__del__")
    except Exception as exc:
        errors.append("binding_utils: {}: {}".format(type(exc).__name__, exc))
    return {"reason": EGL_CLEANUP_REASON, "patched_methods": patched, "errors": errors}


def _load_tasks(task_ids, suite_names=None):
    benchmark, _ = _libero_api()
    suite_names = ["libero_spatial"] if suite_names is None else suite_names
    registry = benchmark.get_benchmark_dict()
    suites = []
    for name in suite_names:
        if name not in registry:
            raise RuntimeError("LIBERO 中没有注册 " + name)
        suite = registry[name](task_order_index=0)
        if suite.n_tasks != 10:
            raise RuntimeError(name + " 不包含预期的10个任务")
        suites.append(suite)
    tasks = []
    seen_names = set()
    for task_id in task_ids:
        if type(task_id) is not int or not 0 <= task_id < 10 * len(suites):
            raise ValueError("全局task_id超出所选集合范围")
        suite_index, local_id = divmod(task_id, 10)
        suite = suites[suite_index]
        task = suite.get_task(local_id)
        if task.name in seen_names:
            raise ValueError("任务名称重复：" + task.name)
        seen_names.add(task.name)
        initial_states = suite.get_task_init_states(local_id)
        if len(initial_states) <= INIT_STATE_INDEX:
            raise RuntimeError("task {} 没有 init state index 0".format(task_id))
        bddl_path = Path(suite.get_task_bddl_file_path(local_id))
        if not bddl_path.is_file():
            raise FileNotFoundError("LIBERO BDDL 不存在：" + str(bddl_path))
        tasks.append({
            "task_id": task_id,
            "suite": suite_names[suite_index],
            "suite_task_id": local_id,
            "max_steps": SUITE_MAX_STEPS[suite_names[suite_index]],
            "task_name": task.name,
            "instruction": task.language,
            "bddl_path": bddl_path,
            "init_state": initial_states[INIT_STATE_INDEX],
        })
    return tasks


def _make_env(env_class, task):
    return env_class(
        bddl_file_name=str(task["bddl_path"]),
        control_freq=CONTROL_FREQ_HZ,
        horizon=task.get("max_steps", MAX_STEPS) + SETTLE_STEPS,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        render_gpu_device_id=RENDER_GPU_DEVICE_ID,
    )


def _seed_process(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_env_episode(env, task, np):
    env.seed(SEED)
    env.reset()
    from perceptbudget.fixtures import enforce_fixture_layout
    enforce_fixture_layout(env)
    observation = env.set_init_state(task["init_state"])
    stable_action = np.zeros(7, dtype=np.float32)
    stable_action[6] = -1.0
    for _ in range(SETTLE_STEPS):
        result = env.step(stable_action)
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError("LIBERO env.step 应返回 (obs, reward, done, info)")
        observation, _, done, _ = result
        if done:
            raise RuntimeError("LIBERO 在正式计时前的稳定步意外结束")
    actual_frequency = float(env.env.control_freq)
    if not math.isclose(actual_frequency, CONTROL_FREQ_HZ, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            "LIBERO 实际 control_freq={}，要求 {}".format(
                actual_frequency, CONTROL_FREQ_HZ
            )
        )
    return observation, actual_frequency


def _make_snapshot(observation, norm_stats, state_enabled):
    return prepare_libero_observation(
        observation, norm_stats, state_enabled=state_enabled
    )


def _infer_actions(model, snapshot, prompt, horizon, timing):
    predict_args = {"prompt": prompt}
    if "state" in snapshot:
        predict_args["state"] = snapshot["state"]
    timing["inference_started_monotonic"] = time.monotonic()
    try:
        actions = model.predict(snapshot["images"], **predict_args)
    finally:
        timing["inference_ended_monotonic"] = time.monotonic()
    return validate_actions(actions, horizon)


def _check_cuda_and_native(config):
    from scripts.run_flashrt import select_flashrt

    select_flashrt(config["flashrt_root"], config["openpi_root"])
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；未加载 checkpoint")
    device_name = torch.cuda.get_device_name(0)
    native_paths = {}
    for module_name in REQUIRED_NATIVE_MODULES:
        module = importlib.import_module(module_name)
        native_paths[module_name] = getattr(module, "__file__", None)
    return torch, {"cuda_device": device_name, "native_modules": native_paths}


def _load_and_check_model(config, torch):
    model = load_model(config)
    pipe = getattr(model, "_pipe", None)
    if pipe is None:
        raise RuntimeError("load_model 返回值缺少原版 frontend _pipe")
    actual_horizon = getattr(pipe, "chunk_size", None)
    if actual_horizon != config["H"]:
        raise RuntimeError(
            "FlashRT frontend chunk_size={}，要求 H={}".format(
                actual_horizon, config["H"]
            )
        )
    if getattr(pipe, "_force_bf16", None) is not True:
        raise RuntimeError("FlashRT frontend 未确认 full BF16")
    if any(getattr(pipe, name, False) for name in (
        "_fp8_weights", "_int8_weights", "_use_int8_encoder", "_use_int8_vision"
    )):
        raise RuntimeError("FlashRT frontend 检测到非 BF16 权重量化路径")
    torch.cuda.synchronize()
    return model


def _power_result(monitor, start, stop):
    report = monitor.integrate_report(start, stop)
    result = {
        "covered_energy_mj": report.energy_mj,
        "coverage_fraction": report.coverage_fraction,
        "missing_count": report.missing_count,
        "total_energy_mj": report.total_energy_mj,
    }
    result["raw_readings"] = [
        {
            "timestamp_monotonic": reading.timestamp,
            "power_mw": reading.power_mw,
            "error": reading.error,
        }
        for reading in monitor.readings
    ]
    return result


def _run_episode(env, task, q, model, norm_stats, state_enabled, horizon, np,
                 torch, trace_stream, episode):
    monitor = None
    executor = None
    controller = None
    task_start = None
    power_stop = None
    task_error = None
    control_steps = 0
    actual_frequency = None
    max_steps = task.get("max_steps", MAX_STEPS)
    try:
        observation, actual_frequency = _prepare_env_episode(
            env, task, np
        )
        initial_snapshot = _make_snapshot(observation, norm_stats, state_enabled)
        episode.update(
            status="running",
            control_freq_hz=actual_frequency,
        )
        _trace(
            trace_stream, "episode_environment_ready", task_id=episode["task_id"],
            q=q, control_freq_hz=actual_frequency,
        )

        request_records = []
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-infer")
        period = 1.0 / actual_frequency

        def submit(snapshot, observation_step):
            request = {
                "observation_step": observation_step,
                "submitted_at_monotonic": time.monotonic(),
            }
            future = executor.submit(
                _infer_actions, model, snapshot, task["instruction"], horizon,
                request,
            )
            request["future"] = future
            request_records.append(request)
            return future

        synchronous = episode.get("execution_mode") == "synchronous_quality"
        controller_class = SynchronousActionController if synchronous else AsyncActionController
        controller = controller_class(
            submit,
            q=q,
            horizon=horizon,
            control_period_s=period,
        )
        monitor = PowerMonitor(poll_interval=POWER_POLL_SECONDS)
        monitor.start()
        task_start = PowerMonitor.mark()
        _trace(
            trace_stream, "episode_task_start", task_id=episode["task_id"], q=q,
            timestamp_monotonic=task_start,
            first_inference_included=True,
            power_window_starts_here=True,
        )

        # The per-task graph warmup may consume CUDA RNG. Reset immediately
        # before the timed bootstrap so every q starts from the same seed.
        _seed_process(SEED, np, torch)
        controller.bootstrap(initial_snapshot, observation_step=0)
        # Keep the Future returned by the public submit callback; the controller
        # deliberately exposes poll/inflight, but not its private Future object.
        bootstrap_future = request_records[-1]["future"]
        bootstrap_wait_start = time.monotonic()
        while not bootstrap_future.done():
            time.sleep(BOOTSTRAP_SLEEP_SECONDS)
        controller.poll()
        bootstrap_wait_seconds = time.monotonic() - bootstrap_wait_start
        episode["bootstrap_wait_seconds"] = bootstrap_wait_seconds
        _trace(
            trace_stream, "bootstrap_ready", task_id=episode["task_id"], q=q,
            wait_seconds=bootstrap_wait_seconds,
            prediction_shape=[horizon, 7],
            control_steps_started=False,
        )

        next_deadline = time.monotonic()
        last_gripper = -1.0
        current_observation = observation
        terminal_status = None
        deadline_misses = 0
        pacing_sleep_seconds = 0.0
        while control_steps < max_steps:
            step = controller.control_step
            cycle_start = time.monotonic()
            snapshot = _make_snapshot(current_observation, norm_stats, state_enabled)
            before_dropped = controller.dropped_actions
            request_submitted = controller.request(
                snapshot, observation_step=step
            )
            predicted_action = controller.pop()
            underflow = predicted_action is None
            if underflow:
                action = np.zeros(7, dtype=np.float32)
                action[6] = last_gripper
            else:
                action = np.asarray(predicted_action, dtype=np.float32).copy()
                last_gripper = float(action[6])

            result = env.step(action)
            if not isinstance(result, tuple) or len(result) != 4:
                raise RuntimeError("LIBERO env.step 应返回 (obs, reward, done, info)")
            current_observation, reward, env_done, _info = result
            control_steps += 1
            episode["control_steps"] = control_steps
            success = bool(env.check_success())
            if success:
                terminal_status = "success"
            elif control_steps >= max_steps:
                terminal_status = "timeout"
            elif env_done:
                terminal_status = "failure"

            cycle_end = time.monotonic()
            deadline_missed = False
            step_sleep_seconds = 0.0
            if terminal_status is None:
                next_deadline += period
                now = time.monotonic()
                deadline_missed = now > next_deadline
                if deadline_missed:
                    deadline_misses += 1
                    # Re-anchor without adding another full period to an overdue step.
                    next_deadline = now
                while True:
                    remaining = next_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    sleep_for = min(PACE_SLEEP_SLICE_SECONDS, remaining)
                    time.sleep(sleep_for)
                    step_sleep_seconds += sleep_for
                    pacing_sleep_seconds += sleep_for
            record = {
                "task_id": episode["task_id"],
                "q": q,
                "control_step": step,
                "raw_state": snapshot["raw_state"].tolist(),
                "state_passed_to_model": state_enabled,
                "request_submitted": request_submitted,
                "action_underflow": underflow,
                "executed_action": action.tolist(),
                "dropped_action_delta": controller.dropped_actions - before_dropped,
                "dropped_actions_total": controller.dropped_actions,
                "env_reward": reward,
                "env_done": bool(env_done),
                "env_success": success,
                "cycle_seconds": cycle_end - cycle_start,
                "control_step_end_monotonic": cycle_end,
                "deadline_missed": deadline_missed,
                "pacing_sleep_seconds": step_sleep_seconds,
            }
            _trace(trace_stream, "control_step", **record)

            if terminal_status is not None:
                break

        power_stop = PowerMonitor.mark()
        monitor.stop()
        task_seconds = power_stop - task_start
        episode.update(
            status=terminal_status,
            control_steps=control_steps,
            simulated_control_seconds=control_steps / actual_frequency,
            real_task_seconds=task_seconds,
            deadline_misses=deadline_misses,
            pacing_sleep_seconds=pacing_sleep_seconds,
            completed_at_monotonic=power_stop,
        )
        episode["power"] = _power_result(monitor, task_start, power_stop)
        _trace(
            trace_stream, "episode_terminal", task_id=episode["task_id"], q=q,
            status=terminal_status, control_steps=control_steps,
            real_task_seconds=task_seconds,
            simulated_control_seconds=control_steps / actual_frequency,
            deadline_misses=deadline_misses,
            power=episode["power"],
        )

        return episode
    except Exception as exc:
        task_error = exc
        episode.update(
            status="invalid",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        _trace(
            trace_stream, "episode_invalid", task_id=episode.get("task_id"), q=q,
            error_type=type(exc).__name__, error=str(exc),
        )
        raise
    finally:
        if monitor is not None and task_start is not None and power_stop is None:
            power_stop = PowerMonitor.mark()
            episode["control_steps"] = control_steps
            if actual_frequency is not None:
                episode["simulated_control_seconds"] = control_steps / actual_frequency
            try:
                monitor.stop()
                episode["real_task_seconds"] = power_stop - task_start
                episode["power"] = _power_result(monitor, task_start, power_stop)
                episode["completed_at_monotonic"] = power_stop
                _trace(
                    trace_stream, "episode_power_stopped", task_id=episode.get("task_id"),
                    q=q, power=episode["power"],
                    terminal_timestamp_monotonic=power_stop,
                    invalid=task_error is not None,
                )
            except Exception as cleanup_exc:
                episode["power_error"] = str(cleanup_exc)
        if executor is not None:
            executor.shutdown(wait=True)
        if controller is not None:
            pending_inference_error = None
            try:
                # Drain the final Future too; prediction errors invalidate the run.
                controller.poll()
            except Exception as cleanup_exc:
                if task_error is None:
                    episode.update(
                        status="invalid", error_type=type(cleanup_exc).__name__,
                        error=str(cleanup_exc), traceback=traceback.format_exc(),
                    )
                    _trace(
                        trace_stream, "episode_invalid", task_id=episode.get("task_id"),
                        q=q, error_type=type(cleanup_exc).__name__, error=str(cleanup_exc),
                    )
                    pending_inference_error = cleanup_exc
                episode["pending_inference_error"] = str(cleanup_exc)
            request_steps = [item["observation_step"] for item in request_records]
            requests = []
            inference_busy_seconds = 0.0
            for item in request_records:
                started = item.get("inference_started_monotonic")
                ended = item.get("inference_ended_monotonic")
                request = {
                    "observation_step": item["observation_step"],
                    "submitted_monotonic": item["submitted_at_monotonic"],
                    "inference_started_monotonic": started,
                    "inference_ended_monotonic": ended,
                }
                if started is not None and ended is not None:
                    request["inference_seconds"] = ended - started
                    if task_start is not None and power_stop is not None:
                        inference_busy_seconds += max(
                            0.0, min(ended, power_stop) - max(started, task_start)
                        )
                requests.append(request)
            task_seconds = episode.get("real_task_seconds")
            episode["inference"] = {
                "submitted_requests": controller.submitted_requests,
                "failed_requests": controller.failed_requests,
                "requests": requests,
                "busy_skipped_trigger_steps": list(controller.skipped_trigger_steps),
                "effective_refresh_intervals_steps": [
                    right - left for left, right in zip(request_steps, request_steps[1:])
                ],
                "effective_refresh_intervals_seconds": [
                    (right - left) * period
                    for left, right in zip(request_steps, request_steps[1:])
                ],
                "inference_busy_seconds_within_task": inference_busy_seconds,
                "inference_duty_fraction": (
                    inference_busy_seconds / task_seconds
                    if task_seconds and task_seconds > 0 else None
                ),
                "dropped_actions": controller.dropped_actions,
                "expired_actions": controller.expired_actions,
                "overwritten_unused_actions": controller.overwritten_actions,
                "underflow_steps": list(controller.underflow_steps),
                "buffer_summary_actions": controller.buffer_summary,
                "chunk_policy": ("synchronous q-step replanning; execute new prefix from zero"
                    if episode.get("execution_mode") == "synchronous_quality" else
                    "discard expired action prefixes by absolute control step; overlapping "
                    "actions may also be overwritten; this is not RTC"
                ),
            }
            _trace(
                trace_stream, "episode_inference_summary", task_id=episode["task_id"],
                q=q, inference=episode["inference"],
            )
            if pending_inference_error is not None:
                raise pending_inference_error


def _prepare_only(config, norm_stats, task_ids, egl_cleanup):
    import numpy as np

    _, env_class = _libero_api()
    tasks = _load_tasks(task_ids, config["screen"].get("suites"))
    task = tasks[0]
    env = _make_env(env_class, task)
    try:
        observation, actual_frequency = _prepare_env_episode(
            env, task, np
        )
        prepared = _make_snapshot(
            observation, norm_stats, config["state_enabled"]
        )
        report = {
            "status": "prepared_only",
            "cuda_model_loaded": False,
            "host_egl_rendering": True,
            "egl_cleanup": egl_cleanup,
            "render_gpu_device_id": RENDER_GPU_DEVICE_ID,
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "instruction": task["instruction"],
            "tasks": [
                {key: item[key] for key in ("task_id", "suite", "suite_task_id", "task_name", "instruction")}
                for item in tasks
            ],
            "control_freq_requested_hz": CONTROL_FREQ_HZ,
            "control_freq_actual_hz": actual_frequency,
            "max_steps": MAX_STEPS,
            "env_horizon_including_settle": MAX_STEPS + SETTLE_STEPS,
            "observation_shapes": {
                "agentview_image": list(observation["agentview_image"].shape),
                "robot0_eye_in_hand_image": list(
                    observation["robot0_eye_in_hand_image"].shape
                ),
                "raw_state": list(prepared["raw_state"].shape),
                "model_images": [list(image.shape) for image in prepared["images"]],
                "model_state": list(prepared["state"].shape)
                if config["state_enabled"] else None,
            },
            "state_enabled_from_config": config["state_enabled"],
            "state_input_mode": "extra_state_prompt_experiment"
            if config["state_enabled"] else "no_state_argument",
            "state_protocol_note": STATE_PROTOCOL_NOTE,
            "formal_eval_started": False,
        }
        if not math.isclose(
            report["control_freq_actual_hz"], CONTROL_FREQ_HZ,
            rel_tol=0.0, abs_tol=1e-9,
        ):
            raise RuntimeError("prepare-only 检测到实际控制频率不等于 20 Hz")
        return report
    finally:
        env.close()


def _run_screen(config, tasks, q_values, mode, norm_stats, torch, cuda_report,
                run_dir, trace_stream, summary):
    import numpy as np

    model_started = time.monotonic()
    model = _load_and_check_model(config, torch)
    summary["model"] = {
        **cuda_report,
        "loaded_once": True,
        "load_seconds": time.monotonic() - model_started,
        "requested_horizon": config["H"],
        "actual_horizon": model._pipe.chunk_size,
        "requested_denoising_steps": config["N"],
        "requested_precision": config["precision"],
        "full_bf16_confirmed": True,
    }
    _trace(trace_stream, "model_loaded", **summary["model"])

    episodes = summary["episodes"]
    _, env_class = _libero_api()
    for task in tasks:
        print(
            "task warmup: id={} name={} state_enabled={}".format(
                task["task_id"], task["task_name"], config["state_enabled"]
            ),
            flush=True,
        )
        task_env = None
        try:
            task_env = _make_env(env_class, task)
        except Exception as exc:
            _trace(
                trace_stream, "task_environment_invalid", task_id=task["task_id"],
                task_name=task["task_name"], error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        try:
            warmup_started = time.monotonic()
            warmup_timing = {}
            _trace(
                trace_stream, "task_prompt_warmup_start", task_id=task["task_id"],
                task_name=task["task_name"], state_enabled=config["state_enabled"],
                included_in_formal_time=False,
            )
            try:
                _seed_process(SEED, np, torch)
                warmup_observation, _ = _prepare_env_episode(
                    task_env, task, np
                )
                warmup_snapshot = _make_snapshot(
                    warmup_observation, norm_stats, config["state_enabled"]
                )
                _infer_actions(
                    model, warmup_snapshot, task["instruction"], config["H"],
                    warmup_timing,
                )
                warmup = {
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "warmup_seconds": time.monotonic() - warmup_started,
                    **warmup_timing,
                }
                summary["task_warmups"].append(warmup)
                _trace(trace_stream, "task_prompt_warmup_done", **warmup)
            except Exception as exc:
                _trace(
                    trace_stream, "task_prompt_warmup_invalid",
                    task_id=task["task_id"], task_name=task["task_name"],
                    error_type=type(exc).__name__, error=str(exc),
                    **warmup_timing,
                )
                raise

            for q in q_values:
                print(
                    "episode {}/{}: task={} ({}) q={}".format(
                        len(episodes) + 1, summary["expected_episodes"],
                        task["task_id"], task["task_name"], q,
                    ),
                    flush=True,
                )
                episode = {
                    "task_id": task["task_id"],
                    "suite": task.get("suite", "libero_spatial"),
                    "suite_task_id": task.get("suite_task_id", task["task_id"]),
                    "execution_mode": config.get("execution_mode", "asynchronous"),
                    "max_steps": task.get("max_steps", MAX_STEPS),
                    "task_name": task["task_name"],
                    "instruction": task["instruction"],
                    "q": q,
                    "mode": mode,
                    "status": "created",
                }
                episodes.append(episode)
                _trace(
                    trace_stream, "episode_start", task_id=task["task_id"],
                    task_name=task["task_name"], instruction=task["instruction"], q=q,
                    mode=mode,
                )
                try:
                    _run_episode(
                        task_env, task, q, model, norm_stats,
                        config["state_enabled"], config["H"], np, torch,
                        trace_stream, episode,
                    )
                finally:
                    _save_summary(run_dir, summary)
        finally:
            task_env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="只跑 task 0、q=5 的一条独立冒烟评测",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="读取 LIBERO env/input 形状（需要主机 EGL 离屏渲染）；不加载模型权重",
    )
    args = parser.parse_args(argv)
    if args.smoke and args.prepare_only:
        parser.error("--smoke 与 --prepare-only 不能同时使用")

    try:
        config = load_config(repo_root=PROJECT_ROOT)
        task_ids, q_values, mode = _plan(config, args.smoke)
        norm_stats = load_libero_norm_stats(config["norm_stats_path"])
        if config["state_enabled"]:
            # Validate exact checkpoint state dimensions before any model allocation.
            state_stats = norm_stats.get("state", {})
            if len(state_stats.get("q01", [])) != 8 or len(state_stats.get("q99", [])) != 8:
                raise ValueError("启用 state 时 checkpoint state q01/q99 必须恰为 8 维")
        # Set rendering backends before importing LIBERO, robosuite, or OpenGL.
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        egl_cleanup = _patch_egl_cleanup()
        if args.prepare_only:
            print(json.dumps(
                _prepare_only(config, norm_stats, task_ids, egl_cleanup),
                ensure_ascii=False, indent=2,
            ))
            return 0
    except Exception as exc:
        print("准备失败：{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1

    purpose = "smoke" if args.smoke else ("q-sync" if mode == "synchronous_quality" else "q-screen")
    try:
        run_dir = create_run(config, "M1", purpose, repo_root=PROJECT_ROOT)
    except Exception as exc:
        print("创建 run 失败：{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1

    summary = {
        "schema_version": 1,
        "status": "running",
        "mode": mode,
        "run_dir": str(run_dir),
        "expected_episodes": len(task_ids) * len(q_values),
        "task_ids": task_ids,
        "q_values": q_values,
        "state_enabled": config["state_enabled"],
        "state_input_mode": "extra_state_prompt_experiment"
        if config["state_enabled"] else "no_state_argument",
        "state_protocol_note": STATE_PROTOCOL_NOTE,
        "host_egl_rendering": True,
        "egl_environment": {
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "PYOPENGL_PLATFORM": "egl",
        },
        "egl_cleanup": egl_cleanup,
        "render_gpu_device_id": RENDER_GPU_DEVICE_ID,
        "control_freq_hz": CONTROL_FREQ_HZ,
        "max_steps_by_suite": SUITE_MAX_STEPS,
        "env_horizon_rule": "suite max_steps + 10 settle steps",
        "execution_mode": config.get("execution_mode", "asynchronous"),
        "seed": SEED,
        "init_state_index": INIT_STATE_INDEX,
        "settle_steps_outside_timing": SETTLE_STEPS,
        "bootstrap_first_inference_in_task_time_and_power": True,
        "task_prompt_warmup": "same environment reused per task; reset before each q; excluded from formal time and power",
        "power_rail": "INA3221 VDD_IN module input; not GPU-only",
        "power_window": "task start through environment terminal; integrate_report clips raw bracketing samples",
        "underflow_action": "zero pose delta and hold previous gripper; initial held gripper is -1",
        "pacing": "short sleeps; deadline misses recorded; no catch-up step bursts; not hard real-time",
        "episodes": [],
        "task_warmups": [],
    }
    trace_path = run_dir / "trace.jsonl"
    try:
        _save_run_status(
            run_dir, "running", mode=mode,
            state_enabled=config["state_enabled"],
        )
        with trace_path.open("a", encoding="utf-8") as trace_stream:
            _trace(
                trace_stream, "run_started", mode=mode,
                expected_episodes=summary["expected_episodes"],
                task_ids=task_ids, q_values=q_values,
                state_enabled=config["state_enabled"],
                egl_cleanup=egl_cleanup,
            )
            tasks = _load_tasks(task_ids, config["screen"].get("suites"))
            summary["tasks"] = [
                {key: task[key] for key in ("task_id", "suite", "suite_task_id", "task_name", "instruction")}
                for task in tasks
            ]
            print("run_dir=" + str(run_dir), flush=True)
            for task in tasks:
                print(
                    "verified LIBERO mapping: task_id={} name={} instruction={!r}".format(
                        task["task_id"], task["task_name"], task["instruction"]
                    ),
                    flush=True,
                )
            torch, cuda_report = _check_cuda_and_native(config)
            summary["cuda_preflight"] = cuda_report
            _run_screen(
                config, tasks, q_values, mode, norm_stats, torch,
                cuda_report, run_dir, trace_stream, summary,
            )
        if len(summary["episodes"]) != summary["expected_episodes"]:
            raise RuntimeError("screen 未完成预期 episode 数")
        summary.update(status="completed", finished_at_monotonic=PowerMonitor.mark())
        _save_summary(run_dir, summary)
        _save_run_status(
            run_dir, "completed", mode=mode,
            completed_episodes=len(summary["episodes"]),
        )
        print("评测完成：{} episodes；结果目录 {}".format(
            len(summary["episodes"]), run_dir
        ), flush=True)
        return 0
    except Exception as exc:
        summary.update(
            status="invalid",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback=traceback.format_exc(),
        )
        try:
            _save_summary(run_dir, summary)
            _save_run_status(
                run_dir, "invalid", mode=mode,
                error=summary["error"],
                completed_episodes=len(summary["episodes"]),
            )
        except Exception as write_error:
            print("无法保存 invalid 状态：{}".format(write_error), file=sys.stderr)
        traceback.print_exc()
        print("评测已停止并标记 invalid：" + str(run_dir), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
