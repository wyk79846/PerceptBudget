#!/usr/bin/env python3
"""One-minute inference on/off power calibration; keeps simulation rendering alive."""

import gc
import json
import fcntl
import os
from pathlib import Path
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from scripts.probe_delay_buffer import device_state
from scripts.run_async_q import file_sha256
from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import load_libero_norm_stats
from perceptbudget.power import PowerMonitor


def block_schedule():
    return ['idle', 'busy'] * 3 + ['idle']


def measurement_summary(readings, blocks):
    import numpy as np
    from scripts.analyze_opportunity import _integral
    measured = []
    for block in blocks:
        # Keep transitions visible in raw data, use middle of long windows for averages.
        start, stop = block['start_monotonic'] + 1., block['stop_monotonic'] - .5
        if stop <= start:
            raise ValueError('Power windows too short')
        report = _integral(readings, [(start, stop)])
        measured.append(dict(block, measured_start=start, measured_stop=stop, power=report))
    grouped = {}
    for mode in ('idle', 'busy'):
        rows = [b['power'] for b in measured if b['mode'] == mode]
        duration = sum(b['duration_seconds'] for b in rows)
        complete = all(b['full_coverage'] for b in rows)
        energy = sum(b['total_energy_j'] for b in rows) if complete else None
        grouped[mode] = dict(windows=len(rows), duration_seconds=duration,
                             mean_power_w=energy/duration if energy is not None else None,
                             energy_j=energy, full_coverage=complete)
    on, idle = grouped['busy']['mean_power_w'], grouped['idle']['mean_power_w']
    valid = [r for r in readings if r.power_mw is not None]
    changes = [b.timestamp for a, b in zip(valid, valid[1:]) if a.power_mw != b.power_mw]
    intervals = np.diff(changes).tolist()
    return dict(blocks=measured, grouped=grouped, P_on_w=on, P_idle_w=idle,
                r=idle/on if idle is not None and on else None,
                samples=len(readings), missing_samples=sum(r.power_mw is None for r in readings),
                observed_change_interval_median_s=float(np.median(intervals)) if intervals else None,
                change_interval_note='Observed value changes, not an exact hardware update period')


def run(root, config):
    import numpy as np
    monitor, env = None, None
    summary = dict(status='loading', config=config, blocks=[],
        power_rail='INA3221 VDD_IN module input',
        scope='Same loaded model and scene; zero arm commands, open gripper; simulation/render/control continue in both modes',
        model_input='Fixed double-view snapshot; full model computation, no KV reuse',
        environment_horizon_steps=2010,
        requested_block_seconds=8, requested_control_period_s=.05,
        poll_interval_seconds=.02, device_before=device_state())
    base._write_json_atomic(root/'summary.json', summary)
    try:
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        torch, cuda = base._check_cuda_and_native(config)
        stats = load_libero_norm_stats(config['norm_stats_path'])
        before_load = time.monotonic()
        model = base._load_and_check_model(config, torch)
        summary.update(cuda=cuda, load_seconds=time.monotonic()-before_load,
                       checkpoint_sha256=file_sha256(config['model_path']))
        task = base._load_tasks([0], config['screen']['suites'])[0]
        _, env_class = base._libero_api()
        # Power windows, not a task trial: allow the same stationary scene for a minute.
        env = base._make_env(env_class, dict(task, max_steps=2000))
        obs, _ = base._prepare_env_episode(env, task, np)
        snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
        base._infer_actions(model, snapshot, task['instruction'], config['H'], {})
        action = np.zeros(7, dtype=np.float32)
        action[6] = -1
        monitor = PowerMonitor(poll_interval=.02)
        monitor.start()
        requests = []
        summary.update(status='running', warmup_excluded=True)
        base._write_json_atomic(root/'summary.json', summary)
        with ThreadPoolExecutor(max_workers=1) as executor:
            for index, mode in enumerate(block_schedule()):
                stop_event = threading.Event()
                block = dict(index=index, mode=mode, start_monotonic=time.monotonic(), env_steps=0)
                print(f'block={index+1}/7 mode={mode}', flush=True)
                def predict_loop():
                    while not stop_event.is_set():
                        timing = {'block_index': index}
                        base._infer_actions(model, snapshot, task['instruction'], config['H'], timing)
                        requests.append(timing)
                future = executor.submit(predict_loop) if mode == 'busy' else None
                deadline = time.monotonic()
                try:
                    while time.monotonic() - block['start_monotonic'] < 8.:
                        obs, _, _, _ = env.step(action)
                        block['env_steps'] += 1
                        if future is not None and future.done():
                            future.result()
                            raise RuntimeError('Continuous inference stopped unexpectedly')
                        deadline += .05
                        now = time.monotonic()
                        if now > deadline:
                            deadline = now
                        else:
                            time.sleep(deadline-now)
                finally:
                    stop_event.set()
                    if future is not None:
                        future.result()
                block.update(stop_monotonic=time.monotonic(), device=device_state())
                block['control_hz'] = block['env_steps']/(block['stop_monotonic']-block['start_monotonic'])
                summary['blocks'].append(block)
                base._write_json_atomic(root/'summary.json', summary)
        monitor.stop()
        summary.update(measurement_summary(monitor.readings, summary['blocks']),
                       requests=requests, status='completed', device_after=device_state(),
                       raw_readings=[dict(timestamp_monotonic=r.timestamp, power_mw=r.power_mw, error=r.error)
                                     for r in monitor.readings])
        summary['limitations'] = [
            'Fixed-scene repeated inference, not a task-success test; model remains resident during idle.',
            'Both modes keep simulation/render/control alive; realized control rates are reported and may differ under contention.',
            'The first 1 s and last 0.5 s of each block are excluded from long-window means; transitions remain in raw data.',
            '20 ms polling is not a claim of 50 Hz hardware sampling; no per-inference energy precision is claimed.',
            'No frequency or power-mode changes; P_idle is not a low-frequency or sleep-state measurement.']
        print(json.dumps({k: summary[k] for k in ('P_on_w', 'P_idle_w', 'r', 'samples', 'missing_samples')}), flush=True)
    except BaseException as exc:
        summary.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if monitor is not None:
            monitor.stop()
            summary['raw_readings'] = [dict(timestamp_monotonic=r.timestamp, power_mw=r.power_mw, error=r.error)
                                       for r in monitor.readings]
        base._write_json_atomic(root/'summary.json', summary)
        if env is not None:
            env.close()
            gc.collect()


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    config = load_config(ROOT/'config.json')
    config.update(execution_mode='m12_power_calibration')
    root = create_run(config, 'M1', 'm12-power-calibration', repo_root=ROOT)
    print('run_dir='+str(root), flush=True)
    (root/'source').mkdir()
    (root/'source'/'probe_power_states.py').write_bytes(Path(__file__).read_bytes())
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, config)


if __name__ == '__main__':
    main()
