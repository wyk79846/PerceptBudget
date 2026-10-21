#!/usr/bin/env python3
"""Isolate LIBERO environment residency without loading model weights."""
import fcntl
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import ctypes
    import numpy as np
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from perceptbudget.device_quality import install_render_buffer_cleanup, SUITES
    from perceptbudget.rtc_memory import memory_record
    from scripts import eval_q as base
    from scripts.complete_m1_h50 import calibrate
    config = load_config(repo_root=ROOT)
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                      PYOPENGL_PLATFORM='egl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    run = create_run(config, 'M2', 'environment-memory')
    print('run_dir='+str(run), flush=True)
    record = dict(status='running', rows=[], model_loads=0)
    def save():
        _write_json_atomically(run/'summary.json', record)
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch, _ = base._check_cuda_and_native(config)
        base._patch_egl_cleanup()
        record['render_cleanup'] = install_render_buffer_cleanup(shared_egl=True)
        _, cls = base._libero_api()
        def sample(stage, cycle):
            gc.collect()
            ctypes.CDLL(None).malloc_trim(0)
            row = memory_record(torch, stage)
            row['cycle'] = cycle
            record['rows'].append(row)
            save()
            print(json.dumps(row), flush=True)
            if row['available_mib'] < 900:
                raise RuntimeError('RAM below900MiB')
        env = None
        try:
            sample('start', -1)
            for cycle in range(9):
                task = base._load_tasks([0], [SUITES[cycle % 3]])[0]
                base._seed_process(7, np, torch)
                env = base._make_env(cls, task)
                env._pb_lock_fixtures = True
                sample('constructed', cycle)
                observation, _ = calibrate(env, task, np)
                sample('calibrated', cycle)
                observation, _ = base._prepare_env_episode(env, task, np)
                sample('episode_reset', cycle)
                env.close()
                env = observation = None
                sample('closed', cycle)
            record['status'] = 'completed'
        except BaseException as exc:
            record.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if env is not None:
                env.close()
            save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
