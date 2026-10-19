"""Short no-model env.step profile; process-local timers, no dependency edits."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from contextlib import nullcontext, contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base


def blas_libraries():
    import ctypes
    paths = {s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
             if 'libopenblas' in s and '/' in s}
    libraries = []
    for path in sorted(paths):
        lib = ctypes.CDLL(path)
        for suffix in ('', '64_'):
            get = getattr(lib, 'openblas_get_num_threads'+suffix, None)
            set_ = getattr(lib, 'openblas_set_num_threads'+suffix, None)
            if get is not None and set_ is not None:
                get.restype = ctypes.c_int
                set_.argtypes = [ctypes.c_int]
                libraries.append((path, get, set_))
                break
    return libraries


def threadpool_info():
    return [dict(path=path, threads=get()) for path,get,_ in blas_libraries()]


@contextmanager
def one_blas_thread():
    saved = [(get(), set_) for _,get,set_ in blas_libraries()]
    try:
        for _,set_ in saved: set_(1)
        yield
    finally:
        for count,set_ in saved: set_(count)


def main():
    config = load_config(repo_root=ROOT)
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'env-step-profile', repo_root=ROOT)
        shutil.copy2(__file__, run/'probe_env_step.py')
        print('run_dir='+str(run), flush=True)
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        base._patch_egl_cleanup()
        import numpy as np
        task = base._load_tasks([0], ['libero_spatial'])[0]
        _, env_type = base._libero_api()
        env = base._make_env(env_type, task)
        env._pb_lock_fixtures = True
        result = dict(status='running', model_loaded=False, threadpools=threadpool_info(), modes=[])
        try:
            inner = env.env
            result.update(physics_timestep=inner.model_timestep, control_timestep=inner.control_timestep)
            action = np.zeros(7); action[6] = -1
            final_states = []
            trajectories = []
            for mode in ('default', 'blas_one_thread', 'reuse_forward_in_control'):
                context = one_blas_thread() if mode == 'blas_one_thread' else nullcontext()
                with context:
                    base._prepare_env_episode(env, task, np)
                    # Warm JIT/render/controller paths, outside profile.
                    for _ in range(5): env.step(action)
                    samples = {}
                    stack = []
                    originals = []
                    optimization_originals = []
                    skipped = [0]
                    if mode == 'reuse_forward_in_control':
                        original_forward, original_pre = inner.sim.forward, inner._pre_action
                        in_control = [False]
                        def cached_forward(*args, **kwargs):
                            if in_control[0]:
                                skipped[0] += 1
                                return None
                            return original_forward(*args, **kwargs)
                        def control_without_second_forward(*args, **kwargs):
                            in_control[0] = True
                            try: return original_pre(*args, **kwargs)
                            finally: in_control[0] = False
                        optimization_originals = [(inner.sim, 'forward', original_forward),
                                                  (inner, '_pre_action', original_pre)]
                        inner.sim.forward, inner._pre_action = cached_forward, control_without_second_forward
                    def install(obj, attr, label):
                        native = getattr(obj, attr)
                        originals.append((obj, attr, native))
                        def wrapped(*args, **kwargs):
                            frame = [time.perf_counter(), 0.]
                            stack.append(frame)
                            try:
                                return native(*args, **kwargs)
                            finally:
                                elapsed = time.perf_counter()-frame[0]
                                stack.pop()
                                samples.setdefault(label, []).append((elapsed-frame[1])*1000)
                                if stack: stack[-1][1] += elapsed
                        setattr(obj, attr, wrapped)
                    install(inner, '_pre_action', 'control_dispatch')
                    install(inner.sim, 'forward', 'sim_forward')
                    install(inner.sim, 'step', 'physics_step')
                    install(inner.sim, 'render', 'camera_render')
                    for i, robot in enumerate(inner.robots):
                        install(robot.controller, 'run_controller', f'controller{i}')
                    wall = []
                    states, controls, image_hashes = [], [], []
                    try:
                        for _ in range(30):
                            started = time.perf_counter()
                            obs, _, _, _ = env.step(action)
                            wall.append((time.perf_counter()-started)*1000)
                            states.append(inner.sim.get_state().flatten().copy())
                            controls.append(inner.sim.data.ctrl.copy())
                            image_hashes.append(hashlib.sha256(obs['agentview_image'].tobytes()+
                                                              obs['robot0_eye_in_hand_image'].tobytes()).hexdigest())
                    finally:
                        for obj, attr, native in reversed(originals): setattr(obj, attr, native)
                        for obj, attr, native in optimization_originals: setattr(obj, attr, native)
                    final_states.append(inner.sim.get_state().flatten().copy())
                    trajectories.append((np.asarray(states), np.asarray(controls), image_hashes))
                    result['modes'].append(dict(mode=mode, steps=30,
                        mean_step_ms=statistics.mean(wall), median_step_ms=statistics.median(wall),
                        stages={k:dict(calls=len(v), exclusive_ms_per_step=sum(v)/30) for k,v in samples.items()},
                        threadpools=threadpool_info(), skipped_forward_calls=skipped[0],
                        state_max_abs_vs_default=float(np.max(np.abs(trajectories[0][0]-trajectories[-1][0]))),
                        controls_max_abs_vs_default=float(np.max(np.abs(trajectories[0][1]-trajectories[-1][1]))),
                        images_exact_vs_default=trajectories[0][2]==image_hashes))
                    print(json.dumps(result['modes'][-1]), flush=True)
            difference = float(np.max(np.abs(final_states[0]-final_states[1])))
            result.update(status='completed', final_sim_state_max_abs_difference=difference,
                          note='No model or inference contention; constant commands; timers are exclusive and add overhead; not a success-rate test')
        finally:
            env.close()
            base._write_json_atomic(run/'summary.json', result)


if __name__ == '__main__':
    main()
