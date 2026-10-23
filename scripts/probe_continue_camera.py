#!/usr/bin/env python3
"""No-model paired camera-on/off Continue microcheck; no frequency changes."""
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import fcntl
    import numpy as np
    import torch
    from scripts import eval_q as base
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from perceptbudget.continue_camera import suspend_cameras
    from perceptbudget.device_quality import install_render_buffer_cleanup
    from perceptbudget.env_forward import reuse_control_forward
    from perceptbudget.device_service import fresh_observation

    cfg = load_config(repo_root=ROOT)
    source = Path(cfg['output_root'])/'M2/20261022T071343.413366Z-continue-control-bcf8dd87/baseline_0_trace.jsonl'
    actions = [json.loads(line)['action'] for line in source.read_text().splitlines()]
    if len(actions) != 5:
        raise RuntimeError('Expected five recorded committed actions')
    with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = create_run(cfg, 'M2', 'continue-camera-check', repo_root=ROOT)
        print('run_dir='+str(run), flush=True)
        result = dict(status='running', model_loads=0, source_actions=str(source), cases=[],
                      scope='Five saved actions at paired Spatial0 initial state; no inference, no DVFS, no task-success or energy claim')
        env = None
        try:
            base._patch_egl_cleanup()
            install_render_buffer_cleanup(shared_egl=True)
            _, cls = base._libero_api()
            task = base._load_tasks([0], ['libero_spatial'])[0]
            env = base._make_env(cls, task)
            env._pb_lock_fixtures = True
            references = None
            for repeat in range(3):
                for disabled in (False, True):
                    base._seed_process(7, np, torch)
                    observation, _ = base._prepare_env_episode(env, task, np)
                    initial = env.env.sim.get_state().flatten().copy()
                    render = env.env.sim.render
                    calls = []
                    def counted(*args, **kwargs):
                        calls.append(1)
                        return render(*args, **kwargs)
                    env.env.sim.render = counted
                    states, durations, outcomes = [], [], []
                    try:
                        with (suspend_cameras(env) if disabled else nullcontext()):
                            with (reuse_control_forward(env) if cfg.get('reuse_control_forward') else nullcontext()):
                                for action in actions:
                                    start = time.monotonic()
                                    observation, reward, done, _ = env.step(np.asarray(action, dtype=np.float32))
                                    durations.append((time.monotonic()-start)*1000)
                                    states.append(env.env.sim.get_state().flatten().copy())
                                    outcomes.append([float(reward), bool(done), bool(env.check_success())])
                                    if disabled and any('image' in k for k in observation):
                                        raise RuntimeError('Disabled camera still returned image')
                        step_renders = len(calls)
                        refresh_start = time.monotonic()
                        fresh, _ = fresh_observation(env)
                        refresh_ms = (time.monotonic()-refresh_start)*1000
                    finally:
                        env.env.sim.render = render
                    images = [fresh[k].copy() for k in ('agentview_image', 'robot0_eye_in_hand_image')]
                    if references is None:
                        references = (initial, states, outcomes, images)
                    errors = [float(np.max(np.abs(a-b))) for a,b in zip(states,references[1])]
                    image_error = max(int(np.abs(a.astype(int)-b.astype(int)).max()) for a,b in zip(images,references[3]))
                    passed = np.array_equal(initial,references[0]) and max(errors)==0 and outcomes==references[2] and image_error==0
                    row = dict(repeat=repeat, cameras_disabled=disabled, step_ms=durations,
                               step_render_calls=step_renders, fresh_render_calls=len(calls)-step_renders,
                               fresh_acquisition_ms=refresh_ms, max_physics_error=max(errors),
                               refreshed_image_max_error=image_error, passed=bool(passed))
                    result['cases'].append(row)
                    _write_json_atomically(run/'summary.json', result)
                    print(json.dumps(row), flush=True)
                    if not passed or (disabled and step_renders):
                        raise RuntimeError('Camera suspension parity failed')
            result['status'] = 'completed'
        except BaseException as exc:
            result.update(status='invalid', error=str(exc))
            raise
        finally:
            if env is not None:
                env.close()
            _write_json_atomically(run/'summary.json', result)


if __name__ == '__main__':
    main()
