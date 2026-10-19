"""Replay identical saved moving actions through default/optimized simulation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from perceptbudget.env_forward import reuse_control_forward
from scripts import eval_q as base


def main():
    config = load_config(repo_root=ROOT)
    parent = Path(config['output_root'])/'M1/20261019T061721.555052Z-rtc-async-spatial-ten-10cc127a'
    source = Path(json.loads((parent/'summary.json').read_text())['cases'][0]['source_run'])
    config['replay_source'] = str(source)
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'env-moving-replay', repo_root=ROOT)
        print('run_dir='+str(run), flush=True)
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        base._patch_egl_cleanup()
        import numpy as np
        rows = [json.loads(s) for s in (source/'rtc/trace.jsonl').open()]
        actions = np.asarray([r['action'] for r in rows if r['event']=='control_step'], dtype=np.float32)
        task = base._load_tasks([0], ['libero_spatial'])[0]
        _, cls = base._libero_api()
        env = base._make_env(cls, task)
        env._pb_lock_fixtures = True
        result = dict(status='running', source=str(source), steps=len(actions), model_loaded=False, modes=[])
        trajectories = []
        try:
            for optimized in (False, True, False):
                base._prepare_env_episode(env, task, np)
                inner = env.env
                states, controls, images, successes = [], [], [], []
                with (reuse_control_forward(env) if optimized else nullcontext()) as audit:
                    for action in actions:
                        obs, _, _, _ = env.step(action)
                        states.append(inner.sim.get_state().flatten().copy())
                        controls.append(inner.sim.data.ctrl.copy())
                        images.append(hashlib.sha256(obs['agentview_image'].tobytes()+obs['robot0_eye_in_hand_image'].tobytes()).hexdigest())
                        successes.append(bool(env.check_success()))
                trajectories.append((np.asarray(states), np.asarray(controls), images, successes))
                result['modes'].append(dict(optimized=optimized, success=any(successes), audit=audit))
                print(json.dumps(result['modes'][-1]), flush=True)
            a, b, repeat = trajectories
            result.update(status='completed', state_max_abs=float(np.max(np.abs(a[0]-b[0]))),
                          control_max_abs=float(np.max(np.abs(a[1]-b[1]))),
                          identical_image_steps=sum(x==y for x,y in zip(a[2],b[2])),
                          success_flags_equal=a[3]==b[3],
                          default_repeat_state_max_abs=float(np.max(np.abs(a[0]-repeat[0]))),
                          default_repeat_identical_image_steps=sum(x==y for x,y in zip(a[2],repeat[2])))
            np.savez_compressed(run/'trajectories.npz', actions=actions, original_states=a[0],
                                optimized_states=b[0], original_controls=a[1], optimized_controls=b[1])
            print(json.dumps(result), flush=True)
        finally:
            env.close()
            base._write_json_atomic(run/'summary.json', result)


if __name__ == '__main__':
    main()
