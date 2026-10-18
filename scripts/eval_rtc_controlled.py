#!/usr/bin/env python3
"""One-task paired RTC diagnostic with exactly five simulated delay steps.

Wall-clock waits freeze simulation; these results are NOT real-time performance.
Both conditions use the identical differentiable expert and graph, differing only
in whether the RTC mask is enabled. No weight updates or FlashRT source changes.
"""
import fcntl
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base
from scripts import eval_seam_sync as sync
from scripts import probe_delay_buffer as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rtc-only', action='store_true', help='Audit RTC only; retain previous paired baseline separately')
    parser.add_argument('--paper-reference', action='store_true', help='Use slow independent paper equations, without a captured expert graph')
    args = parser.parse_args()
    if args.paper_reference and not args.rtc_only:
        parser.error('--paper-reference currently requires --rtc-only')
    config = load_config(repo_root=ROOT)
    config.update(H=20, N=10, state_enabled=False, controlled_delay_steps=5,
                  execution_mode='controlled_simulation_delay_not_real_time',
                  candidate_q=[10], pending_arm_gain=1.0,
                  rtc_solver_backend='paper_reference_eager' if args.paper_reference else 'captured_graph')
    config['screen'].update(seed=7, init_state_index=0)
    with (Path(config['output_root']) / '.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # Queue behind the already running SEAM test.
        run = create_run(config, 'M1', 'rtc-controlled-d5-q10', repo_root=ROOT)
        print('run_dir=' + str(run), flush=True)
        for name in ('scripts/eval_rtc_controlled.py', 'scripts/probe_delay_buffer.py',
                     'perceptbudget/rtc_rollout.py', 'perceptbudget/rtc_guidance.py',
                     'perceptbudget/rtc_expert.py'):
            dest = run / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        base._patch_egl_cleanup()
        torch, _ = base._check_cuda_and_native(config)
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats
        from perceptbudget.rtc_rollout import RTCGuidanceAdapter
        from flash_rt.subgraphs.pi05.context_action import enable
        stats = load_libero_norm_stats(config['norm_stats_path'])
        task = sync.get_tasks([0])[0]
        model = base._load_and_check_model(config, torch)
        enable(model)
        _, env_class = base._libero_api()
        env = sync.make_env(env_class, task)
        result = {'status': 'running', 'task': task, 'conditions': {},
                  'note': 'controlled simulation delay d5; wall-clock simulation pauses; not real-time'}
        try:
            obs, _ = base._prepare_env_episode(env, task, np)
            base._infer_actions(model, base._make_snapshot(obs, stats, False),
                                task['instruction'], 20, {})
            adapter = RTCGuidanceAdapter(model, torch, horizon=20, num_steps=10,
                                         attention_backend='math', audit_reference=True,
                                         reference_backend=args.paper_reference)
            conditions = [('rtc', True)] if args.rtc_only else [('unguided', False), ('rtc', True)]
            for name, guided in conditions:
                adapter.guidance_enabled = guided
                directory = run / name
                directory.mkdir()
                print(name + ' start', flush=True)
                with (directory / 'trace.jsonl').open('w') as stream:
                    episode = probe.run_episode(
                        env, task, 10, model, stats, config, np, torch,
                        directory, stream, rtc_adapter=adapter,
                        condition=name, refresh_policy='latest_due', pending_arm_gain=1.)
                requests = json.loads((directory / 'q10_requests.json').read_text())
                commits = episode['commits'][1:]
                assert all(c['commit_step'] - c['observation_step'] == 5 for c in commits)
                assert not episode['underflow_steps']
                assert all(r['rtc_guidance']['guidance_enabled'] == guided for r in requests[1:])
                checked = 0
                for line in (directory / 'trace.jsonl').open():
                    row = json.loads(line)
                    if row['event'] != 'control_step' or row['source'] is None:
                        continue
                    src = row['source']
                    assert src['origin_step'] + src['offset'] == row['step']
                    expected = requests[src['prediction_id']]['actions'][src['offset']]
                    assert np.array_equal(np.asarray(row['action'], dtype=np.float32),
                                          np.asarray(expected, dtype=np.float32))
                    checked += 1
                episode['executed_actions_audited'] = checked
                episode['controlled_delay_audit_passed'] = True
                result['conditions'][name] = episode
                base._write_json_atomic(run / 'summary.json', result)
                print(name + ' result=' + episode['status'], flush=True)
            if not args.rtc_only:
                a, b = result['conditions']['unguided'], result['conditions']['rtc']
                assert a['initial_sim_state_sha256'] == b['initial_sim_state_sha256']
                assert a['initial_frames_sha256'] == b['initial_frames_sha256']
                result['initial_conditions_match'] = True
            result['status'] = 'completed'
        except BaseException as exc:
            result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run / 'summary.json', result)
            base._save_run_status(run, result['status'])
            env.close()


if __name__ == '__main__':
    main()
