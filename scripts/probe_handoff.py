#!/usr/bin/env python3
"""Small, isolated handoff repair comparison. Never resumes the 150-run scan."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from scripts import probe_delay_buffer as probe
from scripts.run_async_q import validate_pairing
from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import load_libero_norm_stats


def save(run_dir, summary):
    base._write_json_atomic(run_dir/'summary.json', summary)
    base._save_run_status(run_dir, summary['status'], completed_episodes=len(summary['episodes']))
    lines = ['# 动作衔接修复小测', '',
             '同任务、同初态、同模型；区分旧裁切、原前缀锁定和前缀时间条件修订。', '',
             '| 条件 | q | 结果 | 步数 | 真实秒 | 实际Hz | 断供步 | 迟到丢弃 |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for e in summary['episodes']:
        hz = e.get('post_bootstrap_control_hz')
        rate = f'{hz:.2f}' if hz else '—'
        lines.append(f"| {e['condition']} | {e['q']} | {e['status']} | {e['steps']} | {e['real_seconds']:.3f} | {rate} | {len(e['underflow_steps'])} | {e['late_discarded_predictions']} |")
    (run_dir/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def style_fingerprints(model, np):
    pipe = model._pipe
    pipe._graph_torch_stream.synchronize()
    result = {}
    for name, precomputed in pipe._precomputed_styles.items():
        actual = pipe.pipeline.bufs['decoder_'+name].download_new(precomputed.shape, np.uint16)
        result[name] = hashlib.sha256(actual.tobytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', type=int, default=0, choices=range(30))
    parser.add_argument('--q', type=int, default=5, choices=(5, 8, 10, 12, 15))
    parser.add_argument('--modes', nargs='+', default=['crop', 'prefix', 'prefix_t0'],
                        choices=('crop', 'prefix', 'prefix_t0', 'progress', 'head',
                                 'crop_due', 'head_due', 'progress_due',
                                 'prefix_noisy', 'prefix_noisy_due'))
    parser.add_argument('--prefix-len', type=int, default=8, choices=(6, 7, 8))
    parser.add_argument('--pending-arm-gain', type=float, default=1.0)
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.trials <= 3:
        parser.error('diagnostic trials must be 1..3')
    if len(set(args.modes)) != len(args.modes):
        parser.error('modes must be unique')
    if not 0 < args.pending_arm_gain <= 1 or (args.pending_arm_gain != 1 and
                                             any(m.startswith('prefix') for m in args.modes)):
        parser.error('gain must be in (0,1]; prefix conditions require gain=1')
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('handoff diagnostic interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
    config = load_config(repo_root=ROOT)
    config.update(execution_mode='async_handoff_repair_diagnostic', candidate_q=[args.q],
                  repair_modes=args.modes, pending_arm_gain=args.pending_arm_gain,
                  prefix_lock={'length': args.prefix_len, 'full_rtc': False})
    config['screen'] = {**config['screen'], 'task_ids': [args.task_id], 'trials_per_task': args.trials}
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_dir = create_run(config, 'M1', 'handoff-repair', repo_root=ROOT)
        print('run_dir='+str(run_dir), flush=True)
        summary = {'status': 'running', 'config': config, 'episodes': [],
                   'task_id': args.task_id, 'q': args.q, 'modes': args.modes,
                   'expected_episodes': len(args.modes)*args.trials,
                   'seed': 7, 'init_state_index': 0,
                   'claim': 'repair experiment; performance and conditioning not prevalidated'}
        for name in ('scripts/probe_handoff.py', 'scripts/probe_delay_buffer.py',
                     'scripts/eval_q.py', 'perceptbudget/prefix_model.py',
                     'perceptbudget/model.py', 'perceptbudget/libero_input.py'):
            target = run_dir/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        summary['norm_stats_sha256'] = hashlib.sha256(Path(config['norm_stats_path']).read_bytes()).hexdigest()
        save(run_dir, summary)
        env = None
        try:
            summary['egl_cleanup'] = base._patch_egl_cleanup()
            torch, cuda = base._check_cuda_and_native(config)
            import numpy as np
            from perceptbudget.prefix_model import PrefixModel
            summary['cuda'] = cuda
            stats = load_libero_norm_stats(config['norm_stats_path'])
            task = base._load_tasks([args.task_id], config['screen']['suites'])[0]
            summary['task'] = {k: v for k, v in task.items() if k != 'init_state'}
            model = base._load_and_check_model(config, torch)
            needs_prefix = any(mode.startswith('prefix') for mode in args.modes)
            adapter = PrefixModel(model, prefix_len=args.prefix_len, clean_prefix_t0=False) if needs_prefix else None
            _, env_class = base._libero_api()
            env = base._make_env(env_class, task)
            obs, _ = base._prepare_env_episode(env, task, np)
            # Capture and any repair preparation occur before episode timing.
            snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
            if adapter:
                adapter.predict(snapshot, task['instruction'], {})
            else:
                base._infer_actions(model, snapshot, task['instruction'], config['H'], {})
            if adapter and 'prefix_t0' in args.modes:
                adapter.prepare_clean_prefix_styles()
            if adapter and any(mode.startswith('prefix_noisy') for mode in args.modes):
                adapter.prepare_noisy_prefix_graph()
            summary['original_style_sha256'] = style_fingerprints(model, np)
            first_bootstrap = None
            for trial in range(1, args.trials+1):
                modes = args.modes if trial % 2 else list(reversed(args.modes))
                for mode in modes:
                    if adapter:
                        adapter.noisy_prefix = False
                        adapter.clean_prefix_t0 = mode == 'prefix_t0'
                        adapter.noisy_prefix = mode.startswith('prefix_noisy')
                    case_dir = run_dir/f'trial{trial:02d}_{mode}'
                    case_dir.mkdir()
                    print(f'trial={trial} mode={mode} task={args.task_id} q={args.q}', flush=True)
                    with (case_dir/'trace.jsonl').open('w', encoding='utf-8') as stream:
                        episode = probe.run_episode(
                            env, task, args.q, model, stats, config, np, torch,
                            case_dir, stream, adapter if mode.startswith('prefix') else None,
                            condition=mode, trial_index=trial,
                            alignment_policy=mode.split('_')[0] if mode.split('_')[0] in ('progress', 'head') else None,
                            refresh_policy='latest_due' if mode.endswith('_due') else 'boundary',
                            pending_arm_gain=args.pending_arm_gain)
                    episode.update(artifact_dir=str(case_dir.relative_to(run_dir)),
                                   suite=task['suite'], clean_prefix_t0=mode == 'prefix_t0')
                    episode['restored_style_sha256'] = style_fingerprints(model, np)
                    if episode['restored_style_sha256'] != summary['original_style_sha256']:
                        raise RuntimeError('decoder style buffers were not restored after condition')
                    requests = json.loads((case_dir/f'q{args.q}_requests.json').read_text())
                    bootstrap = np.asarray(requests[0]['actions'])
                    if first_bootstrap is None:
                        first_bootstrap = bootstrap.copy()
                    episode['bootstrap_action_max_abs_difference_vs_first'] = float(np.max(np.abs(bootstrap-first_bootstrap)))
                    if hasattr(adapter, 'style_checks'):
                        episode['style_checks'] = list(adapter.style_checks)
                    validate_pairing(summary['episodes']+[episode], run_dir)
                    base._write_json_atomic(case_dir/'episode.json', episode)
                    summary['episodes'].append(episode)
                    save(run_dir, summary)
            summary['status'] = 'completed'
        except BaseException as exc:
            summary.update(status='interrupted', error=f'{type(exc).__name__}: {exc}',
                           traceback=traceback.format_exc())
            raise
        finally:
            save(run_dir, summary)
            if env is not None:
                env.close()


if __name__ == '__main__':
    main()
