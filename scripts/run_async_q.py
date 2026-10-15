#!/usr/bin/env python3
"""Run paired fixed-q async LIBERO episodes, using async q=5 as reference."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from scripts import probe_delay_buffer as probe
from perceptbudget.experiment import create_run, load_config


HANDOFF_PROFILES = ('crop_overwrite', 'latency_limited_progress')


def configure_handoff(config, requested=None, *, resume=False):
    profile = config.get('action_handoff', 'crop_overwrite') if resume else (requested or 'crop_overwrite')
    if profile not in HANDOFF_PROFILES or (resume and requested is not None and requested != profile):
        raise ValueError('Resume cannot change handoff profile')
    if not resume:
        config['action_handoff'] = profile
        if profile == 'latency_limited_progress':
            config.update(alignment_policy='progress', refresh_policy='latest_due',
                          pending_arm_gain=0.5)
    return profile


def handoff_options(config):
    profile = config.get('action_handoff', 'crop_overwrite')
    options = {'condition': profile}
    if profile == 'latency_limited_progress':
        options.update(alignment_policy=config['alignment_policy'],
                       refresh_policy=config['refresh_policy'],
                       pending_arm_gain=config['pending_arm_gain'])
    return options


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_episode(episode, horizon):
    """Reject missing measurements, not genuine task failures or underflows."""
    checks = episode.get('field_checks', {})
    if not checks or not all(checks.values()):
        raise RuntimeError('Incomplete episode fields: ' + repr(checks))
    profile = episode.get('condition')
    if profile not in HANDOFF_PROFILES:
        raise RuntimeError('Unknown handoff profile')
    if not episode.get('initial_sim_state_sha256') or not episode.get('initial_frames_sha256'):
        raise RuntimeError('Missing paired initial-state/frame fingerprints')
    for item in episode['commits']:
        elapsed = max(0, item['commit_step'] - item['observation_step'])
        remaining = max(0, horizon - elapsed)
        if item['new_prediction_remaining_actions'] != remaining:
            raise RuntimeError('Incorrect remaining prediction interval')
        if profile == 'crop_overwrite':
            if item['expired_new_actions'] != min(horizon, elapsed):
                raise RuntimeError('Incorrect expired prediction prefix')
        else:
            alignment = item.get('action_alignment', {})
            dropped = min(horizon, elapsed)
            offset = alignment.get('selected_prediction_offset')
            if (not isinstance(offset, int) or not 0 <= offset <= dropped or
                    alignment.get('policy') != 'progress' or
                    alignment.get('elapsed_control_steps') != elapsed or
                    item['expired_new_actions'] != offset or
                    alignment.get('discarded_prefix_actions') != offset or
                    alignment.get('discarded_tail_actions') != dropped-offset or
                    alignment.get('remaining_actions') != remaining):
                raise RuntimeError('Incorrect aligned prediction discard counts')
            if (alignment.get('retained_prediction_offset_interval') != [offset, offset+remaining] or
                    not 0 <= offset <= offset+remaining <= horizon):
                raise RuntimeError('Incorrect retained model offset interval')
            scheduled = [item['commit_step'], item['commit_step']+remaining]
            if (alignment.get('scheduled_interval_steps') != scheduled or
                    alignment.get('original_horizon_end_step') != item['observation_step']+horizon or
                    (remaining and scheduled[1] > item['observation_step']+horizon) or
                    item.get('new_prediction_remaining_interval_steps') != scheduled):
                raise RuntimeError('Incorrect scheduled prediction interval')
    if not episode['commits'] or episode['commits'][0]['expired_new_actions'] != 0:
        raise RuntimeError('Bootstrap must not advance the simulator while waiting')


def validate_pairing(episodes, run_dir=None):
    if not episodes:
        return
    if len({ep['initial_sim_state_sha256'] for ep in episodes}) != 1:
        raise RuntimeError('Initial conditions differ across q: initial_sim_state_sha256')
    baseline = episodes[0]
    for ep in episodes[1:]:
        if ep['initial_frames_sha256'] == baseline['initial_frames_sha256']:
            continue
        if run_dir is None:
            raise RuntimeError('Initial conditions differ across q: initial_frames_sha256')
        import numpy as np
        def frame(record):
            path = Path(run_dir) / record['artifact_dir'] / f"q{record['q']}_frames.npz"
            with np.load(path) as data:
                return data['frames'][0].astype(np.int16)
        a, b = frame(baseline), frame(ep)
        if a.shape != b.shape:
            raise RuntimeError('Initial image shapes differ')
        difference = np.abs(a-b)
        comparison = {'baseline_sha256': baseline['initial_frames_sha256'],
                      'max_channel_difference': int(difference.max()),
                      'changed_channel_fraction': float(np.mean(difference != 0)),
                      'mean_channel_difference': float(difference.mean()),
                      'rule': 'same sim state; max uint8 difference<=1; changed fraction<=0.0001'}
        comparison['accepted'] = (comparison['max_channel_difference'] <= 1 and
                                  comparison['changed_channel_fraction'] <= 0.0001)
        ep['initial_frame_comparison'] = comparison
        if not comparison['accepted']:
            raise RuntimeError('Initial frame differs beyond rendering tolerance: ' + repr(comparison))


def write_progress(run_dir, summary):
    base._write_json_atomic(run_dir / 'summary.json', summary)
    lines = ['# 固定 q 异步测试', '',
             '异步 q=5 为基线；同任务和初态与 q=8/10/12/15 配对。', '',
             f"状态：{summary['status']}；完成 {len(summary['episodes'])}/{summary['expected_episodes']} 次。", '',
             '| task | q | 结果 | 步数 | 真实秒 | 实际Hz | 断供步 |',
             '|---:|---:|---|---:|---:|---:|---:|']
    for ep in summary['episodes']:
        hz = ep.get('post_bootstrap_control_hz')
        hz_text = f'{hz:.2f}' if hz is not None else '—'
        lines.append(f"| {ep['task_id']} | {ep['q']} | {ep['status']} | {ep['steps']} | {ep['real_seconds']:.3f} | {hz_text} | {len(ep['underflow_steps'])} |")
    (run_dir / 'progress.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    base._save_run_status(run_dir, summary['status'],
                         completed_episodes=len(summary['episodes']),
                         expected_episodes=summary['expected_episodes'])


def worker(run_dir, task_id):
    """A fresh process per task releases model/EGL memory between five-q groups."""
    config = json.loads((run_dir / 'run.json').read_text())['config_snapshot']
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
    task_dir = run_dir / 'tasks' / f'task{task_id:02d}'
    task_dir.mkdir(parents=True, exist_ok=True)
    progress = task_dir / 'summary.json'
    summary = json.loads(progress.read_text()) if progress.exists() else {'episodes': []}
    summary.update(status='running', task_id=task_id)
    summary.pop('error', None)
    summary.pop('traceback', None)
    env = None
    try:
        summary['egl_cleanup'] = base._patch_egl_cleanup()
        torch, cuda = base._check_cuda_and_native(config)
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats
        task = base._load_tasks([task_id], config['screen']['suites'])[0]
        summary['task'] = {key: str(value) if isinstance(value, Path) else value
                           for key, value in task.items() if key != 'init_state'}
        stats = load_libero_norm_stats(config['norm_stats_path'])
        summary['cuda'] = cuda
        loaded = time.monotonic()
        model = base._load_and_check_model(config, torch)
        summary['model_load_seconds'] = time.monotonic() - loaded
        _, env_class = base._libero_api()
        env = base._make_env(env_class, task)
        problem = env.env.parsed_problem
        summary['problem_metadata'] = {key: problem.get(key) for key in
                                       ('goal_state', 'obj_of_interest', 'regions', 'objects', 'fixtures')}
        base._write_json_atomic(progress, summary)
        obs, _ = base._prepare_env_episode(env, task, np)
        base._infer_actions(model, base._make_snapshot(obs, stats, config['state_enabled']),
                            task['instruction'], config['H'], {})
        completed = {ep['q'] for ep in summary['episodes']}
        for q in config['candidate_q']:
            if q in completed:
                continue
            # A failed attempt keeps its own files; resuming never mixes traces.
            episode_dir = Path(tempfile.mkdtemp(prefix=f'q{q}-', dir=task_dir))
            print(f'task={task_id} q={q} start artifact={episode_dir.name}', flush=True)
            with (episode_dir / 'trace.jsonl').open('w', encoding='utf-8') as stream:
                episode = probe.run_episode(env, task, q, model, stats, config,
                                            np, torch, episode_dir, stream,
                                            **handoff_options(config))
                episode.update(suite=task['suite'], suite_task_id=task['suite_task_id'],
                               artifact_dir=str(episode_dir.relative_to(run_dir)),
                               execution_mode=config['execution_mode'])
                validate_episode(episode, config['H'])
                validate_pairing(summary['episodes'] + [episode], run_dir)
                base._write_json_atomic(episode_dir / 'episode.json', episode)
                summary['episodes'].append(episode)
                base._write_json_atomic(progress, summary)
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='invalid', error=f'{type(exc).__name__}: {exc}',
                       traceback=traceback.format_exc())
        raise
    finally:
        base._write_json_atomic(progress, summary)
        if env is not None:
            env.close()


def main():
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-ids', nargs='+', type=int)
    parser.add_argument('--q', nargs='+', type=int, choices=(5, 8, 10, 12, 15))
    parser.add_argument('--purpose', default='async-q-paired')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--handoff-profile', choices=HANDOFF_PROFILES,
                        default=argparse.SUPPRESS, help='handoff profile (default: crop_overwrite)')
    parser.add_argument('--validation-amendment', help='record a validation-only source revision on resume')
    parser.add_argument('--worker-task', type=int)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.worker_task is not None:
        if args.run_dir is None:
            parser.error('--worker-task needs --run-dir')
        worker(args.run_dir.resolve(), args.worker_task)
        return
    config = load_config(repo_root=ROOT)
    if args.resume:
        run_dir = args.resume.resolve()
        config = json.loads((run_dir / 'run.json').read_text())['config_snapshot']
        if config.get('execution_mode') != 'asynchronous_q5_paired':
            parser.error('Resume requires this runner\'s asynchronous paired run')
        try:
            configure_handoff(config, getattr(args, 'handoff_profile', None), resume=True)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        q_values = args.q or config['candidate_q']
        task_ids = args.task_ids or config['screen']['task_ids']
        if q_values[0] != 5 or len(set(q_values)) != len(q_values):
            parser.error('q=5 must be first and q values must be unique')
        if len(set(task_ids)) != len(task_ids) or any(not 0 <= t < 30 for t in task_ids):
            parser.error('task IDs must be unique integers in [0,29]')
        config.update(execution_mode='asynchronous_q5_paired', baseline_q=5,
                      candidate_q=q_values,
                      control_period_s=0.05, worker_scope='one task, all q; fresh process per task')
        configure_handoff(config, getattr(args, 'handoff_profile', None))
        config['screen'] = {**config['screen'], 'task_ids': task_ids,
                            'trials_per_task': 1}
        run_dir = create_run(config, 'M1', args.purpose, repo_root=ROOT)
    print('run_dir=' + str(run_dir), flush=True)
    # Only this parent owns the GPU job; children run strictly sequentially.
    with (Path(config['output_root']) / '.async_gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An async-q runner is already active')
        if args.resume:
            summary = json.loads((run_dir / 'summary.json').read_text())
            changed = {}
            for name, expected in summary['source_sha256'].items():
                if file_sha256(ROOT / name) != expected:
                    changed[name] = {'old_sha256': expected, 'new_sha256': file_sha256(ROOT / name)}
            if changed:
                permitted = {'scripts/run_async_q.py', 'scripts/analyze_async_q.py'}
                if not args.validation_amendment or set(changed) - permitted:
                    raise RuntimeError('Evaluation source changed since this run: ' + repr(changed))
                revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
                for name, hashes in changed.items():
                    target = run_dir / 'source' / 'amendments' / revision / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((ROOT / name).read_bytes())
                    summary['source_sha256'][name] = hashes['new_sha256']
                summary.setdefault('validation_amendments', []).append({
                    'reason': args.validation_amendment, 'revision': revision,
                    'files': changed, 'prior_completed_episodes': len(summary['episodes'])})
        else:
            names = ['scripts/run_async_q.py', 'scripts/probe_delay_buffer.py',
                     'scripts/eval_q.py', 'scripts/run_flashrt.py',
                     'scripts/analyze_async_q.py', 'scripts/analyze_sync_quality.py',
                     'perceptbudget/model.py', 'perceptbudget/libero_input.py',
                     'perceptbudget/power.py', 'perceptbudget/experiment.py']
            sources = {}
            for name in names:
                target = run_dir / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
                sources[name] = file_sha256(target)
            summary = {'status': 'preparing', 'execution_mode': config['execution_mode'],
                       'baseline_q': 5, 'q_values': config['candidate_q'],
                       'task_ids': config['screen']['task_ids'], 'episodes': [],
                       'expected_episodes': len(config['candidate_q']) * len(config['screen']['task_ids']),
                       'config': config, 'source_sha256': sources,
                       'seed': 7, 'init_state_index': 0,
                       'state_protocol': base.STATE_PROTOCOL_NOTE,
                       'limitations': ['external disturbance untested',
                                       'unexpired queued horizon is not quality-verified B_safe',
                                       'actual control rate measured, not assumed real 20Hz',
                                       'one initial condition per task; uncertainty is across paired tasks'],
                       'checkpoint_sha256': file_sha256(config['model_path']),
                       'norm_stats_sha256': file_sha256(config['norm_stats_path'])}
        summary['status'] = 'running'
        summary.pop('error', None)
        write_progress(run_dir, summary)
        try:
            for task_id in config['screen']['task_ids']:
                task_dir = run_dir / 'tasks' / f'task{task_id:02d}'
                task_dir.mkdir(parents=True, exist_ok=True)
                task_summary = task_dir / 'summary.json'
                previous = json.loads(task_summary.read_text()) if task_summary.exists() else {}
                if previous.get('status') != 'completed':
                    with (task_dir / 'worker.log').open('a', encoding='utf-8') as log:
                        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                                    '--worker-task', str(task_id), '--run-dir', str(run_dir)],
                                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                   pass_fds=(lock.fileno(),))
                        try:
                            summary.update(active_task_id=task_id, active_worker_pid=process.pid)
                            write_progress(run_dir, summary)
                            code = process.wait()
                        except BaseException:
                            if process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait()
                            raise
                    if code:
                        raise RuntimeError(f'task {task_id} worker exited {code}; inspect {task_dir / "worker.log"}')
                data = json.loads(task_summary.read_text())
                summary['episodes'] = [ep for ep in summary['episodes'] if ep['task_id'] != task_id] + data['episodes']
                if (len(data['episodes']) != len(config['candidate_q']) or
                        {ep['q'] for ep in data['episodes']} != set(config['candidate_q'])):
                    raise RuntimeError(f'Incomplete q group for task {task_id}')
                validate_pairing(data['episodes'], run_dir)
                write_progress(run_dir, summary)
                print(f"completed={len(summary['episodes'])}/{summary['expected_episodes']} task={task_id}", flush=True)
            summary.update(status='completed', active_task_id=None, active_worker_pid=None)
        except BaseException as exc:
            summary.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write_progress(run_dir, summary)
        if summary['status'] == 'completed':
            with (run_dir / 'analysis.log').open('a', encoding='utf-8') as log:
                code = subprocess.call([sys.executable, str(ROOT / 'scripts/analyze_async_q.py'),
                                        str(run_dir)], cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT)
            summary['analysis_status'] = 'completed' if code == 0 else 'failed'
            write_progress(run_dir, summary)


if __name__ == '__main__':
    main()
