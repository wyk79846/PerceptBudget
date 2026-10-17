#!/usr/bin/env python3
"""Real-delay SEAM screen: one isolated process per LIBERO-10 task."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base
from scripts import eval_seam_sync as sync
from scripts import probe_delay_buffer as probe


def audit_episode(task_dir, episode, np):
    requests = json.loads((task_dir / 'q10_requests.json').read_text())
    actions_checked = overlap_steps = 0
    for line in (task_dir / 'trace.jsonl').open():
        row = json.loads(line)
        if row['event'] != 'control_step':
            continue
        source = row['source']
        if source is not None:
            origin, offset = source['origin_step'], source['offset']
            if origin + offset != row['step'] or not 0 <= offset < 20:
                raise RuntimeError('executed action has an expired or shifted index')
            expected = requests[source['prediction_id']]['actions'][offset]
            if not np.array_equal(np.asarray(row['action'], dtype=np.float32),
                                  np.asarray(expected, dtype=np.float32)):
                raise RuntimeError('executed action differs from the selected prediction')
            actions_checked += 1
            overlap_steps += bool(row['inference_active_at_action'])
    commits = episode['commits'][1:]
    if not commits or not overlap_steps:
        raise RuntimeError('no actual concurrent inference/execution was recorded')
    if any(c['commit_step'] > c['observation_step'] + 20 for c in commits):
        raise RuntimeError('a prediction committed after its original deadline')
    guided = [r['seam_guidance'] for r in requests if r.get('seam_guidance', {}).get('vls_enabled')]
    if any(g['guided_start'] != 0 or g['guided_window'] != min(10, g['old_tail_length'])
           for g in guided):
        raise RuntimeError('async VLS did not preserve the observation-time full guidance window')
    return {
        'executed_prediction_actions_checked': actions_checked,
        'control_steps_with_prediction_source_while_inference_pending': overlap_steps,
        'guided_requests': len(guided),
        'requests_without_available_prior': sum(bool(r.get('seam_async', {}).get('prior_unavailable')) for r in requests),
        'guided_window_steps': [r['guided_window'] for r in guided],
        'estimated_delay_steps': [r['estimated_inference_delay_steps'] for r in guided],
        'actual_delay_steps': [r['control_steps_during_inference'] for r in commits],
        'first_chunk_sha256': hashlib.sha256(np.asarray(
            requests[0]['actions_raw'], dtype=np.float32).tobytes()).hexdigest(),
    }


def worker(run_dir, task_id):
    config = json.loads((run_dir / 'run.json').read_text())['config_snapshot']
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
    task_dir = run_dir / 'tasks' / f'task{task_id:02d}'
    task_dir.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'running', 'task_id': task_id}
    base._write_json_atomic(task_dir / 'summary.json', summary)
    env = None
    try:
        base._patch_egl_cleanup()
        torch, cuda = base._check_cuda_and_native(config)
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats
        from perceptbudget.seam_rollout import SEAMGuidanceAdapter
        from flash_rt.subgraphs.pi05.context_action import enable as enable_context

        task = sync.get_tasks([task_id])[0]
        stats = load_libero_norm_stats(config['norm_stats_path'])
        loaded = time.monotonic()
        model = base._load_and_check_model(config, torch)
        summary.update(cuda=cuda, model_load_seconds=time.monotonic()-loaded,
                       task_name=task['task_name'], max_steps=task['max_steps'])
        enable_context(model)
        _, env_class = base._libero_api()
        env = sync.make_env(env_class, task)
        obs, _ = base._prepare_env_episode(env, task, np)
        snapshot = base._make_snapshot(obs, stats, False)
        base._seed_process(7, np, torch)
        base._infer_actions(model, snapshot, task['instruction'], 20, {})
        adapter = SEAMGuidanceAdapter(model, torch, horizon=20, num_steps=10,
                                      strength=0.1, max_window=10, physical_action_dim=7)
        obs, _ = base._prepare_env_episode(env, task, np)
        snapshot = base._make_snapshot(obs, stats, False)
        raw_frames_hash = hashlib.sha256(np.concatenate(
            (obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1).tobytes()).hexdigest()
        summary.update(
            initial_model_frames_sha256=sync.digest_arrays(snapshot['images']),
            fixture_layout_sha256=hashlib.sha256(json.dumps(
                env._pb_fixture_layout, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            solver_precision_diagnostic=adapter.solver_precision_diagnostic)
        base._write_json_atomic(task_dir / 'summary.json', summary)
        with (task_dir / 'trace.jsonl').open('w', encoding='utf-8') as stream:
            episode = probe.run_episode(
                env, task, 10, model, stats, config, np, torch, task_dir, stream,
                condition='seam_async_actual_delay', seam_adapter=adapter,
                refresh_policy='latest_due', pending_arm_gain=1.0)
        summary['episode'] = episode
        base._write_json_atomic(task_dir / 'summary.json', summary)
        if episode['initial_frames_sha256'] != raw_frames_hash:
            raise RuntimeError('episode reset did not match the initial preflight frame')
        summary['audit'] = audit_episode(task_dir, episode, np)
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='invalid', error=f'{type(exc).__name__}: {exc}',
                       traceback=traceback.format_exc())
        raise
    finally:
        base._write_json_atomic(task_dir / 'summary.json', summary)
        if env is not None:
            env.close()


def save_progress(run_dir, summary):
    valid = [x for x in summary['tasks'] if x['status'] == 'completed']
    summary['valid_tasks'] = len(valid)
    summary['successes'] = sum(x['episode']['status'] == 'success' for x in valid)
    base._write_json_atomic(run_dir / 'summary.json', summary)
    rows = ['# SEAM 异步实际延迟测试', '',
            'H20 / N10 / K10 / M上限10 / λ0.1 / BF16网络+FP32动作状态 / 无额外state。',
            '每10个已执行控制步刷新，旧队列原速继续执行；实际延迟裁掉新块过期前缀。',
            '', f"状态：{summary['status']}；有效任务 {len(valid)}/{summary['expected_tasks']}；成功 {summary['successes']}。",
            '', '| task | 结果 | 步数 | 真实秒 | 断供步 | 并发执行步 |',
            '|---:|---|---:|---:|---:|---:|']
    for item in summary['tasks']:
        e = item.get('episode', {})
        rows.append('| {} | {} | {} | {} | {} | {} |'.format(
            item['task_id'], e.get('status', item['status']), e.get('steps', '—'),
            round(e['real_seconds'], 3) if 'real_seconds' in e else '—',
            len(e.get('underflow_steps', [])),
            item.get('audit', {}).get('control_steps_with_prediction_source_while_inference_pending', '—')))
    (run_dir / 'summary.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    base._save_run_status(run_dir, summary['status'], completed_tasks=len(valid),
                         expected_tasks=summary['expected_tasks'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-ids', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--worker-run', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-task', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_run is not None:
        worker(args.worker_run, args.worker_task)
        return 0
    if len(set(args.task_ids)) != len(args.task_ids) or any(i not in range(10) for i in args.task_ids):
        parser.error('task ids must be distinct integers 0..9')
    config = load_config(repo_root=ROOT)
    config.update(H=20, N=10, state_enabled=False, candidate_q=[10],
                  execution_mode='seam_asynchronous_actual_delay',
                  action_handoff='absolute_step_crop_with_seam_vls',
                  refresh_anchor='request', pending_arm_gain=1.0,
                  model_lifetime='task',
                  seam={'K':10, 'M_max':10, 'lambda':0.1, 'physical_action_dim':7,
                        'guidance_timebase': 'observation origin; guide full old tail up to M, including committed rows',
                        'delay': 'measured for diagnostics; actual elapsed steps determine commit crop',
                        'handoff': 'crop by actual elapsed control steps; preserve origin+H deadline'})
    config['screen'] = {**config['screen'], 'suites':['libero_10'],
                        'task_ids':args.task_ids, 'trials_per_task':1, 'seed':7,
                        'init_state_index':0}
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_dir = create_run(config, 'M1', 'seam-async-h20-k10', repo_root=ROOT)
        print('run_dir='+str(run_dir), flush=True)
        source = run_dir/'source'
        source.mkdir()
        for name in ('scripts/eval_seam_async.py', 'scripts/probe_delay_buffer.py',
                     'scripts/eval_seam_sync.py', 'perceptbudget/seam_rollout.py',
                     'perceptbudget/seam_guidance.py'):
            dest = source/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dest)
        summary = {'status':'running', 'expected_tasks':len(args.task_ids),
                   'task_ids':args.task_ids, 'settings':config, 'tasks':[],
                   'limitations':['SEAM mechanism moved into async execution; not the paper synchronous setting',
                                  'one seed/init per task; M is a maximum, actual overlap is reduced by latency']}
        save_progress(run_dir, summary)
        for task_id in args.task_ids:
            task_dir = run_dir/'tasks'/f'task{task_id:02d}'
            task_dir.mkdir(parents=True, exist_ok=True)
            print(f'task={task_id} start', flush=True)
            with (task_dir/'worker.log').open('w') as log:
                result = subprocess.run([
                    config['python_executable'], str(ROOT/'scripts/guarded_run.py'), '--',
                    config['python_executable'], '-u', str(Path(__file__).resolve()),
                    '--worker-run', str(run_dir), '--worker-task', str(task_id)],
                    stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            progress = task_dir/'summary.json'
            item = json.loads(progress.read_text()) if progress.exists() else {'task_id':task_id}
            if result.returncode or item.get('status') != 'completed':
                item.update(status='invalid', worker_exit_code=result.returncode)
            summary['tasks'].append(item)
            save_progress(run_dir, summary)
            print(f'task={task_id} result={item.get("episode", {}).get("status", item["status"])}', flush=True)
        summary['status'] = ('completed' if summary['valid_tasks']==len(args.task_ids) else 'partial_invalid')
        save_progress(run_dir, summary)
        return 0 if summary['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
