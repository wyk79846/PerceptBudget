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
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base
from scripts import libero_tasks as sync
from scripts import probe_delay_buffer as probe


def spatial_screen(config, args):
    """Sequential isolated task workers; durable results in one run folder."""
    reused = []
    if args.resume_from:
        previous_config = json.loads((args.resume_from/'run.json').read_text())['config_snapshot']
        for key in ('H', 'N', 'state_enabled', 'controlled_delay_steps', 'candidate_q',
                    'rtc_attention_backend', 'rtc_deterministic_attention',
                    'checkpoint_path', 'flashrt_revision'):
            if previous_config.get(key) != config.get(key):
                raise ValueError(f'resume cannot mix settings: {key}')
        old = json.loads((args.resume_from/'summary.json').read_text())
        for item in old['tasks']:
            if item.get('status') == 'completed':
                reused.append({**item, 'source_task_dir': item.get('source_task_dir') or
                               str(args.resume_from/'tasks'/f"task{item['task_id']:02d}")})
        config['resume_source'] = str(args.resume_from)
    run = create_run(config, 'M1', f'rtc-spatial-n5-q{args.q}-d5' + ('-state' if args.state else ''), repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    summary = {'status': 'running', 'expected_tasks': 10, 'tasks': reused,
               'note': 'fixed five simulated delay steps; not real-time async'}
    if args.resume_from:
        summary.update(resume_source=str(args.resume_from), reused_task_ids=[r['task_id'] for r in reused],
                       prior_solver_backend=previous_config.get('rtc_solver_backend'),
                       current_solver_backend=config['rtc_solver_backend'],
                       amendment='Completed source tasks preserved; invalid attempts excluded. Recording/workspace/solver execution changes are explicit in config and source; RTC equations, attention backend and action timing unchanged.')
    def save():
        valid = [r for r in summary['tasks'] if r.get('status') == 'completed']
        summary['valid_tasks'] = len(valid)
        summary['successes'] = sum(r['conditions']['rtc']['status'] == 'success' for r in valid)
        base._write_json_atomic(run/'summary.json', summary)
        rows = [f'# RTC Spatial 十任务，N5/q{args.q}/d5，state={config["state_enabled"]}', '', summary['note'], '',
                f"状态：{summary['status']}；有效{len(valid)}/10；成功{summary['successes']}。", '',
                '| task | 结果 | 步数 | 请求中位ms | 断供 |', '|---:|---|---:|---:|---:|']
        for item in summary['tasks']:
            e = item.get('conditions', {}).get('rtc', {})
            rows.append('| {} | {} | {} | {} | {} |'.format(
                item['task_id'], e.get('status', item['status']), e.get('steps', '—'),
                e.get('steady_inference_ms', {}).get('median', '—'), len(e.get('underflow_steps', []))))
        (run/'summary.md').write_text('\n'.join(rows)+'\n')
        base._save_run_status(run, summary['status'])
    save()
    for task_id in range(10):
        if any(r['task_id'] == task_id for r in reused):
            continue
        directory = run/'tasks'/f'task{task_id:02d}'
        directory.mkdir(parents=True)
        command = [config['python_executable'], str(ROOT/'scripts/guarded_run.py'), '--',
                   config['python_executable'], '-u', str(Path(__file__).resolve()),
                   '--rtc-only', '--suite', 'libero_spatial', '--task-id', str(task_id),
                   '--q', str(args.q), '--denoising-steps', str(args.denoising_steps),
                   '--attention-backend', args.attention_backend, '--worker-run', str(run)]
        if args.deterministic_attention:
            command.append('--deterministic-attention')
        if args.state:
            command.append('--state')
        if args.eager_rtc:
            command.append('--eager-rtc')
        print(f'task={task_id} start', flush=True)
        with (directory/'worker.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = directory/'summary.json'
        item = json.loads(path.read_text()) if path.exists() else {}
        item['task_id'] = task_id
        if completed.returncode or item.get('status') != 'completed':
            item.update(status='invalid', worker_exit_code=completed.returncode)
        summary['tasks'].append(item)
        save()
        print(f"task={task_id} result={item.get('conditions', {}).get('rtc', {}).get('status', item['status'])}", flush=True)
        if item['status'] != 'completed':
            summary['status'] = 'stopped_invalid'
            save()
            return
    summary['status'] = 'completed' if summary['valid_tasks'] == 10 else 'partial_invalid'
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-delay', action='store_true', help='No controlled simulator waits; q is minimum feasible refresh interval')
    parser.add_argument('--paper-schedule', action='store_true', help='Algorithm 1: max of last 10 observed delays; q=s_min')
    parser.add_argument('--reuse-control-forward', action='store_true', help='Experimental process-local duplicate forward removal')
    parser.add_argument('--rtc-only', action='store_true', help='Audit RTC only; retain previous paired baseline separately')
    parser.add_argument('--paper-reference', action='store_true', help='Use slow independent paper equations, without a captured expert graph')
    parser.add_argument('--attention-backend', choices=('math', 'flash', 'flash-varlen', 'fa2-seqused'), default='math')
    parser.add_argument('--deterministic-attention', action='store_true')
    parser.add_argument('--state', action='store_true', help='Enable normalized robot state in the FlashRT prompt')
    parser.add_argument('--eager-rtc', action='store_true', help='Validate same RTC equations without CUDA graph workspace/capture')
    parser.add_argument('--preflight-rtc', action='store_true', help='Validate and time RTC before the task; keep live task timing separate')
    parser.add_argument('--native-fused-ffn',action='store_true',help='Experimental native FFN and fused input gradients; not baseline-equivalent')
    parser.add_argument('--fused-attention-ops',action='store_true',help='Experimental fused Ada/RoPE/residual; requires native-fused-ffn')
    parser.add_argument('--state-prompt-capacity',type=int,choices=(64,80,96,200),help='Physical fixed-state capacity only; never truncate actual tokens')
    parser.add_argument('--profile-rtc-stages',action='store_true')
    parser.add_argument('--denoising-steps', type=int, choices=(5, 10), default=10)
    parser.add_argument('--q', type=int, choices=(5, 8, 10), default=10)
    parser.add_argument('--suite', choices=('libero_10', 'libero_spatial'), default='libero_10')
    parser.add_argument('--task-id', type=int, choices=range(10), default=0)
    parser.add_argument('--spatial-screen', action='store_true')
    parser.add_argument('--worker-run', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--resume-from', type=Path, help='Reuse completed tasks from a compatible stopped screen; preserve original attempts')
    args = parser.parse_args()
    if args.reuse_control_forward and not args.actual_delay:
        parser.error('--reuse-control-forward requires --actual-delay')
    if args.paper_schedule and not args.actual_delay:
        parser.error('--paper-schedule requires --actual-delay')
    if args.actual_delay and (not args.rtc_only or not args.preflight_rtc or args.spatial_screen or args.worker_run):
        parser.error('--actual-delay requires single --rtc-only --preflight-rtc')
    if args.fused_attention_ops and not args.native_fused_ffn:
        parser.error('--fused-attention-ops requires --native-fused-ffn')
    if args.state_prompt_capacity is not None and (not args.state or not args.preflight_rtc or args.spatial_screen):
        parser.error('--state-prompt-capacity requires single --state --preflight-rtc')
    if args.native_fused_ffn and (args.spatial_screen or not args.rtc_only or not args.preflight_rtc
                                  or args.attention_backend!='flash-varlen'):
        parser.error('--native-fused-ffn requires single --rtc-only --preflight-rtc --attention-backend flash-varlen')
    if args.resume_from and not args.spatial_screen:
        parser.error('--resume-from is only valid for a screen parent')
    if args.paper_reference and not args.rtc_only:
        parser.error('--paper-reference currently requires --rtc-only')
    if args.eager_rtc and (args.paper_reference or not args.rtc_only):
        parser.error('--eager-rtc requires --rtc-only and cannot combine with --paper-reference')
    if args.preflight_rtc and (not args.rtc_only or args.eager_rtc or args.paper_reference or args.spatial_screen):
        parser.error('--preflight-rtc is for a single captured-RTC task')
    if args.spatial_screen and (args.worker_run or not args.rtc_only or args.suite != 'libero_spatial'
                               or args.q not in (5, 8) or args.denoising_steps != 5 or args.paper_reference):
        parser.error('spatial screen requires --rtc-only --suite libero_spatial --q 5/8 --denoising-steps 5')
    config = (json.loads((args.worker_run/'run.json').read_text())['config_snapshot']
              if args.worker_run else load_config(repo_root=ROOT))
    config.update(H=20, N=args.denoising_steps, state_enabled=args.state, controlled_delay_steps=5,
                  execution_mode='controlled_simulation_delay_not_real_time',
                  candidate_q=[args.q], pending_arm_gain=1.0,
                  rtc_solver_backend=('paper_reference_eager' if args.paper_reference else
                                      'eager_single_forward' if args.eager_rtc else 'captured_graph'),
                  rtc_attention_backend=args.attention_backend,
                  rtc_deterministic_attention=args.deterministic_attention,
                  release_rtc_graph_before_artifacts=args.rtc_only and args.state,
                  disk_backed_frames=args.rtc_only and args.state,
                  rtc_validation_before_task=args.preflight_rtc,
                  rtc_native_fused_ffn=args.native_fused_ffn,
                  rtc_fused_attention_ops=args.fused_attention_ops,
                  rtc_profile_stages=args.profile_rtc_stages)
    if args.state_prompt_capacity is not None:
        config['state_prompt_capacity']=args.state_prompt_capacity
    if args.actual_delay:
        config.pop('controlled_delay_steps', None)
        config.update(execution_mode='rtc_actual_delay_no_simulation_freeze',
                      rtc_wait_for_delay_feasibility=True,
                      q_semantics='minimum request interval, additionally s>=estimated d; serialized inference')
    if args.paper_schedule:
        config.update(rtc_paper_schedule=True, rtc_delay_buffer_size=10,
                      rtc_wait_for_delay_feasibility=False,
                      q_semantics='Algorithm1 s_min; start when idle and s>=q; delay=max(last10 actual step delays)')
    config['reuse_control_forward']=args.reuse_control_forward
    config['screen'].update(seed=7, init_state_index=0, suites=[args.suite],
                            task_ids=list(range(10)) if args.spatial_screen else [args.task_id],
                            trials_per_task=1, task_ids_status='explicit_controlled_RTC_selection')
    if args.spatial_screen:
        return spatial_screen(config, args)
    with (Path(config['output_root']) / '.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # Queue behind the already running SEAM test.
        if args.worker_run:
            run = args.worker_run/'tasks'/f'task{args.task_id:02d}'
            base._write_json_atomic(run/'run.json', {'status': 'created', 'config_snapshot': config})
        else:
            run = create_run(config, 'M1', ('rtc-paper-schedule' if args.paper_schedule else 'rtc-actual-delay' if args.actual_delay else 'rtc-controlled-d5')+f'-q{args.q}' + ('-state' if args.state else ''), repo_root=ROOT)
        print('run_dir=' + str(run), flush=True)
        for name in ('scripts/eval_rtc_controlled.py', 'scripts/probe_delay_buffer.py',
                     'perceptbudget/rtc_rollout.py', 'perceptbudget/rtc_guidance.py',
                     'perceptbudget/rtc_expert.py', 'perceptbudget/rtc_native_probe.py',
                     'perceptbudget/rtc_fused_backward.py','perceptbudget/rtc_fused_backward.cu',
                     'perceptbudget/model.py','perceptbudget/compact_state.py','perceptbudget/gemm_policy.py',
                     'perceptbudget/rtc_schedule.py','perceptbudget/env_forward.py'):
            dest = run / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        frame_source = run/'source/perceptbudget/trace_frames.py'
        frame_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'perceptbudget/trace_frames.py', frame_source)
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        if args.deterministic_attention:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        base._patch_egl_cleanup()
        torch, _ = base._check_cuda_and_native(config)
        if args.deterministic_attention:
            torch.use_deterministic_algorithms(True)
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats
        from perceptbudget.rtc_rollout import RTCGuidanceAdapter
        from flash_rt.subgraphs.pi05.context_action import enable
        stats = load_libero_norm_stats(config['norm_stats_path'])
        task = (sync.get_tasks([args.task_id])[0] if args.suite == 'libero_10'
                else base._load_tasks([args.task_id], ['libero_spatial'])[0])
        model = base._load_and_check_model(config, torch)
        enable(model)
        _, env_class = base._libero_api()
        env = sync.make_env(env_class, task)
        result = {'status': 'running', 'task': task, 'conditions': {},
                  'note': ('actual-delay asynchronous execution; no controlled simulation freeze; measured tick rates reported separately'
                           if args.actual_delay else 'controlled simulation delay d5; wall-clock simulation pauses; not real-time')}
        try:
            obs, _ = base._prepare_env_episode(env, task, np)
            base._infer_actions(model, base._make_snapshot(obs, stats, config['state_enabled']),
                                task['instruction'], 20, {})
            adapter = RTCGuidanceAdapter(model, torch, horizon=20, num_steps=args.denoising_steps,
                                         attention_backend=args.attention_backend, audit_reference=True,
                                         reference_backend=args.paper_reference, eager_backend=args.eager_rtc,
                                         native_fused_output_root=config['output_root'] if args.native_fused_ffn else None,
                                         fused_attention_ops=args.fused_attention_ops,
                                         profile_stages=args.profile_rtc_stages)
            result['initial_graph_capture_seconds'] = adapter.capture_seconds
            if args.preflight_rtc:
                from scripts.probe_rtc_step import timed
                started = time.monotonic()
                snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
                previous = model._pipe._noise_out.float().cpu().numpy().copy()[args.q:]
                check = {}
                adapter.predict(snapshot, task['instruction'], previous,
                                delay=5, executed=args.q, request=check)
                if not check.get('rtc_reference_audit', {}).get('passed'):
                    raise RuntimeError('preflight independent RTC reference gate did not pass')
                result['preflight_reference_audit'] = check['rtc_reference_audit']
                def inference():
                    return adapter.predict(snapshot, task['instruction'], previous,
                                           delay=5, executed=args.q, request={})
                result['preflight_full_request_warm'] = timed(torch, inference)
                result['preflight_expert_graph_warm'] = timed(torch, adapter.graph.graph.replay)
                if args.actual_delay:
                    adapter.latency_estimate_seconds=result['preflight_full_request_warm']['median_ms']/1000
                if args.paper_schedule:
                    import math
                    config['rtc_initial_delay_steps']=math.ceil(adapter.latency_estimate_seconds/0.05)
                    result['rtc_schedule']={'initial_delay_steps':config['rtc_initial_delay_steps'],
                        'buffer_size':10,'estimate':'max(observed controller step delays)',
                        'minimum_execution_steps':args.q,
                        'bootstrap_note':'allow d>s only when old overlap still covers d; no mask clipping'}
                    run_record=json.loads((run/'run.json').read_text())
                    run_record['config_snapshot']=config
                    base._write_json_atomic(run/'run.json',run_record)
                result['preflight_seconds'] = time.monotonic()-started
                base._write_json_atomic(run/'summary.json', result)
                print('preflight='+json.dumps({k: v for k, v in result.items() if k.startswith('preflight')}), flush=True)
            conditions = [('rtc', True)] if args.rtc_only else [('unguided', False), ('rtc', True)]
            for name, guided in conditions:
                adapter.guidance_enabled = guided
                directory = run / name
                directory.mkdir()
                print(name + ' start', flush=True)
                with (directory / 'trace.jsonl').open('w') as stream:
                    episode = probe.run_episode(
                        env, task, args.q, model, stats, config, np, torch,
                        directory, stream, rtc_adapter=adapter,
                        condition=name, refresh_policy='latest_due', pending_arm_gain=1.)
                if args.reuse_control_forward and not episode['env_forward_optimization']['skipped_forward_calls']:
                    raise RuntimeError('environment forward optimization did not activate')
                requests = json.loads((directory / f'q{args.q}_requests.json').read_text())
                if config['state_enabled']:
                    assert all(np.asarray(r['model_state']).shape == (8,) and
                               np.isfinite(r['model_state']).all() for r in requests)
                commits = episode['commits'][1:]
                if not args.actual_delay:
                    assert all(c['commit_step'] - c['observation_step'] == 5 for c in commits)
                    assert not episode['underflow_steps']
                else:
                    assert all('controlled_wait_seconds' not in r for r in requests)
                    episode['actual_inference_delay_steps']=[c['commit_step']-c['observation_step'] for c in commits]
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
                episode['actual_delay_audit_passed' if args.actual_delay else 'controlled_delay_audit_passed'] = True
                times = [(r['inference_ended_monotonic']-r['inference_started_monotonic'])*1000
                         for r in requests[1:] if 'rtc_reference_audit' not in r]
                episode['steady_inference_ms'] = ({'count': len(times), 'median': float(np.median(times)),
                    'p95': float(np.percentile(times, 95)), 'max': max(times)} if times else {})
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
