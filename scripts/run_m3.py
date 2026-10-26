#!/usr/bin/env python3
"""M3 single entry: offline action review or resident Full recording."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from perceptbudget.baseline import validate
    from perceptbudget.closed_loop import load_settings, offline_review, recording_session
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from scripts.eval_pooled_views import CLEANUP_GATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('record-only', 'deadline-high', 'deadline-dvfs', 'dvac-probe', 'analyze'), default='record-only')
    parser.add_argument('--suite', choices=('libero_spatial','libero_object','libero_goal'), default='libero_spatial',
                        help='Single task0 suite for the deadline-high smoke test')
    parser.add_argument('--settings', type=Path, default=ROOT/'m3.json')
    parser.add_argument('--source-run', type=Path, nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings = load_settings(args.settings)
    if args.mode in ('deadline-high','deadline-dvfs'):
        settings = dict(settings, mode=args.mode.replace('-','_'), calibration_tasks=[[args.suite,0]])
    cfg = load_config(repo_root=ROOT)
    validate(cfg)
    if args.mode == 'analyze' and (not args.source_run or args.worker):
        parser.error('Offline review requires --source-run, without worker')
    if args.mode == 'dvac-probe' and (not args.source_run or len(args.source_run)!=1):
        parser.error('DVAC probe requires exactly one --source-run')
    if args.mode not in ('analyze','dvac-probe') and args.source_run:
        parser.error('--source-run is only for offline review or DVAC probe')
    if args.dry_run and args.mode=='dvac-probe':
        print(json.dumps(dict(mode=args.mode, model_loads=1, expert_loads=1,
            saved_inputs=5, new_episodes=0, beta_variants=[5.,0.], repeat_pairs=3,
            H=50, N=5, state=True, policy_changed=False, clocks_changed=False,
            timing_scope='Fixed-context expert replay; excludes VLM'),ensure_ascii=False))
        return 0
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, model_loads=0 if args.mode=='analyze' else 1,
            expert_loads=0 if args.mode=='analyze' else 1,
            tasks=settings['calibration_tasks'], q=5, H=50, N=5, state=True,
            policy_changed=args.mode in ('deadline-high','deadline-dvfs'), clocks_changed=args.mode=='deadline-dvfs',
            event_thresholds=settings['events']), ensure_ascii=False))
        return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(cfg['output_root'])/'M3').resolve():
            parser.error('Worker requires an M3 directory')
        if args.mode=='deadline-dvfs':
            subprocess.run(['sudo','-n','true'],check=True)
        if args.mode=='dvac-probe':
            from perceptbudget.device_profile import fixed_view_probe
            fixed_view_probe(cfg,args.run_dir,args.source_run[0],'full',dvac_probe=True)
        else:
            recording_session(cfg, settings, args.run_dir, CLEANUP_GATE)
        return 0
    if args.mode=='deadline-dvfs':
        subprocess.run(['sudo','-v'],check=True)
    cfg = dict(cfg, m3_settings=settings)
    run = create_run(cfg, 'M3', 'action-review' if args.mode=='analyze' else 'dvac-probe' if args.mode=='dvac-probe' else 'record-session', repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    result = dict(status='running', mode=args.mode)
    try:
        if args.mode == 'analyze':
            result = offline_review(run, args.source_run, settings)
        else:
            with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.mode=='deadline-dvfs':
                    from perceptbudget.closed_loop import clock_preflight
                    clock_preflight(run)
                command = [cfg['python_executable'], '-u', str(Path(__file__).resolve()),
                           '--worker', '--run-dir', str(run), '--settings', str(args.settings.resolve()),
                           '--mode',args.mode,'--suite',args.suite]
                if args.source_run:
                    command += ['--source-run',*[str(p.resolve()) for p in args.source_run]]
                if args.mode!='deadline-dvfs':
                    command = [cfg['python_executable'],str(ROOT/'scripts/guarded_run.py'),'--',*command]
                with (run/'worker.log').open('w') as log:
                    if args.mode=='deadline-dvfs':
                        from scripts.profile_clock_screen import monitored_worker
                        code = monitored_worker(command, log)
                    else:
                        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                if code:
                    tail = (run/'worker.log').read_text()[-2500:]
                    raise RuntimeError(f'Recording worker exit{code}; {tail}')
                result = json.loads((run/'summary.json').read_text())
                if result['status'] != 'completed' or (args.mode!='dvac-probe' and
                        len(result.get('episodes', [])) != len(settings['calibration_tasks'])):
                    raise RuntimeError('Incomplete M3 recording session')
    except BaseException as exc:
        result = dict(status='invalid', mode=args.mode, error=f'{type(exc).__name__}: {exc}')
        recording = run/'recording.json'
        if recording.exists():
            partial = json.loads(recording.read_text())
            partial.update(status='invalid', error=result['error'])
            _write_json_atomically(recording, partial)
        raise
    finally:
        _write_json_atomically(run/'summary.json', result)
        meta = json.loads((run/'run.json').read_text())
        meta['status'] = result['status']
        _write_json_atomically(run/'run.json', meta)
        print('M3_TERMINAL_STATUS='+json.dumps(dict(run=str(run), **result), ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
