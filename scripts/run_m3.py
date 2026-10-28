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
    parser.add_argument('--mode', choices=('record-only', 'deadline-high', 'deadline-dvfs', 'dvac-probe', 'dvac-validate', 'dvac-paired', 'dvac-horizon-paired', 'trigger-comparison', 'watchdog', 'analyze'), default='record-only')
    parser.add_argument('--suite', choices=('libero_spatial','libero_object','libero_goal'), default='libero_spatial',
                        help='Single task0 suite for the deadline-high smoke test')
    parser.add_argument('--settings', type=Path, default=ROOT/'m3.json')
    parser.add_argument('--trigger-suite', choices=('libero_spatial','libero_object','libero_goal','libero_10'))
    parser.add_argument('--trigger-task-id', type=int, choices=range(10), default=0)
    parser.add_argument('--static-q', type=int, choices=(5,10,20,30,40), help='Pair one fixed-q high-frequency baseline with variance-only DVFS')
    parser.add_argument('--compare-event-dedup', action='store_true', help='Resident before/after event deduplication pair with exact bootstrap audit')
    parser.add_argument('--strategies', nargs='+', choices=('events_only','events_only_merge','variance_only','events_and_variance','events_and_variance_hold3','events_and_variance_open_probe','events_and_variance_valley','events_and_variance_latest','events_and_variance_no_turn','events_and_variance_slowdown_probe','events_and_variance_merge'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--device-dvfs', action='store_true', help='Use low-frequency Continue and early recovery in trigger-comparison')
    parser.add_argument('--no-age-limit', action='store_true', help='Disable the fixed observation-age cutoff for trigger-comparison')
    parser.add_argument('--compare-device', action='store_true', help='Pair high and DVFS episodes in the same resident model')
    parser.add_argument('--source-run', type=Path, nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if 'events_and_variance_slowdown_probe' in (args.strategies or []) and (
            args.mode!='trigger-comparison' or args.trigger_suite!='libero_10' or args.trigger_task_id!=4
            or not args.device_dvfs or args.compare_device or args.compare_event_dedup or args.static_q is not None):
        parser.error('Single slowdown probe requires the LIBERO-10 task4 DVFS pair')
    if 'events_and_variance_open_probe' in (args.strategies or []) and (
            args.mode!='trigger-comparison' or args.trigger_suite!='libero_object' or args.trigger_task_id!=0
            or not args.device_dvfs or args.compare_device or args.compare_event_dedup or args.static_q is not None):
        parser.error('The retrospective open probe is restricted to the Object task0 DVFS pair')
    if args.compare_event_dedup and (args.mode!='trigger-comparison' or not args.device_dvfs
            or args.compare_device or args.static_q is not None or args.repeats!=1
            or args.strategies not in (None,['events_and_variance'],['events_and_variance','variance_only'])):
        parser.error('Event dedup pair requires trigger-comparison --device-dvfs, one repeat, and events_and_variance; optionally append variance_only')
    if (args.static_q is not None or args.trigger_task_id) and args.mode!='trigger-comparison':
        parser.error('Task selection and static baseline require trigger-comparison')
    if args.static_q is not None and (args.compare_device or args.device_dvfs or args.repeats!=1 or args.strategies not in (None,['variance_only'])):
        parser.error('Static pair uses variance_only, one repeat, and its own device pairing')
    if args.device_dvfs and args.mode!='trigger-comparison':
        parser.error('--device-dvfs requires --mode trigger-comparison')
    if args.no_age_limit and args.mode!='trigger-comparison':
        parser.error('--no-age-limit requires --mode trigger-comparison')
    if args.compare_device and (args.mode!='trigger-comparison' or args.device_dvfs):
        parser.error('--compare-device requires trigger-comparison without --device-dvfs')
    settings = load_settings(args.settings)
    if args.mode in ('deadline-high','deadline-dvfs'):
        settings = dict(settings, mode=args.mode.replace('-','_'), calibration_tasks=[[args.suite,0]])
    if args.mode in ('dvac-paired','dvac-horizon-paired'):
        settings=dict(settings,mode='deadline_high',dvac_paired=True,
                      dvac_horizon_paired=args.mode=='dvac-horizon-paired')
    if args.mode=='watchdog':
        settings=dict(settings,mode='deadline_dvfs',dvac_paired=True,watchdog_checks=True,
            calibration_tasks=[[args.suite,0]],
            timing=dict(settings['timing'],maximum_action_offset=50,maximum_observation_age_seconds=None))
    if args.mode=='trigger-comparison':
        settings=dict(settings,mode='deadline_dvfs' if args.device_dvfs or args.compare_device or args.static_q is not None else 'deadline_high',dvac_paired=True,trigger_comparison=True,
            device_comparison=args.compare_device or args.static_q is not None,static_q=args.static_q,
            event_dedup_comparison=args.compare_event_dedup,
            trigger_variants=args.strategies,trigger_repeats=args.repeats,
            timing=dict(settings['timing'],maximum_action_offset=50))
        if args.trigger_suite:
            settings['calibration_tasks']=[[args.trigger_suite,args.trigger_task_id]]
        elif args.trigger_task_id:
            parser.error('--trigger-task-id requires --trigger-suite')
        if args.no_age_limit:
            settings['timing']['maximum_observation_age_seconds']=None
        from perceptbudget.closed_loop import comparison_variants
        try:
            variants_per_task=len(comparison_variants(dict(m3_trigger_comparison=True,
                m3_trigger_variants=args.strategies,m3_trigger_repeats=args.repeats,m3_device_comparison=args.compare_device,m3_static_q=args.static_q,
                m3_event_dedup_comparison=args.compare_event_dedup)))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.trigger_suite or args.strategies or args.repeats!=1:
            parser.error('Trigger selection and repeats require --mode trigger-comparison')
        variants_per_task=3 if settings.get('watchdog_checks') else 2 if settings.get('dvac_paired') else 1
    uses_dvfs=settings.get('mode')=='deadline_dvfs'
    cfg = load_config(repo_root=ROOT)
    validate(cfg)
    if args.mode == 'analyze' and (not args.source_run or args.worker):
        parser.error('Offline review requires --source-run, without worker')
    if args.mode == 'dvac-probe' and (not args.source_run or len(args.source_run)!=1):
        parser.error('DVAC probe requires exactly one --source-run')
    if args.mode=='dvac-validate' and not args.source_run:
        parser.error('DVAC validation requires saved source runs')
    if args.mode not in ('analyze','dvac-probe','dvac-validate') and args.source_run:
        parser.error('--source-run is only for offline review or DVAC probe')
    if args.dry_run and args.mode in ('dvac-probe','dvac-validate'):
        print(json.dumps(dict(mode=args.mode, model_loads=1, expert_loads=1,
            saved_inputs=5*len(args.source_run), new_episodes=0, beta_variants=[5.,0.], repeat_pairs=3,
            H=50, N=5, state=True, policy_changed=False, clocks_changed=False,
            timing_scope='Fixed-context expert replay; excludes VLM'),ensure_ascii=False))
        return 0
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode, model_loads=0 if args.mode=='analyze' else 1,
            expert_loads=0 if args.mode=='analyze' else 1,
            tasks=settings['calibration_tasks'], q=5, static_q=args.static_q,
            exact_bootstrap_pair=args.compare_event_dedup or args.strategies==['events_only','events_and_variance'] or any(s in (args.strategies or []) for s in ('events_and_variance_hold3','events_and_variance_open_probe','events_and_variance_valley','events_and_variance_latest','events_and_variance_no_turn','events_and_variance_slowdown_probe','events_and_variance_merge','events_only_merge')),
            exact_preintervention_prefix=any(s in (args.strategies or []) for s in ('events_and_variance_open_probe','events_and_variance_slowdown_probe')), H=50, N=5, state=True,
            episodes=len(settings['calibration_tasks'])*variants_per_task,
            policy_changed=args.mode in ('deadline-high','deadline-dvfs','dvac-paired','dvac-horizon-paired','trigger-comparison','watchdog'), clocks_changed=uses_dvfs,
            fault_checks=['late','underflow','hang'] if settings.get('watchdog_checks') else [],
            request_timeout_seconds=3. if settings.get('watchdog_checks') else None,
            action_offset_limits=[20,50] if settings.get('dvac_horizon_paired') else [settings['timing']['maximum_action_offset']],
            maximum_observation_age_seconds=settings['timing']['maximum_observation_age_seconds'],
            slowdown_confirmation=[2,3] if 'events_and_variance_hold3' in (args.strategies or []) else [settings['events'].get('slowdown_confirm_steps') or settings['events'].get('confirm_steps',2)],
            slowdown_detectors=['drop','valley'] if 'events_and_variance_valley' in (args.strategies or []) else [settings['events'].get('slowdown_detector','drop')],
            signal_policies=['earliest','latest'] if 'events_and_variance_latest' in (args.strategies or []) else ['earliest'],
            merge_gap_steps=[None,8] if any(s in (args.strategies or []) for s in ('events_and_variance_merge','events_only_merge')) else [None],
            denoising_triggers=[True]*3 if settings.get('watchdog_checks') else [s not in ('events_only','events_only_merge') for s in (args.strategies or [])],
            event_thresholds=settings['events']), ensure_ascii=False))
        return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(cfg['output_root'])/'M3').resolve():
            parser.error('Worker requires an M3 directory')
        if uses_dvfs:
            subprocess.run(['sudo','-n','true'],check=True)
        if args.mode in ('dvac-probe','dvac-validate'):
            from perceptbudget.device_profile import fixed_view_probe
            fixed_view_probe(cfg,args.run_dir,args.source_run[0],'full',dvac_probe=args.mode=='dvac-probe',
                             dvac_sources=args.source_run if args.mode=='dvac-validate' else None)
        else:
            recording_session(cfg, settings, args.run_dir, CLEANUP_GATE)
        return 0
    if uses_dvfs:
        subprocess.run(['sudo','-v'],check=True)
    cfg = dict(cfg, m3_settings=settings)
    run = create_run(cfg, 'M3', 'action-review' if args.mode=='analyze' else args.mode if args.mode in ('dvac-probe','dvac-validate') else args.mode if settings.get('dvac_paired') else 'record-session', repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    result = dict(status='running', mode=args.mode)
    try:
        if args.mode == 'analyze':
            result = offline_review(run, args.source_run, settings)
        else:
            with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if uses_dvfs:
                    from perceptbudget.closed_loop import clock_preflight
                    clock_preflight(run)
                command = [cfg['python_executable'], '-u', str(Path(__file__).resolve()),
                           '--worker', '--run-dir', str(run), '--settings', str(args.settings.resolve()),
                           '--mode',args.mode,'--suite',args.suite]
                if args.source_run:
                    command += ['--source-run',*[str(p.resolve()) for p in args.source_run]]
                if args.mode=='trigger-comparison':
                    if args.compare_event_dedup:
                        command += ['--compare-event-dedup']
                    command += ['--trigger-task-id',str(args.trigger_task_id)]
                    if args.static_q is not None:
                        command += ['--static-q',str(args.static_q)]
                    command += ['--repeats',str(args.repeats)]
                    if args.device_dvfs:
                        command += ['--device-dvfs']
                    if args.no_age_limit:
                        command += ['--no-age-limit']
                    if args.compare_device:
                        command += ['--compare-device']
                    if args.trigger_suite:
                        command += ['--trigger-suite',args.trigger_suite]
                    if args.strategies:
                        command += ['--strategies',*args.strategies]
                if not uses_dvfs:
                    command = [cfg['python_executable'],str(ROOT/'scripts/guarded_run.py'),'--',*command]
                with (run/'worker.log').open('w') as log:
                    if uses_dvfs:
                        from scripts.profile_clock_screen import monitored_worker
                        if args.mode=='watchdog':
                            from perceptbudget.watchdog import InferenceTimeout, validate_fault_suite
                            try:
                                code=monitored_worker(command,log,request_watchdog=run/'request_watchdog.json',watchdog_grace_seconds=3.)
                            except InferenceTimeout:
                                result=validate_fault_suite(run)
                                return 0
                            raise RuntimeError('Watchdog suite returned without observing the injected hang')
                        else:
                            code = monitored_worker(command, log)
                    else:
                        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                if code:
                    tail = (run/'worker.log').read_text()[-2500:]
                    raise RuntimeError(f'Recording worker exit{code}; {tail}')
                result = json.loads((run/'summary.json').read_text())
                skipped=result.get('skipped_episodes',[])
                probe_strategy=(args.strategies or [None])[-1]
                no_target_probe=(args.strategies in (['events_and_variance','events_and_variance_open_probe'],['events_and_variance','events_and_variance_slowdown_probe'])
                    and result.get('diagnostic_status')=='not_applicable'
                    and len(result.get('episodes',[]))==1 and result['episodes'][0].get('view')=='events_and_variance'
                    and result['episodes'][0].get('audit_passed') is True
                    and len(skipped)==1 and skipped[0].get('view')==probe_strategy
                    and skipped[0].get('reason_code')==('no_slowdown_refresh' if probe_strategy=='events_and_variance_slowdown_probe' else 'no_comparable_open_event'))
                if result['status'] != 'completed' or (args.mode not in ('dvac-probe','dvac-validate') and
                        len(result.get('episodes', [])) != len(settings['calibration_tasks'])*variants_per_task and not no_target_probe):
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
