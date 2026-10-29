#!/usr/bin/env python3
"""Opt-in small real-async M4 joint-selection test; one resident model."""
import argparse,fcntl,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run,_write_json_atomically
from perceptbudget.closed_loop import load_settings,recording_session,clock_preflight
from perceptbudget.baseline import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suites',nargs='+',choices=('libero_spatial','libero_object','libero_goal'),default=['libero_spatial'])
    parser.add_argument('--task-id',type=int,choices=range(10),default=0)
    parser.add_argument('--cover-close-slowdown',action='store_true',help='Experimental timely Full close also covers following slowdown within existing5-step range')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    cfg=load_config(repo_root=ROOT);validate(cfg)
    settings=load_settings(ROOT/'m3.json');m4=json.loads((ROOT/'m4.json').read_text())
    settings.update(mode='deadline_dvfs',calibration_tasks=[[s,args.task_id] for s in args.suites])
    settings['timing'].update(m4['timing'],maximum_action_offset=50,maximum_observation_age_seconds=None,margin_updates=False)
    cfg.update(m4_enabled=True,m4_settings=m4,m4_cover_close_slowdown=args.cover_close_slowdown,m3_action_events_enabled=True,m3_dvac_enabled=False,
        m3_event_coverage_mode='event_only',m3_signal_policy='earliest',m3_merge_slowdown_steps=None)
    if args.dry_run:
        print(json.dumps(dict(tasks=settings['calibration_tasks'],model_loads=1,Full=True,Fast='pool2_agent_no_state',
            eta_ms=25,H=50,N=5,cover_close_slowdown=args.cover_close_slowdown,compute_profiles=['baseline','gpu_emc'],policies=10,
            source='complete common-window plans, no inference in Continue')));return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent!=(Path(cfg['output_root'])/'M4').resolve():
            parser.error('Worker requires M4 directory')
        subprocess.run(['sudo','-n','true'],check=True)
        cfg.update(m3_watchdog_path=str(args.run_dir/'request_watchdog.json'),m3_request_timeout_seconds=3.)
        from scripts.eval_pooled_views import CLEANUP_GATE
        result=recording_session(cfg,settings,args.run_dir,CLEANUP_GATE)
        # Audit selected compute frequency, not M3's fixed high-frequency assumption.
        from perceptbudget.device_clock_screen import frequency_check
        audits=[]
        for row in result['episodes']:
            run=Path(row['run']);events=[]
            for line in (run/'rtc/runtime_clocks.jsonl').read_text().splitlines():
                try:events.append(json.loads(line))
                except ValueError:pass
            requests=json.loads((run/'rtc/q5_requests.json').read_text())
            checks=[frequency_check(events,q['m4_compute_profile'],q['observation_monotonic'],q['ready_monotonic']) for q in requests]
            ep=json.loads((run/'summary.json').read_text())['episode']
            passed=all(c['matches'] for c in checks) and 'RESTORED' in (run/'rtc/runtime_clocks.jsonl').read_text().splitlines() and ep['including_drain_power']['coverage_fraction']>=1-1e-9
            audit=dict(passed=passed,compute_windows=checks)
            _write_json_atomically(run/'m4_clock_audit.json',audit);audits.append(audit)
            if not passed:raise RuntimeError('M4 frequency/power/restore audit failed')
        result.update(mode='m4_joint',m4_clock_audits=audits)
        _write_json_atomically(args.run_dir/'summary.json',result)
        return 0
    subprocess.run(['sudo','-v'],check=True)
    run=create_run(dict(cfg,m4_tasks=settings['calibration_tasks']),'M4','joint-smoke',repo_root=ROOT)
    print('run_dir='+str(run),flush=True);status=dict(status='running')
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);clock_preflight(run)
            command=[cfg['python_executable'],'-u',str(Path(__file__).resolve()),'--worker','--run-dir',str(run),
                '--task-id',str(args.task_id),'--suites',*args.suites]
            if args.cover_close_slowdown:command.append('--cover-close-slowdown')
            from scripts.profile_clock_screen import monitored_worker
            with (run/'worker.log').open('w') as log:
                code=monitored_worker(command,log,request_watchdog=run/'request_watchdog.json',watchdog_grace_seconds=3.)
            if code:raise RuntimeError('Worker failed; inspect '+str(run/'worker.log'))
            result=json.loads((run/'summary.json').read_text())
            if result['status']!='completed' or len(result['episodes'])!=len(args.suites):raise RuntimeError('Incomplete M4 smoke test')
            status=result
    except BaseException as exc:
        status=dict(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        _write_json_atomically(run/'summary.json',status)
        meta=json.loads((run/'run.json').read_text());meta['status']=status['status'];_write_json_atomically(run/'run.json',meta)
        print('M4_TERMINAL_STATUS='+status['status'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
