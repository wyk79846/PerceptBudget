#!/usr/bin/env python3
"""Opt-in small real-async M4 joint-selection test; one resident model."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run,_write_json_atomically
from perceptbudget.closed_loop import load_settings,recording_session,clock_preflight
from perceptbudget.baseline import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suites',nargs='+',choices=('libero_spatial','libero_object','libero_goal','libero_10'),default=['libero_spatial'])
    parser.add_argument('--task-id',type=int,choices=range(10),default=0)
    parser.add_argument('--acceptance-group',choices=('q30','joint','m3','m3_fast'),help='Fixed four-suite/three-task acceptance group')
    parser.add_argument('--first-tasks',type=int,choices=(3,12),default=12,help='Use the first3 fixed tasks for an M3 screen')
    parser.add_argument('--task-indices',type=int,nargs='+',choices=range(12),help='Explicit canonical task indices for an M3 paired screen')
    parser.add_argument('--result-index',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--resume-from',type=Path,help='Continue audited completed joint tasks from a saved group')
    parser.add_argument('--cover-close-slowdown',action='store_true',help='Experimental timely Full close also covers following slowdown within existing5-step range')
    parser.add_argument('--m3-eta-ms',type=int,choices=(25,50),default=25,help='M3 safety margin;50 reproduces original delivery setting')
    parser.add_argument('--reuse-preparation',action='store_true',help='Experimental setup reuse; pending cross-task GPU parity validation')
    parser.add_argument('--fast-advance','--fast-reserve-full',dest='fast_reserve_full',action='store_true',help='Advance Fast submission by its complete recovery-to-commit budget plus margin, rounded to control steps')
    parser.add_argument('--fast-gripper-gate',action='store_true',help='Fast only after executed open command; closed or unknown requires Full')
    parser.add_argument('--full-only',action='store_true',help='M3 Fast ablation: retain window/controller but forbid Fast candidates')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--prepare-environments',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--environment-cache-from',type=Path,help='Reuse a passed identical M4 task environment preflight')
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.fast_reserve_full and (args.acceptance_group!='m3_fast' or args.full_only):parser.error('--fast-reserve-full requires m3_fast without --full-only')
    if args.fast_gripper_gate and not args.fast_reserve_full:parser.error('--fast-gripper-gate requires --fast-advance')
    if args.full_only and args.acceptance_group!='m3_fast':parser.error('--full-only requires m3_fast')
    if args.first_tasks!=12 and args.acceptance_group not in ('m3','m3_fast'):parser.error('Three-task screen is for the M3 candidate')
    if args.task_indices and (args.acceptance_group not in ('m3','m3_fast') or args.first_tasks!=12 or len(set(args.task_indices))!=len(args.task_indices)):
        parser.error('Explicit indices require M3, default first-tasks and unique indices')
    if args.m3_eta_ms!=25 and args.acceptance_group not in ('m3','m3_fast'):parser.error('Margin override requires M3')
    cfg=load_config(repo_root=ROOT);validate(cfg)
    cfg['experimental_setup_reuse']=args.reuse_preparation
    settings=load_settings(ROOT/'m3.json');m4=json.loads((ROOT/'m4.json').read_text())
    from perceptbudget.m4_planner import POLICY_REVISION
    cfg['m4_policy_revision']=POLICY_REVISION
    settings.update(mode='deadline_dvfs',calibration_tasks=[[s,args.task_id] for s in args.suites])
    settings['timing'].update(m4['timing'],maximum_action_offset=50,maximum_observation_age_seconds=None,margin_updates=False)
    cfg.update(m4_enabled=True,m4_settings=m4,m4_cover_close_slowdown=args.cover_close_slowdown,m3_action_events_enabled=True,m3_dvac_enabled=False,
        m3_event_coverage_mode='event_only',m3_signal_policy='earliest',m3_merge_slowdown_steps=None)
    if args.acceptance_group:
        from perceptbudget.m4_acceptance import manifest
        fixed=manifest(cfg)
        cfg.update(m4_acceptance_group=args.acceptance_group,m4_acceptance_manifest=fixed)
        if args.acceptance_group=='joint':
            cfg.update(m4_fast_prompt_capacity=20,
                m4_fast_cost_scope='M2 costs are initial estimates from exact no-state text shapes; '
                'acceptance uses fixed20 (maximum across the12 fixed tasks) capacity with valid-token masking; task calibration measures actual latency, '
                'whole-task energy is measured independently')
        settings['calibration_tasks']=fixed['tasks']
        if args.acceptance_group=='q30':
            settings['mode']='record_only'
            cfg.update(m4_enabled=False,m3_hold_high_clocks=True)
        if args.acceptance_group in ('m3','m3_fast'):
            from perceptbudget.m4_acceptance import configure_m3,configure_m3_fast
            (configure_m3_fast if args.acceptance_group=='m3_fast' else configure_m3)(cfg,settings)
            cfg['m3_fast_reserve_full']=False
            cfg['m3_fast_advance']=args.fast_reserve_full
            cfg['m3_fast_gripper_gate']=args.fast_gripper_gate
            if args.fast_reserve_full:cfg['m3_fast_policy']='m3_service_ledger_fast_advance_v4'
            cfg['m3_fast_full_only']=args.full_only
            settings['timing']['safety_margin_seconds']=args.m3_eta_ms/1000
            cfg['m3_replay_eta_ms']=args.m3_eta_ms
            settings['calibration_tasks']=[fixed['tasks'][i] for i in args.task_indices] if args.task_indices else fixed['tasks'][:args.first_tasks]
            cfg['m3_comparison_tasks']=settings['calibration_tasks']
    continuation=None
    if args.resume_from:
        if args.acceptance_group!='joint':parser.error('Resume is only available for joint acceptance')
        from perceptbudget.m4_acceptance import resume_joint
        continuation=resume_joint(args.resume_from,cfg)
        cfg['m3_resume_plan']=continuation
    pending_tasks=continuation['remaining_tasks'] if continuation else settings['calibration_tasks']
    if args.dry_run:
        print(json.dumps(dict(tasks=pending_tasks,reused_episodes=len(continuation['episodes']) if continuation else 0,group=args.acceptance_group or 'joint',model_loads=1,Full=True,Fast='pool2_agent_no_state' if cfg['m4_enabled'] or cfg.get('m3_fast_enabled') else 'disabled',
            eta_ms=None if args.acceptance_group=='q30' else args.m3_eta_ms if args.acceptance_group in ('m3','m3_fast') else 25,H=50,N=5,cover_close_slowdown=args.cover_close_slowdown,
            compute_profiles=['baseline','gpu_emc'] if cfg['m4_enabled'] else ['baseline'],
            policies=10 if cfg['m4_enabled'] else 1,
            fast_gripper_gate=args.fast_gripper_gate,fast_reserve_full=args.fast_reserve_full,full_only=args.full_only,m3_fast_selection=bool(cfg.get('m3_fast_enabled')),perception_policy=cfg.get('m3_fast_policy'),source='original M3 D_joint DeadlineRuntime, Full-high/low-Continue, explicit replay margin' if args.acceptance_group in ('m3','m3_fast') else 'fixed q30 Full high-frequency actual async' if args.acceptance_group=='q30' else 'complete common-window plans, no inference in Continue')));return 0
    if args.prepare_environments:
        if not args.acceptance_group or args.run_dir is None or args.run_dir.resolve().parent!=(Path(cfg['output_root'])/'M4').resolve():
            parser.error('Environment preparation requires acceptance group and M4 directory')
        if args.environment_cache_from:
            from perceptbudget.m4_acceptance import copy_precompiled_environment_cache,precompiled_environments
            reused=copy_precompiled_environment_cache(args.environment_cache_from,args.run_dir,pending_tasks,cfg)
            with precompiled_environments(args.run_dir,pending_tasks,cfg):
                pass
            print('ENVIRONMENT_CACHE_REUSED='+json.dumps(reused),flush=True)
        else:
            from perceptbudget.closed_loop import compiled_environments
            with compiled_environments(args.run_dir,pending_tasks,cfg):
                pass
        print('ENVIRONMENT_PREPARATION=completed',flush=True)
        return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent!=(Path(cfg['output_root'])/'M4').resolve():
            parser.error('Worker requires M4 directory')
        subprocess.run(['sudo','-n','true'],check=True)
        cfg.update(m3_watchdog_path=str(args.run_dir/'request_watchdog.json'),m3_request_timeout_seconds=3.)
        if args.acceptance_group:cfg['m4_precompiled_environments']=True
        from scripts.eval_pooled_views import CLEANUP_GATE
        result=recording_session(cfg,settings,args.run_dir,CLEANUP_GATE)
        # Audit selected compute frequency, not M3's fixed high-frequency assumption.
        from perceptbudget.m4_acceptance import clock_audit
        audits=[];service_ledgers=[]
        from perceptbudget.service_ledger import write_run_ledger
        for row in result['episodes']:
            run=Path(row['run']);audit=clock_audit(run)
            _write_json_atomically(run/'m4_clock_audit.json',audit);audits.append(audit)
            if not audit['passed']:raise RuntimeError('M4 frequency/power/restore audit failed')
            service_ledgers.append(write_run_ledger(run,forbid_hot_fast=bool(cfg.get('m3_fast_advance'))))
        result.update(mode=('m3_fast_selection' if args.acceptance_group=='m3_fast' else 'm3_frozen_joint') if args.acceptance_group in ('m3','m3_fast') else 'm4_'+(args.acceptance_group or 'joint'),m4_clock_audits=audits,service_continue_ledgers=service_ledgers)
        _write_json_atomically(args.run_dir/'summary.json',result)
        return 0
    subprocess.run(['sudo','-v'],check=True)
    run=create_run(dict(cfg,m4_tasks=settings['calibration_tasks']),'M4',
        'acceptance-'+args.acceptance_group if args.acceptance_group else 'joint-smoke',repo_root=ROOT)
    if args.result_index:
        _write_json_atomically(args.result_index,dict(run=str(run)))
    print('run_dir='+str(run),flush=True);status=dict(status='running')
    stage_times={}
    def finish_stage(name,started):
        stage_times[name]=time.monotonic()-started
        _write_json_atomically(run/'stage_timings.json',stage_times)
        print('ACCEPTANCE_STAGE='+json.dumps(dict(stage=name,seconds=round(stage_times[name],3))),flush=True)
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);clock_preflight(run)
            command=[cfg['python_executable'],'-u',str(Path(__file__).resolve()),'--worker','--run-dir',str(run),
                '--task-id',str(args.task_id),'--suites',*args.suites]
            if args.acceptance_group in ('m3','m3_fast'):command.extend(['--m3-eta-ms',str(args.m3_eta_ms)])
            if args.fast_reserve_full:command.append('--fast-reserve-full')
            if args.fast_gripper_gate:command.append('--fast-gripper-gate')
            if args.full_only:command.append('--full-only')
            if args.reuse_preparation:command.append('--reuse-preparation')
            if args.cover_close_slowdown:command.append('--cover-close-slowdown')
            if args.acceptance_group:command.extend(['--acceptance-group',args.acceptance_group])
            command.extend(['--first-tasks',str(args.first_tasks)])
            if args.task_indices:command.extend(['--task-indices',*map(str,args.task_indices)])
            if args.resume_from:command.extend(['--resume-from',str(args.resume_from)])
            from scripts.profile_clock_screen import monitored_worker
            if args.acceptance_group:
                preparation=[cfg['python_executable'],'-u',str(Path(__file__).resolve()),
                    '--prepare-environments','--acceptance-group',args.acceptance_group,'--run-dir',str(run)]
                if args.acceptance_group in ('m3','m3_fast'):preparation.extend(['--m3-eta-ms',str(args.m3_eta_ms)])
                if args.reuse_preparation:preparation.append('--reuse-preparation')
                if args.environment_cache_from:preparation.extend(['--environment-cache-from',str(args.environment_cache_from)])
                preparation.extend(['--first-tasks',str(args.first_tasks)])
                if args.task_indices:preparation.extend(['--task-indices',*map(str,args.task_indices)])
                if args.resume_from:preparation.extend(['--resume-from',str(args.resume_from)])
                started=time.monotonic()
                print('ACCEPTANCE_STAGE=environment_preparation_started',flush=True)
                with (run/'environment_preparation.log').open('w') as log:
                    prepared=monitored_worker(preparation,log)
                finish_stage('environment_preparation',started)
                if prepared:raise RuntimeError('Environment preparation failed; no model loaded')
            started=time.monotonic()
            print('ACCEPTANCE_STAGE=model_worker_started',flush=True)
            with (run/'worker.log').open('w') as log:
                code=monitored_worker(command,log,request_watchdog=run/'request_watchdog.json',watchdog_grace_seconds=3.)
            finish_stage('model_worker',started)
            if code:raise RuntimeError('Worker failed; inspect '+str(run/'worker.log'))
            result=json.loads((run/'summary.json').read_text())
            if result['status']!='completed' or len(result['episodes'])!=len(settings['calibration_tasks']):raise RuntimeError('Incomplete M4 test')
            status=result
    except BaseException as exc:
        status=dict(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        _write_json_atomically(run/'summary.json',status)
        meta=json.loads((run/'run.json').read_text());meta['status']=status['status'];_write_json_atomically(run/'run.json',meta)
        print('M4_TERMINAL_STATUS='+status['status'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
