#!/usr/bin/env python3
"""Compare original M3 or current M4 against a reusable fixed-q30 task set."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run,_write_json_atomically
from perceptbudget.m4_acceptance import manifest,summarize,clock_audit
from perceptbudget.m4_planner import POLICY_REVISION


def validate_baseline(source,fixed):
    source=Path(source)
    meta=json.loads((source/'run.json').read_text())['config_snapshot']
    result=json.loads((source/'summary.json').read_text())
    if (meta.get('m4_acceptance_group')!='q30' or
            meta.get('m4_acceptance_manifest')!=fixed or result.get('status')!='completed'):
        raise ValueError('Baseline must be a completed q30 group with the identical fixed manifest')
    expected={tuple(t) for t in fixed['tasks']}
    if {(e['suite'],e['task_id']) for e in result['episodes']}!=expected or len(result['episodes'])!=len(expected):
        raise ValueError('Incomplete baseline task list')
    for episode in result['episodes']:
        run=Path(episode['run'])
        cfg=json.loads((run/'run.json').read_text())['config_snapshot']
        audit=json.loads((run/'system_audit.json').read_text())
        if (cfg.get('candidate_q')!=[30] or cfg.get('m4_enabled') or
                cfg.get('m3_scheduler_mode')!='record_only' or not cfg.get('m3_hold_high_clocks') or
                not audit['passed'] or not clock_audit(run)['passed']):
            raise ValueError('Baseline configuration/audit mismatch: '+str(run))
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run',type=Path,help='Reuse the completed q30 group path from an earlier session')
    parser.add_argument('--candidate',choices=('joint','m3','m3_fast'),default='m3',help='Default: frozen M3; Fast and joint scheduling require explicit selection')
    parser.add_argument('--first-tasks',type=int,choices=(3,12),default=12,help='M3 screen on the first3 fixed tasks; reuse full q30 record')
    parser.add_argument('--task-indices',type=int,nargs='+',choices=range(12),help='Explicit canonical task indices for an M3 paired screen')
    parser.add_argument('--m3-reference',type=Path,help='Existing combined M3 result JSON for an additional per-task comparison')
    parser.add_argument('--resume-joint',type=Path,help='Keep completed joint outcomes and run only missing tasks')
    parser.add_argument('--m3-eta-ms',type=int,choices=(25,50),default=25,help='M3 safety margin;50 reproduces original delivery setting')
    parser.add_argument('--reuse-preparation',action='store_true',help='Experimental setup reuse; pending cross-task GPU parity validation')
    parser.add_argument('--environment-cache-from',type=Path,help='Reuse an already passed, identical task environment preflight')
    parser.add_argument('--fast-advance','--fast-reserve-full',dest='fast_reserve_full',action='store_true',help='Advance Fast submission by its complete Fast recovery-to-commit budget plus margin, rounded to control steps')
    parser.add_argument('--fast-selective416',action='store_true',help='Experimental416 visual tokens with state retained')
    parser.add_argument('--critical-wait-phases',action='store_true',help='With critical wait: one completed wait per finite predicted motion phase')
    parser.add_argument('--low-full',action='store_true',help='M3 only: deadline-first Full frequency selection; energy recorded only; no Fast')
    parser.add_argument('--phase-refresh',action='store_true',help='M3: suppress repeated event refresh within an already refreshed motion phase')
    parser.add_argument('--first-refresh-step',type=int,choices=(30,),help='M3 experiment: first event-free refresh request at step30')
    parser.add_argument('--critical-wait',action='store_true',help='M3 only: stop simulation at a predicted key action until the pending refresh arrives')
    parser.add_argument('--fast-key-phases',action='store_true',help='Experimental320 key-phase protection and one Full response')
    parser.add_argument('--full-only',action='store_true',help='M3 Fast ablation: retain window/controller but forbid Fast candidates')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.low_full and args.candidate!='m3':parser.error('--low-full requires M3 without Fast')
    if args.phase_refresh and args.candidate!='m3':parser.error('--phase-refresh requires M3')
    if args.first_refresh_step is not None and args.candidate!='m3':parser.error('--first-refresh-step requires M3')
    if args.critical_wait_phases and not args.critical_wait:parser.error('--critical-wait-phases requires --critical-wait')
    if args.critical_wait and args.candidate!='m3':parser.error('--critical-wait requires M3 without Fast')
    if args.fast_reserve_full and (args.candidate!='m3_fast' or args.full_only):parser.error('--fast-reserve-full requires m3_fast without --full-only')
    if args.fast_selective416 and (not args.fast_reserve_full or args.full_only):parser.error('--fast-selective416 requires --fast-advance without --full-only')
    if args.fast_key_phases and (not args.fast_reserve_full or args.fast_selective416 or args.full_only):parser.error('--fast-key-phases requires --fast-advance with320 Fast')
    if args.full_only and args.candidate!='m3_fast':parser.error('--full-only requires m3_fast')
    if args.first_tasks!=12 and (args.candidate not in ('m3','m3_fast') or not args.baseline_run):
        parser.error('Three-task screen requires --candidate m3 and --baseline-run')
    if args.task_indices and (args.candidate not in ('m3','m3_fast') or not args.baseline_run or args.first_tasks!=12 or len(set(args.task_indices))!=len(args.task_indices)):
        parser.error('Explicit indices require M3, saved baseline, default first-tasks and unique indices')
    if args.m3_eta_ms!=25 and args.candidate not in ('m3','m3_fast'):parser.error('Margin override requires M3')
    if args.m3_reference and args.candidate!='m3_fast':parser.error('M3 reference is for m3_fast')
    if args.m3_reference and not args.m3_reference.is_file():parser.error('M3 reference file missing')
    if args.environment_cache_from and not args.environment_cache_from.is_dir():parser.error('Environment cache source must be a completed M4 run directory')
    cfg=load_config(repo_root=ROOT);fixed=manifest(cfg)
    task_plan=[fixed['tasks'][i] for i in args.task_indices] if args.task_indices else fixed['tasks'][:args.first_tasks]
    continuation=None
    if args.resume_joint:
        if args.candidate!='joint':parser.error('--resume-joint is only for the joint candidate')
        if not args.baseline_run:parser.error('Joint continuation requires --baseline-run')
        from perceptbudget.m4_acceptance import resume_joint
        continuation=resume_joint(args.resume_joint,dict(cfg,m4_settings=json.loads((ROOT/'m4.json').read_text())))
    if args.baseline_run:validate_baseline(args.baseline_run,fixed)
    groups=[args.candidate] if args.baseline_run else ['q30',args.candidate]
    if args.dry_run:
        print(json.dumps(dict(manifest=fixed,tasks_to_run=task_plan,groups_to_run=groups,episodes=(len(continuation['remaining_tasks']) if continuation else len(task_plan)*len(groups)),
            reused_joint_episodes=len(continuation['episodes']) if continuation else 0,
            low_full=args.low_full,phase_refresh=args.phase_refresh,first_refresh_step=args.first_refresh_step,critical_wait_phases=args.critical_wait_phases,critical_wait=args.critical_wait,fast_key_phases=args.fast_key_phases,fast_selective416=args.fast_selective416,fast_reserve_full=args.fast_reserve_full,full_only=args.full_only,model_loads=len(groups),candidate=args.candidate,joint_version=POLICY_REVISION if args.candidate=='joint' else f'{args.candidate}_D_joint_eta{args.m3_eta_ms}',signal='events_only; PACE-inspired key-phase guard and one Full response' if args.fast_key_phases else 'events_only; no PACE rule; close-slowdown coverage off'),ensure_ascii=False))
        return 0
    subprocess.run(['sudo','-v'],check=True)
    session=create_run(dict(cfg,m4_acceptance_manifest=fixed,m4_comparison_tasks=task_plan),'M4','paired-acceptance',repo_root=ROOT)
    print('session_dir='+str(session),flush=True)
    _write_json_atomically(session/'fixed_manifest.json',fixed)
    sources={'q30':str(args.baseline_run.resolve())} if args.baseline_run else {}
    status=dict(status='running',sources=sources,groups_to_run=groups)
    _write_json_atomically(session/'summary.json',status)
    try:
        for group in groups:
            index=session/(group+'_index.json');log=session/(group+'.log')
            print('START '+group+' log='+str(log),flush=True)
            command=[cfg['python_executable'],'-u',str(ROOT/'scripts/run_m4.py'),
                '--acceptance-group',group,'--result-index',str(index),'--first-tasks',str(args.first_tasks)]
            if group in ('m3','m3_fast'):command.extend(['--m3-eta-ms',str(args.m3_eta_ms)])
            if args.fast_reserve_full:command.append('--fast-reserve-full')
            if args.fast_selective416:command.append('--fast-selective416')
            if args.fast_key_phases:command.append('--fast-key-phases')
            if args.critical_wait and group=='m3':command.append('--critical-wait')
            if args.critical_wait_phases and group=='m3':command.append('--critical-wait-phases')
            if args.first_refresh_step is not None and group=='m3':command.extend(['--first-refresh-step',str(args.first_refresh_step)])
            if args.phase_refresh and group=='m3':command.append('--phase-refresh')
            if args.low_full and group=='m3':command.append('--low-full')
            if args.full_only:command.append('--full-only')
            if args.reuse_preparation:command.append('--reuse-preparation')
            if args.environment_cache_from:command.extend(['--environment-cache-from',str(args.environment_cache_from)])
            if args.task_indices:command.extend(['--task-indices',*map(str,args.task_indices)])
            if group=='joint' and args.resume_joint:command.extend(['--resume-from',str(args.resume_joint)])
            with log.open('w') as stream:
                code=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
            if index.exists():sources[group]=json.loads(index.read_text())['run']
            _write_json_atomically(session/'summary.json',dict(status='running',sources=sources,groups_to_run=groups))
            if code:raise RuntimeError(group+' failed; inspect '+str(log)+'; completed episodes remain saved')
            if group=='q30':validate_baseline(sources[group],fixed)
            print('DONE '+group+' run='+sources[group],flush=True)
        result=summarize(session,sources,task_plan=task_plan)
        if args.m3_reference:
            from perceptbudget.m4_acceptance import summarize_m3_increment
            summarize_m3_increment(session,args.m3_reference,result)
        _write_json_atomically(session/'summary.json',result)
        print('M4_ACCEPTANCE_STATUS=completed',flush=True)
    except BaseException as exc:
        _write_json_atomically(session/'summary.json',dict(status='invalid',sources=sources,error=f'{type(exc).__name__}: {exc}'))
        print('M4_ACCEPTANCE_STATUS=invalid',flush=True)
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
