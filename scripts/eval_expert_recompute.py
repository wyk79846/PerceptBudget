#!/usr/bin/env python3
"""One paired RTC task: fresh KV every call vs one stale-KV expert call between Full calls."""
import argparse
import fcntl
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    from perceptbudget.baseline import validate,SUITES
    from scripts.eval_pooled_views import CLEANUP_GATE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=SUITES,default='libero_spatial')
    parser.add_argument('--task-id',type=int,choices=range(10),default=0)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--split-delay',action='store_true',help='Three paired episodes: Full, shared-history reuse, per-mode-history reuse')
    parser.add_argument('--single-insertion',action='store_true',help='Two episodes: Full vs one expert-only insertion, with per-mode delay history')
    parser.add_argument('--cross-suite',action='store_true',help='Spatial/Object/Goal task0, Full vs repeated Full/Fast alternation, six episodes')
    visual_modes=parser.add_mutually_exclusive_group()
    visual_modes.add_argument('--visual-alternation',action='store_true',help='Fresh wrist-view context Fast instead of old KV; optionally combine --cross-suite')
    visual_modes.add_argument('--pooled-alternation',action='store_true',help='Fresh dual-view factor2 token pooling Fast, not old KV')
    parser.add_argument('--suites',nargs='+',choices=SUITES,help='Explicit subset for --cross-suite, avoids repeating completed task pairs')
    args=parser.parse_args()
    args.visual_alternation = args.visual_alternation or args.pooled_alternation
    visual_fast_mode='pool2' if args.pooled_alternation else 'wrist'
    config=load_config(repo_root=ROOT);validate(config)
    if args.cross_suite and (args.single_insertion or args.task_id != 0):
        parser.error('--cross-suite requires task0 and cannot use --single-insertion')
    if args.visual_alternation and (args.single_insertion or args.split_delay):
        parser.error('--visual-alternation cannot use old-KV insertion/split comparison flags')
    if args.suites and (not args.cross_suite or len(set(args.suites))!=len(args.suites)):
        parser.error('--suites requires --cross-suite and distinct suites')
    split_delay = args.split_delay or args.single_insertion or args.cross_suite
    conditions = ['full',visual_fast_mode+'_alternate'] if args.visual_alternation else ['full','kv_reuse_split'] if args.cross_suite else ['full','kv_once'] if args.single_insertion else ['full','kv_reuse','kv_reuse_split'] if split_delay else ['full','kv_reuse']
    task_plan = [(suite,0) for suite in (args.suites or SUITES)] if args.cross_suite else [(args.suite,args.task_id)]
    episode_count = len(conditions)*len(task_plan)
    if args.dry_run:
        print(json.dumps(dict(suite=args.suite,task_id=args.task_id,task_plan=task_plan,episodes=episode_count,
            conditions=conditions,reuse='Full/fresh-'+visual_fast_mode+' context alternating; bootstrap always Full' if args.visual_alternation else 'Exactly one expert-only request after bootstrap; all other requests Full' if args.single_insertion else 'Full/expert-only alternating; bootstrap always Full',
            split_delay_comparison=split_delay or args.visual_alternation, single_insertion=args.single_insertion,
            visual_alternation=args.visual_alternation, fast_visual_mode=visual_fast_mode if args.visual_alternation else None,
            fast_inputs='fresh dual images, factor2 pooling and current state' if args.pooled_alternation else 'fresh wrist image and current state' if args.visual_alternation else 'old KV',
            H=50,N=5,q=5,state=True,state_on_reuse='fresh state encoded on every call' if args.visual_alternation else 'last Full state remains in cached KV',
            original_kv_expiry_enforced=True,num_views=2,pool=1,seed=7,init=0,model_loads=1)))
        return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Worker requires an M2 run directory')
        from perceptbudget.device_quality import resident_quality
        resident_quality(config,args.run_dir,'full',cleanup_gate=CLEANUP_GATE,
            task_plan=task_plan,recompute_comparison=not args.visual_alternation,
            split_delay_comparison=split_delay and not args.visual_alternation,
            single_insertion=args.single_insertion,paired_split=args.cross_suite and not args.visual_alternation,
            visual_comparison=args.visual_alternation,visual_fast_mode=visual_fast_mode)
        return 0
    cfg=dict(config,m2_profile=True,m2_expert_recompute=not args.visual_alternation,
             m2_visual_alternation=args.visual_alternation,m2_visual_fast_mode=visual_fast_mode)
    run=create_run(cfg,'M2','fresh-pooled-alternation' if args.pooled_alternation else 'fresh-wrist-alternation' if args.visual_alternation else 'expert-recompute-cross-suite' if args.cross_suite else 'expert-recompute-once' if args.single_insertion else 'expert-recompute-split' if args.split_delay else 'expert-recompute',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    summary=dict(status='running',split_delay_comparison=split_delay or args.visual_alternation,
        visual_alternation=args.visual_alternation,fast_visual_mode=visual_fast_mode if args.visual_alternation else None,
        single_insertion=args.single_insertion,
        task_plan=task_plan, scope='paired task episodes, no population success-rate or energy claim')
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            command=[config['python_executable'],str(ROOT/'scripts/guarded_run.py'),'--',
                config['python_executable'],'-u',str(Path(__file__).resolve()),'--worker',
                '--run-dir',str(run),'--suite',args.suite,'--task-id',str(args.task_id)]
            if args.split_delay:command+=['--split-delay']
            if args.single_insertion:command+=['--single-insertion']
            if args.cross_suite:command+=['--cross-suite']
            if args.pooled_alternation:command+=['--pooled-alternation']
            elif args.visual_alternation:command+=['--visual-alternation']
            if args.suites:command+=['--suites',*args.suites]
            with (run/'worker.log').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            summary['worker_exit_code']=completed.returncode
            if completed.returncode:
                raise RuntimeError('Worker failed or memory guard stopped it; do not count as task failure')
            record=json.loads((run/'full_quality.json').read_text())
            if record['status']!='completed' or len(record['episodes'])!=episode_count:
                raise RuntimeError('Incomplete paired output')
            references={(e['suite'],e['task_id']):e['initial']['initial_hashes']
                        for e in record['episodes'] if e['view']=='full'}
            summary.update(model_loads=record['model_loads'],suffix_loads=record['suffix_loads'],
                paired_initial_hashes_equal=all(references[(e['suite'],e['task_id'])]==e['initial']['initial_hashes'] for e in record['episodes']),
                mode_delay_calibration=record.get('mode_delay_calibration'),
                mode_delay_calibrations=record.get('mode_delay_calibrations'),
                episodes=[])
            for e in record['episodes']:
                requests=json.loads((Path(e['run'])/'rtc/q5_requests.json').read_text())
                metadata_key='context_refresh' if args.visual_alternation else 'kv_recompute'
                fast_mode=visual_fast_mode if args.visual_alternation else 'expert_only'
                if args.single_insertion:
                    expert_ids=[r['prediction_id'] for r in requests if r['kv_recompute']['mode']=='expert_only']
                    expected=[1] if e['view']=='kv_once' else []
                    if expert_ids!=expected:
                        raise RuntimeError('Single insertion mode did not execute exactly request1 as Fast')
                if args.cross_suite or args.visual_alternation:
                    modes=[r[metadata_key]['mode'] for r in requests]
                    expected=['full' if e['view']=='full' or i%2==0 else fast_mode for i in range(len(modes))]
                    if modes!=expected or (e['view']!='full' and modes.count(fast_mode)<2):
                        raise RuntimeError('Cross-suite condition did not exercise repeated Full/Fast alternation')
                groups={}
                for mode in ('full',fast_mode):
                    rows=[r for r in requests[1:] if r[metadata_key]['mode']==mode]
                    groups[mode]=dict(requests=len(rows),
                        inference_median_ms=statistics.median((r['inference_ended_monotonic']-r['inference_started_monotonic'])*1000 for r in rows) if rows else None,
                        context_median_ms=statistics.median(r[metadata_key]['context_seconds']*1000 for r in rows) if rows else None,
                        estimated_delay_steps=[r['rtc_guidance']['delay_estimate_steps'] for r in rows],
                        actual_delay_steps=[r['rtc_guidance'].get('actual_delay_steps') for r in rows],
                        max_age_steps=max((r[metadata_key].get('age_steps_at_request',0) for r in rows),default=0))
                summary['episodes'].append(dict(e,request_groups=groups))
            if not summary['paired_initial_hashes_equal']:
                raise RuntimeError('Initial hashes do not match; not a paired quality comparison')
            summary['status']='completed'
    except BaseException as exc:
        summary.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        _write_json_atomically(run/'summary.json',summary)
        metadata=json.loads((run/'run.json').read_text());metadata['status']=summary['status']
        _write_json_atomically(run/'run.json',metadata)
        print('RECOMPUTE_TERMINAL_STATUS='+json.dumps(dict(run=str(run),status=summary['status'],error=summary.get('error'))),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
