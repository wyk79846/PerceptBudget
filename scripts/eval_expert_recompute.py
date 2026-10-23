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
    args=parser.parse_args()
    config=load_config(repo_root=ROOT);validate(config)
    split_delay = args.split_delay or args.single_insertion
    conditions = ['full','kv_once'] if args.single_insertion else ['full','kv_reuse','kv_reuse_split'] if split_delay else ['full','kv_reuse']
    if args.dry_run:
        print(json.dumps(dict(suite=args.suite,task_id=args.task_id,episodes=len(conditions),
            conditions=conditions,reuse='Exactly one expert-only request after bootstrap; all other requests Full' if args.single_insertion else 'Full/expert-only alternating; bootstrap always Full',
            split_delay_comparison=split_delay, single_insertion=args.single_insertion,
            H=50,N=5,q=5,state=True,state_on_reuse='last Full state remains in cached KV',
            original_kv_expiry_enforced=True,num_views=2,pool=1,seed=7,init=0,model_loads=1)))
        return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Worker requires an M2 run directory')
        from perceptbudget.device_quality import resident_quality
        resident_quality(config,args.run_dir,'full',cleanup_gate=CLEANUP_GATE,
            task_plan=[(args.suite,args.task_id)],recompute_comparison=True,split_delay_comparison=split_delay,
            single_insertion=args.single_insertion)
        return 0
    cfg=dict(config,m2_profile=True,m2_expert_recompute=True)
    run=create_run(cfg,'M2','expert-recompute-once' if args.single_insertion else 'expert-recompute-split' if args.split_delay else 'expert-recompute',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    summary=dict(status='running',split_delay_comparison=split_delay,single_insertion=args.single_insertion,
        scope='paired single-task episodes, no population success-rate or energy claim')
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            command=[config['python_executable'],str(ROOT/'scripts/guarded_run.py'),'--',
                config['python_executable'],'-u',str(Path(__file__).resolve()),'--worker',
                '--run-dir',str(run),'--suite',args.suite,'--task-id',str(args.task_id)]
            if args.split_delay:command+=['--split-delay']
            if args.single_insertion:command+=['--single-insertion']
            with (run/'worker.log').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            summary['worker_exit_code']=completed.returncode
            if completed.returncode:
                raise RuntimeError('Worker failed or memory guard stopped it; do not count as task failure')
            record=json.loads((run/'full_quality.json').read_text())
            if record['status']!='completed' or len(record['episodes'])!=len(conditions):
                raise RuntimeError('Incomplete paired output')
            first=record['episodes'][0]
            summary.update(model_loads=record['model_loads'],suffix_loads=record['suffix_loads'],
                paired_initial_hashes_equal=all(first['initial']['initial_hashes']==e['initial']['initial_hashes'] for e in record['episodes']),
                mode_delay_calibration=record.get('mode_delay_calibration'),
                episodes=[])
            for e in record['episodes']:
                requests=json.loads((Path(e['run'])/'rtc/q5_requests.json').read_text())
                if args.single_insertion:
                    expert_ids=[r['prediction_id'] for r in requests if r['kv_recompute']['mode']=='expert_only']
                    expected=[1] if e['view']=='kv_once' else []
                    if expert_ids!=expected:
                        raise RuntimeError('Single insertion mode did not execute exactly request1 as Fast')
                groups={}
                for mode in ('full','expert_only'):
                    rows=[r for r in requests[1:] if r['kv_recompute']['mode']==mode]
                    groups[mode]=dict(requests=len(rows),
                        inference_median_ms=statistics.median((r['inference_ended_monotonic']-r['inference_started_monotonic'])*1000 for r in rows) if rows else None,
                        context_median_ms=statistics.median(r['kv_recompute']['context_seconds']*1000 for r in rows) if rows else None,
                        estimated_delay_steps=[r['rtc_guidance']['delay_estimate_steps'] for r in rows],
                        actual_delay_steps=[r['rtc_guidance'].get('actual_delay_steps') for r in rows],
                        max_age_steps=max((r['kv_recompute']['age_steps_at_request'] for r in rows),default=0))
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
