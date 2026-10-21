#!/usr/bin/env python3
"""M2 dual-camera token pooling: resident model, isolated geometry per factor."""
import argparse
import fcntl
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import load_config, create_run, _write_json_atomically
from perceptbudget.baseline import validate, SUITES

CLEANUP_GATE = Path('/home/wyk/workspace/任务产出/perceptbudget/M2/'
                    '20261021T101001.012984Z-device-discovery-3a32022b')
SOURCE_RUN = Path('/home/wyk/workspace/任务产出/perceptbudget/M1/'
                  '20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-factor', type=int, nargs='+', choices=(2,4), required=True)
    parser.add_argument('--pool-view', nargs='+', choices=('both','agent','wrist'),default=['both'],
                        help='Camera to pool: agent=external, wrist=hand camera; other camera stays full')
    parser.add_argument('--suite', choices=SUITES, default='libero_spatial')
    parser.add_argument('--task-ids', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--probe-only', action='store_true', help='Fixed-input two requests; no closed-loop episodes')
    parser.add_argument('--source-run', type=Path, default=SOURCE_RUN)
    parser.add_argument('--cleanup-gate', type=Path, default=CLEANUP_GATE)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.pool_view))!=len(args.pool_view) or len(set(args.pool_factor)) != len(args.pool_factor) or not args.task_ids or \
       len(set(args.task_ids)) != len(args.task_ids) or any(not 0<=t<10 for t in args.task_ids):
        parser.error('Use distinct factors and distinct task IDs0..9')
    config = load_config(repo_root=ROOT)
    validate(config)
    if args.dry_run:
        print(json.dumps(dict(factors=args.pool_factor, suite=args.suite,
            tasks=args.task_ids, pool_views=args.pool_view, episodes=0 if args.probe_only else len(args.task_ids)*len(args.pool_factor)*len(args.pool_view),
            q=5,H=50,N=5,state=True,num_views=2,seed=7,init=0,
            encoded_visual_tokens={str(f):512//f**2 for f in args.pool_factor},
            mixed_visual_tokens={str(f):256+256//f**2 for f in args.pool_factor},
            groups=[dict(pooled_view=v,factor=f,visual_tokens=512//f**2 if v=='both' else 256+256//f**2,
                         tasks=len(args.task_ids)) for v,f in product(args.pool_view,args.pool_factor)],
            model_lifetimes=len(args.pool_factor)*len(args.pool_view),flashrt_modified=False),ensure_ascii=False,indent=2))
        return 0
    if args.worker:
        if len(args.pool_factor)!=1 or len(args.pool_view)!=1 or args.run_dir is None or \
           args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Worker requires a single factor and an M2 run directory')
        cfg = dict(config,m2_profile=True,m2_view='full',m2_vision_pool_factor=args.pool_factor[0],m2_pool_view=args.pool_view[0])
        if args.probe_only:
            from perceptbudget.device_profile import fixed_view_probe
            fixed_view_probe(cfg,args.run_dir,args.source_run,'full',repeats=2)
        else:
            from perceptbudget.device_quality import resident_quality
            resident_quality(cfg,args.run_dir,'full',cleanup_gate=args.cleanup_gate,
                task_plan=[(args.suite,t) for t in args.task_ids])
        return 0
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for pooled_view,factor in product(args.pool_view,args.pool_factor):
            cfg = dict(config,m2_profile=True,m2_view='full',m2_vision_pool_factor=factor,m2_pool_view=pooled_view)
            run = create_run(cfg,'M2',f'token-pool{factor}-{pooled_view}'+('-probe' if args.probe_only else '-quality'),repo_root=ROOT)
            print('run_dir='+str(run),flush=True)
            command=[config['python_executable'],str(ROOT/'scripts/guarded_run.py'),'--',
                config['python_executable'],'-u',str(Path(__file__).resolve()),'--worker',
                '--run-dir',str(run),'--pool-factor',str(factor),'--pool-view',pooled_view,'--suite',args.suite,
                '--task-ids',*map(str,args.task_ids),'--cleanup-gate',str(args.cleanup_gate)]
            if args.probe_only:
                command+=['--probe-only','--source-run',str(args.source_run)]
            summary=dict(status='running',pool_factor=factor,pooled_view=pooled_view,suite=args.suite,
                task_ids=args.task_ids,q=5,num_views=2,probe_only=args.probe_only)
            try:
                _write_json_atomically(run/'summary.json',summary)
                with (run/'worker.log').open('w') as log:
                    completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                summary.update(status='completed' if completed.returncode==0 else 'invalid',
                               worker_exit_code=completed.returncode)
                if completed.returncode==0:
                    result=json.loads((run/('full_probe.json' if args.probe_only else 'full_quality.json')).read_text())
                    if result['status']!='completed':
                        raise RuntimeError('Worker exited without completed output')
                    if args.probe_only:
                        summary.update(visual_tokens=result['visual_tokens'],
                            inference_median_ms=result['measurements'][0]['median_ms'])
                    else:
                        summary.update(episodes=len(result['episodes']),
                            successes=sum(e['outcome']=='success' for e in result['episodes']),
                            model_loads=result['model_loads'],suffix_loads=result['suffix_loads'])
                else:
                    summary['error']='Worker failed or memory guard stopped it; see worker.log. Do not treat as a quality failure.'
            except BaseException as exc:
                summary.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                _write_json_atomically(run/'summary.json',summary)
                metadata=json.loads((run/'run.json').read_text())
                metadata['status']=summary['status']
                _write_json_atomically(run/'run.json',metadata)
                print('POOL_TERMINAL_STATUS='+json.dumps(dict(run=str(run),**summary)),flush=True)
            if summary['status']!='completed':
                return completed.returncode or 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
