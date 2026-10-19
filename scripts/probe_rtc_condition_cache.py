"""Gate/time exact conditioning cache against unmodified slow reference."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import create_run,load_config
from scripts import eval_q as base


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--step-graph',action='store_true')
    parser.add_argument('--packed-conditioning',action='store_true')
    args=parser.parse_args()
    if args.packed_conditioning and not args.step_graph:
        parser.error('--packed-conditioning requires --step-graph')
    config=load_config(repo_root=ROOT)
    config.update(H=25,N=5,state_enabled=True,state_prompt_capacity=80)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run=create_run(config,'M1','rtc-conditioning-cache',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        for name in ('scripts/probe_rtc_condition_cache.py','perceptbudget/rtc_condition_cache.py',
                     'perceptbudget/openpi_reference.py','perceptbudget/rtc_source_step_graph.py'):
            target=run/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        result=dict(status='running',scope='fixed recorded KV, original expert only; no full-request timing or task')
        try:
            import torch
            import numpy as np
            from perceptbudget.openpi_reference import OriginalOpenPiExpert,sample_original_rtc
            from perceptbudget.rtc_condition_cache import enable_condition_cache
            from scripts.probe_rtc_step import timed
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(7)
            data=np.load(Path(config['output_root'])/'M1/20261019T054917.649092Z-capacity-numeric-diagnosis-5f903047/capacity80a/trace.npz')
            def bits(x): return torch.from_numpy(x.copy()).to('cuda').view(torch.bfloat16)
            keys=torch.stack([bits(data[f'21_encoder_layer{i:02d}_key']) for i in range(18)])
            values=torch.stack([bits(data[f'21_encoder_layer{i:02d}_value']) for i in range(18)])
            expert=OriginalOpenPiExpert(config,25,5)
            noise=torch.randn(25,32,device='cuda')
            previous=torch.randn(25,32,device='cuda')*.3
            cases=((570,12,5),(563,10,10),(557,9,8))
            saved=[]
            graph_before=[]
            for valid,d,s in cases:
                expert.refresh_context(keys,values,valid)
                trace=[]
                raw,_=sample_original_rtc(expert,noise,previous[s:],d,s,trace=trace,author_order=True)
                saved.append((raw,trace,timed(torch,lambda:sample_original_rtc(expert,noise,previous[s:],d,s,author_order=True),count=3)))
                if args.packed_conditioning:
                    from perceptbudget.rtc_source_step_graph import SourceStepGraph
                    graph=SourceStepGraph(expert,noise)
                    graph_before.append(timed(torch,lambda:graph.replay(noise,previous[s:],d,s),count=3))
                    del graph
                    import gc
                    gc.collect();torch.cuda.empty_cache()
            if args.packed_conditioning:
                result['cache']=enable_condition_cache(expert,graph_live=True)
            if not args.step_graph:
                result['cache']=enable_condition_cache(expert)
            else:
                result['backend']='original_source_step_graph'
            result['cases']=[]
            for (valid,d,s),(reference,trace,before) in zip(cases,saved):
                expert.refresh_context(keys,values,valid)
                got=[]
                if args.step_graph:
                    import time
                    from perceptbudget.rtc_source_step_graph import SourceStepGraph
                    started=time.monotonic()
                    graph=SourceStepGraph(expert,noise)
                    capture_seconds=time.monotonic()-started
                    raw,_=graph.replay(noise,previous[s:],d,s,trace=got)
                else:
                    raw,_=sample_original_rtc(expert,noise,previous[s:],d,s,trace=got,author_order=True)
                per_step=[dict(step=k,**{name+'_max_abs':float((a[name]-b[name]).abs().max())
                    for name in ('input','velocity','clean','error','vjp','output')}) for k,(a,b) in enumerate(zip(trace,got))]
                after=(timed(torch,lambda:graph.replay(noise,previous[s:],d,s),count=3) if args.step_graph else
                    timed(torch,lambda:sample_original_rtc(expert,noise,previous[s:],d,s,author_order=True),count=3))
                error=float((raw-reference).abs().max())
                row=dict(valid=valid,delay=d,executed=s,raw_max_abs=error,per_step=per_step,
                    before=before,after=after,passed=bool(np.isfinite(error) and error<1e-4
                    and all(np.isfinite(v) and v<1e-4 for x in per_step for n,v in x.items() if n!='step')))
                result['cases'].append(row)
                if args.packed_conditioning:
                    row['graph_before']=graph_before[len(result['cases'])-1]
                if args.step_graph:
                    row['capture_seconds']=capture_seconds
                    del graph
                    import gc
                    gc.collect();torch.cuda.empty_cache()
                print('case='+json.dumps(row),flush=True)
            result.update(status='completed',passed=all(r['passed'] for r in result['cases']))
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json',result)
            print(json.dumps(result),flush=True)


if __name__=='__main__': main()
