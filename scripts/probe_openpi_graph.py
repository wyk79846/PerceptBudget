"""Fixed-input original OpenPI timing and graph equivalence; no rollout."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixed-context',action='store_true')
    parser.add_argument('--changing-lengths',action='store_true',help='exact-length recapture parity, not a rollout')
    args=parser.parse_args()
    config=load_config(repo_root=ROOT)
    config.update(H=20,N=5,state_enabled=True,state_prompt_capacity=80)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run=create_run(config,'M1','openpi-rtc-graph-probe',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        for name in ('scripts/probe_openpi_graph.py','perceptbudget/openpi_reference.py','perceptbudget/openpi_graph.py'):
            dest=run/'source'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest)
        result=dict(status='running',scope='fixed-input expert only; no FlashRT context timing or async task')
        try:
            import numpy as np
            import torch
            from perceptbudget.openpi_reference import OriginalOpenPiExpert,sample_original_rtc
            from perceptbudget.openpi_graph import CapturedOriginalRtc
            from scripts.probe_rtc_step import timed
            torch.use_deterministic_algorithms(True)
            prefix=Path(config['output_root'])/'M1/20261019T054917.649092Z-capacity-numeric-diagnosis-5f903047/capacity80a'
            reference=Path(config['output_root'])/'M1/20261019T092612.204175Z-openpi-expert-independent-7c7d1e10'
            expert=OriginalOpenPiExpert(config,20,5,context_capacity=592 if args.fixed_context else None)
            result['context_capacity']=592 if args.fixed_context else 'valid-only'
            trace=np.load(prefix/'trace.npz')
            def bits(a): return torch.from_numpy(a.copy()).to('cuda').view(torch.bfloat16)
            keys=torch.stack([bits(trace[f'21_encoder_layer{i:02d}_key']) for i in range(18)])
            values=torch.stack([bits(trace[f'21_encoder_layer{i:02d}_value']) for i in range(18)])
            expert.refresh_context(keys,values,keys.shape[1])
            noise=torch.as_tensor(trace['32_fixed_noise'],device='cuda')
            previous=noise[5:].clone()
            saved=np.load(reference/'reference.npz')
            def eager(): return sample_original_rtc(expert,noise,previous,5,5)
            raw,_=eager()
            result['eager_vs_independent_max_abs']=float(np.abs(raw.cpu().numpy()-saved['raw']).max())
            result['eager_rtc_expert']=timed(torch,eager,count=3)
            def forward():
                with torch.no_grad():
                    x=noise.clone()
                    for k in range(5): x=x-expert.velocity(x,k)/5
                    return x
            result['eager_forward_only']=timed(torch,forward,count=3)
            started=time.monotonic()
            graph=CapturedOriginalRtc(expert,noise)
            result['capture_seconds']=time.monotonic()-started
            captured,velocities,corrections=graph.replay(noise,previous,5,5)
            result['graph_vs_independent_max_abs']=float(np.abs(captured.detach().cpu().numpy()-saved['raw']).max())
            result['per_step']=[dict(step=k,
                velocity_max_abs=float(np.abs(velocities[k].detach().cpu().numpy()-saved[f'velocity{k}']).max()),
                vjp_max_abs=float(np.abs(corrections[k].detach().cpu().numpy()-saved[f'vjp{k}']).max())) for k in range(5)]
            initial_raw=captured.detach().clone()
            # Runtime inputs must not have been baked into capture.
            changed_noise=noise+.01
            changed_previous=noise[6:].clone()*.9
            ref,_=sample_original_rtc(expert,changed_noise,changed_previous,7,6)
            got=graph.replay(changed_noise,changed_previous,7,6)[0].detach().clone()
            result['changed_runtime_inputs_max_abs']=float((ref-got).abs().max())
            expert.refresh_context(keys,values*.9,keys.shape[1]-7 if args.fixed_context else keys.shape[1])
            context_reference=sample_original_rtc(expert,noise,previous,5,5)[0]
            context_got=graph.replay(noise,previous,5,5)[0].detach().clone()
            result['changed_context_max_abs']=float((context_reference-context_got).abs().max())
            result['changed_context_changes_output']=not torch.equal(initial_raw,context_got)
            expert.refresh_context(keys,values,keys.shape[1])
            result['graph_rtc_expert']=timed(torch,lambda:graph.replay(noise,previous,5,5))
            if args.changing_lengths:
                import gc
                result['length_transitions']=[]
                for valid in (keys.shape[1]-7,keys.shape[1]-1,keys.shape[1]):
                    started=time.monotonic()
                    del graph
                    gc.collect()
                    torch.cuda.empty_cache()
                    expert.refresh_context(keys,values,valid)
                    reference_raw,reference_norms=sample_original_rtc(expert,noise,previous,5,5)
                    graph=CapturedOriginalRtc(expert,noise)
                    got,_,gradient=graph.replay(noise,previous,5,5)
                    error=float((got-reference_raw).abs().max())
                    norm_error=float((gradient.norm(dim=(1,2))-torch.tensor(reference_norms,device='cuda')).abs().max())
                    result['length_transitions'].append(dict(valid=int(valid),raw_max_abs=error,
                        vjp_norm_max_abs=norm_error,prepare_seconds=time.monotonic()-started,
                        passed=bool(np.isfinite(error) and np.isfinite(norm_error) and max(error,norm_error)<1e-4)))
            result['passed']=(result['changed_context_changes_output']
                and all(max(r['velocity_max_abs'],r['vjp_max_abs'])<1e-4 for r in result['per_step'])
                and all(result[k]<1e-4 for k in ('eager_vs_independent_max_abs','graph_vs_independent_max_abs','changed_runtime_inputs_max_abs','changed_context_max_abs')))
            result['passed']=result['passed'] and all(r['passed'] for r in result.get('length_transitions',[]))
            result['status']='completed' if result['passed'] else 'failed_gate'
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json',result)
            print(json.dumps(result),flush=True)


if __name__=='__main__': main()
