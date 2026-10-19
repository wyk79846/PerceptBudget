"""Frozen-original-expert vs unchanged author function body; no rollout."""
import argparse
import fcntl
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
    parser.add_argument('--cpu-toy',action='store_true')
    parser.add_argument('--author-order',action='store_true',help='candidate preserving source scalar/operation order')
    args=parser.parse_args()
    config=load_config(repo_root=ROOT)
    config.update(H=25,N=5,state_enabled=True)
    source=Path(config['output_root'])/'rtc-author-reference/src/model.py'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run=create_run(config,'M1','rtc-author-source-gate',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        for name in ('scripts/audit_rtc_author_source.py','perceptbudget/rtc_author_source.py',
                     'perceptbudget/openpi_reference.py'):
            target=run/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        shutil.copy2(source,run/'author_model.py')
        result=dict(status='running',author_order=args.author_order,scope='unchanged author RTC body on Torch bridge; not JAX/XLA, not full encoder validation or rollout')
        try:
            import numpy as np
            import torch
            from perceptbudget.openpi_reference import OriginalOpenPiExpert,sample_original_rtc
            from perceptbudget.rtc_author_source import sample_author_source
            torch.manual_seed(7)
            if args.cpu_toy:
                class Toy:
                    horizon,num_steps=25,5
                    def velocity(self,x,k):
                        return .07*torch.tanh(x)+.03*x.roll(1,0)+.01*k
                expert=Toy()
                device='cpu'
            else:
                torch.use_deterministic_algorithms(True)
                device='cuda'
                fixture=Path(config['output_root'])/'M1/20261019T054917.649092Z-capacity-numeric-diagnosis-5f903047/capacity80a/trace.npz'
                data=np.load(fixture)
                def bits(x): return torch.from_numpy(x.copy()).to(device).view(torch.bfloat16)
                keys=torch.stack([bits(data[f'21_encoder_layer{i:02d}_key']) for i in range(18)])
                values=torch.stack([bits(data[f'21_encoder_layer{i:02d}_value']) for i in range(18)])
                expert=OriginalOpenPiExpert(config,25,5)
                expert.refresh_context(keys,values,keys.shape[1])
                result['original_source_hashes']=expert.provenance
                result['weights_frozen']=all(not p.requires_grad for p in expert.suffix.parameters())
            noise=torch.randn(25,32,device=device)
            old=torch.randn(25,32,device=device)*.3
            result['cases']=[]
            for case,(delay,executed) in enumerate(((12,5),(10,10),(9,8))):
                previous=old[executed:].clone()
                ours=[]
                got,_=sample_original_rtc(expert,noise,previous,delay,executed,trace=ours,author_order=args.author_order)
                ref,author,weights,sha=sample_author_source(expert,noise,previous,delay,executed,source=source)
                rows=[]
                arrays=dict(noise=noise.cpu().numpy(),previous=previous.cpu().numpy())
                for k,(a,b) in enumerate(zip(ours,author)):
                    row=dict(step=k)
                    for field in ('input','velocity','clean','error','vjp','output'):
                        row[field+'_max_abs']=float((a[field]-b[field]).abs().max())
                        arrays[f'project_{field}{k}']=a[field].detach().cpu().numpy()
                        arrays[f'author_{field}{k}']=b[field].detach().cpu().numpy()
                    rows.append(row)
                error=float((got-ref).abs().max())
                mask_error=float((ours[0]['weights'][:,0]-weights).abs().max())
                record=dict(delay=delay,executed=executed,raw_max_abs=error,mask_max_abs=mask_error,
                    per_step=rows,passed=bool(np.isfinite(error) and error<1e-4 and mask_error<1e-6
                        and all(np.isfinite(v) and v<1e-4 for row in rows for name,v in row.items() if name!='step')))
                result['cases'].append(record)
                result['author_source_sha256']=sha
                np.savez_compressed(run/f'case{case}.npz',**arrays)
                print('case='+json.dumps(record),flush=True)
            result.update(status='completed',passed=all(r['passed'] for r in result['cases']),device=device)
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json',result)
            print(json.dumps(result),flush=True)


if __name__=='__main__': main()
