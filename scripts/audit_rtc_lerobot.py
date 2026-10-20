"""Same frozen expert/KV/input comparison of author-order and upstream LeRobot RTC."""
import fcntl
import hashlib
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
    config=load_config(repo_root=ROOT)
    config.update(H=25,N=5,state_enabled=True,state_prompt_capacity=80,python_executable=sys.executable)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run=create_run(config,'M1','rtc-lerobot-fixed-kv',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        shutil.copy2(__file__,run/'audit_rtc_lerobot.py')
        result=dict(status='running',scope='same frozen original expert and recorded KV; no encoder or environment')
        try:
            import numpy as np
            import torch
            import inspect
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.configs.types import RTCAttentionSchedule
            from perceptbudget.openpi_reference import OriginalOpenPiExpert,sample_original_rtc
            from perceptbudget.rtc_author_source import sample_author_source
            torch.use_deterministic_algorithms(True)
            source=Path(inspect.getfile(RTCProcessor))
            shutil.copy2(source,run/'upstream_modeling_rtc.py')
            result['lerobot_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
            root=Path(config['output_root'])/'M1'
            fixture=root/'20261019T054917.649092Z-capacity-numeric-diagnosis-5f903047/capacity80a/trace.npz'
            data=np.load(fixture)
            def bits(x):return torch.from_numpy(x.copy()).to('cuda').view(torch.bfloat16)
            keys=torch.stack([bits(data[f'21_encoder_layer{i:02d}_key']) for i in range(18)])
            values=torch.stack([bits(data[f'21_encoder_layer{i:02d}_value']) for i in range(18)])
            request_run=root/'20261019T125112.959616Z-openpi-rtc-actual-5384eecc'
            inputs=json.loads((request_run/'rtc/q5_requests.json').read_text())[1]['rtc_solver_inputs']
            noise=torch.tensor(inputs['noise'],device='cuda',dtype=torch.float32)
            previous=torch.tensor(inputs['previous_raw'],device='cuda',dtype=torch.float32)
            expert=OriginalOpenPiExpert(config,25,5)
            result.update(fixture=str(fixture),request_inputs=str(request_run),original_sources=expert.provenance)
            def compare(a,b):
                return dict(max_abs=float((a-b).abs().max()),bitwise_equal=bool(torch.equal(a,b)))
            result['cases']=[]
            tensors={}
            for valid in (570,563,557):
                expert.refresh_context(keys,values,valid)
                trace=[]
                reference,_=sample_original_rtc(expert,noise,previous,5,5,trace=trace,author_order=True)
                author,author_trace,_,author_hash=sample_author_source(expert,noise,previous,5,5,
                    source=Path(config['output_root'])/'rtc-author-reference/src/model.py')
                rtc=RTCProcessor(RTCConfig(enabled=True,execution_horizon=20,max_guidance_weight=5.,
                    prefix_attention_schedule=RTCAttentionSchedule.EXP,debug=True))
                rows=[]
                for step,record in enumerate(trace):
                    seen={}
                    def velocity(x):
                        v=expert.velocity(x[0],step).unsqueeze(0)
                        seen.update(input_requires_grad=x.requires_grad,velocity_requires_grad=v.requires_grad,
                                    velocity=v.detach().clone())
                        return v
                    corrected=rtc.denoise_step(record['input'],previous,5,1-step/5,velocity,execution_horizon=20)
                    # Tracker stores steps by time; retrieve the exact current time, not insertion guesses.
                    debug=next(s for s in rtc.get_all_debug_steps() if float(s.time)==1-step/5)
                    correction=debug.correction.squeeze(0)
                    error=debug.err.squeeze(0)
                    row=dict(step=step,input_requires_grad=seen['input_requires_grad'],
                        velocity_requires_grad=seen['velocity_requires_grad'],
                        velocity=compare(seen['velocity'][0],record['velocity']),
                        mask=compare(debug.weights.squeeze(0),record['weights']),
                        clean=compare(debug.x1_t.squeeze(0),record['clean']),
                        error=compare(error,record['error']),
                        correction=compare(correction,record['vjp']),
                        correction_vs_own_error=compare(correction,error),
                        updated=compare(record['input']-corrected/5,record['output']))
                    rows.append(row)
                    for name,tensor in (('author_vjp',record['vjp']),('lerobot_vjp',correction)):
                        tensors[f'{valid}_{step}_{name}']=tensor.detach().cpu().numpy()
                x=noise.clone()
                for step in range(5):
                    v=rtc.denoise_step(x,previous,5,1-step/5,
                        lambda a:expert.velocity(a[0],step).unsqueeze(0),execution_horizon=20)
                    x=(x-v/5).detach()
                row=dict(valid=valid,delay=5,executed=5,per_step=rows,raw=compare(x,reference),
                    author_body_raw=compare(author,reference),author_source_sha256=author_hash,
                    author_body_per_step=[{name:compare(a[name],b[name])
                        for name in ('input','velocity','clean','error','vjp','output')}
                        for a,b in zip(author_trace,trace)])
                result['cases'].append(row)
                print('case='+json.dumps(row),flush=True)
            np.savez_compressed(run/'corrections.npz',**tensors)
            result['status']='completed'
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json',result)
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
