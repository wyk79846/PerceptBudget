"""Replay saved observations with identical RTC inputs; no environment rollout."""
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
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--candidate-run',type=Path,required=True)
    parser.add_argument('--request-id',type=int,default=4)
    args=parser.parse_args()
    import numpy as np
    config=load_config(repo_root=ROOT)
    config.update(json.loads((args.candidate_run/'run.json').read_text())['config_snapshot'])
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run=create_run(config,'M1','rtc-observation-replay',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        for name in ('scripts/audit_rtc_observation_replay.py','perceptbudget/rtc_condition_cache.py',
                     'perceptbudget/rtc_source_step_graph.py','perceptbudget/openpi_reference.py'):
            dest=run/'source'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest)
        result=dict(status='running',scope='saved observations and solver inputs only; no simulation')
        try:
            torch,_=base._check_cuda_and_native(config)
            torch.use_deterministic_algorithms(True)
            from flash_rt.subgraphs.pi05.context_action import enable
            from perceptbudget.libero_input import _rotate_resize
            from perceptbudget.rtc_memory import release_unused_native_expert
            from perceptbudget.openpi_reference import OriginalOpenPiExpert,sample_original_rtc
            from perceptbudget.rtc_source_step_graph import SourceStepGraph
            from perceptbudget.rtc_condition_cache import enable_condition_cache
            records=[]
            for path in (args.reference_run,args.candidate_run):
                requests=json.loads((path/'rtc/q5_requests.json').read_text())
                request=next(r for r in requests if r['prediction_id']==args.request_id)
                frames=np.load(path/'rtc/q5_frames.npz')['frames']
                frame=frames[request['observation_step']]
                width=frame.shape[1]//2
                snapshot=dict(images=[_rotate_resize(frame[:,:width]),_rotate_resize(frame[:,width:])],
                              state=np.asarray(request['model_state'],dtype=np.float32))
                records.append((request,snapshot))
            a,b=records
            if a[0]['rtc_solver_inputs']!=b[0]['rtc_solver_inputs']:
                raise ValueError('comparison must use identical saved RTC inputs')
            if not np.array_equal(a[1]['state'],b[1]['state']):
                raise ValueError('comparison must use identical saved state')
            result['resized_image_difference']=[dict(channels=int(np.count_nonzero(x!=y)),
                max_abs=int(np.abs(x.astype(int)-y.astype(int)).max()))
                for x,y in zip(a[1]['images'],b[1]['images'])]
            task=base._load_tasks([config['screen']['task_ids'][0]],['libero_spatial'])[0]
            prompt=task['instruction']
            result['prompt']=prompt
            model=base._load_and_check_model(config,torch)
            enable(model)
            base._infer_actions(model,a[1],prompt,config['H'],{})
            result['native_release']=release_unused_native_expert(model,torch)
            pipe=model._pipe
            pl=pipe.pipeline
            def context(snapshot):
                with torch.cuda.stream(pipe._graph_torch_stream):
                    model.set_prompt(prompt,state=snapshot['state'])
                    pipe._fill_img_buf({'images':snapshot['images']})
                    pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf,pl.input_images_buf,
                                                             pipe._graph_torch_stream.cuda_stream)
                    pl._context_graph.replay(pl._graph_stream)
                pipe._graph_torch_stream.synchronize()
                valid=int(pl.attn.dec_devpos.item()) if pl._fixed_shape else int(pl.encoder_seq_len)
                # Only the valid token range is an input to the expert.
                # Unwritten capacity can contain arbitrary bits/NaNs.
                keys=pl.attn.enc_K[:,:valid].clone()
                values=pl.attn.enc_V[:,:valid].clone()
                torch.cuda.synchronize()  # finish default-stream copies before next context replay
                return keys,values,valid
            contexts=[context(snapshot) for _,snapshot in records]
            again=context(a[1])
            result['context_repeat_bitwise_equal']=all(torch.equal(x,y) for x,y in zip(contexts[0][:2],again[:2]))
            result['context_changed_elements']=[int((x!=y).sum()) for x,y in zip(contexts[0][:2],contexts[1][:2])]
            result['context_scope']='valid tokens only; cloned buffers synchronized before next context replay'
            expert=OriginalOpenPiExpert(config,config['H'],config['N'])
            inputs=a[0]['rtc_solver_inputs']
            noise=torch.tensor(inputs['noise'],device='cuda',dtype=torch.float32)
            previous=torch.tensor(inputs['previous_raw'],device='cuda',dtype=torch.float32)
            d,s=inputs['delay'],inputs['executed']
            def compare(x,y):
                return dict(bitwise_equal=bool(torch.equal(x,y)),max_abs=float((x-y).abs().max()))
            eager=[]
            original=[]
            for context_data,(request,_) in zip(contexts,records):
                expert.refresh_context(*context_data)
                trace=[]
                raw,_=sample_original_rtc(expert,noise,previous,d,s,trace=trace,author_order=True)
                eager.append((raw.clone(),trace))
                original.append(torch.tensor(request['actions_raw'],device='cuda',dtype=torch.float32))
            result['eager_image_effect']=compare(eager[0][0],eager[1][0])
            expert.refresh_context(*again)
            repeated,_=sample_original_rtc(expert,noise,previous,d,s,author_order=True)
            result['eager_repeated_observation']=compare(eager[0][0],repeated)
            expert.refresh_context(*contexts[0])
            plain=SourceStepGraph(expert,noise)
            plain_raw=[]
            for context_data in contexts:
                expert.refresh_context(*context_data)
                raw,_=plain.replay(noise,previous,d,s)
                plain_raw.append(raw.clone())
            enable_condition_cache(expert,graph_live=True)
            packed=SourceStepGraph(expert,noise)
            result['observations']=[]
            for index,(context_data,(reference,trace)) in enumerate(zip(contexts,eager)):
                expert.refresh_context(*context_data)
                got=[]
                raw,_=packed.replay(noise,previous,d,s,trace=got)
                row=dict(index=index,valid=context_data[2],eager_vs_recorded=compare(reference,original[index]),
                    plain_vs_eager=compare(plain_raw[index],reference),packed_vs_eager=compare(raw,reference),
                    per_step=[{k:compare(u[k],v[k]) for k in ('input','velocity','clean','error','vjp','output')}
                              for u,v in zip(trace,got)])
                result['observations'].append(row)
            result.update(status='completed',original_sources=expert.provenance,
                limitation='Only the first divergent saved observation pair, not all tasks or proof of renderer internals.')
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json',result)
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
