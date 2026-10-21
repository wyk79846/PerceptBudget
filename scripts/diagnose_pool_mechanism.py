#!/usr/bin/env python3
"""Fixed-input pooling ablations; no rollout, training, speed or safety claim.

One Full-sized allocation is reused to isolate feature averaging and sequence
shortening. Manual context execution is gated against the original captured
context. Only this diagnostic's pipeline metadata/buffer is temporarily changed.
"""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def transform_tokens(vision, torch, factor, view, mode):
    """Original SigLIP outputs, camera-major 16x16 grid, unchanged BF16 rounding."""
    if mode=='full':
        return vision.reshape(512,1152).clone()
    output=[]
    for camera in range(2):
        x=vision[camera]
        if camera != (0 if view=='agent' else 1):
            output.append(x.reshape(256,1152))
            continue
        if mode=='select_compact':
            output.append(x[::factor,::factor].reshape(-1,1152))
            continue
        pooled=torch.zeros(16//factor,16//factor,1152)
        for dr in range(factor):
            for dc in range(factor):
                pooled+=x.float()[dr::factor,dc::factor]
        pooled=(pooled/factor**2).to(torch.bfloat16)
        if mode=='mean_full_length':
            pooled=pooled.repeat_interleave(factor,0).repeat_interleave(factor,1)
        elif mode!='mean_compact':
            raise ValueError('Unknown token transform')
        output.append(pooled.reshape(-1,1152))
    return torch.cat(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--pool-view',choices=('agent','wrist'),default='agent')
    parser.add_argument('--factor',type=int,choices=(2,4),default=2)
    args=parser.parse_args()
    import numpy as np
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    from perceptbudget.baseline import validate
    from scripts import eval_q as base
    from perceptbudget.h50_checkpoint import image_input,normalize_state
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import release_unused_native_expert,memory_record
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.openpi_reference import sample_original_rtc
    config=load_config(repo_root=ROOT);validate(config)
    config.update(rtc_audit_inputs=False,rtc_audit_attention_same_input=False,rtc_audit_fused_norm=False)
    run=create_run(config,'M2','pool-mechanism',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    result=dict(status='running',cases=[],model_loads=0,source_run=str(args.source_run),
        factor=args.factor,pooled_view=args.pool_view,
        scope='fixed-input action sensitivity; no rollout or latency estimate; Full-sized allocation reused',
        position_control='mean_full_length keeps 512 tokens and positions; compact modes change both count and positions')
    def save():_write_json_atomically(run/'summary.json',result)
    save()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    adapter=None
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch,_=base._check_cuda_and_native(config)
            torch.use_deterministic_algorithms(True)
            from flash_rt.core.cuda_buffer import CudaBuffer
            requests=json.loads((args.source_run/'rtc/q5_requests.json').read_text())
            source=json.loads((args.source_run/'run.json').read_text())['config_snapshot']
            for key in ('H','N','checkpoint_provenance_revision','state_enabled','rtc_beta'):
                if source[key]!=config[key]:raise ValueError('Source mismatch: '+key)
            indices=sorted(set([1,len(requests)//2,len(requests)-1]))
            if len(indices)!=3:raise ValueError('Require three recorded guided requests')
            with np.load(args.source_run/'rtc/q5_frames.npz') as archive:frames=archive['frames']
            task=base._load_tasks(source['screen']['task_ids'],source['screen']['suites'])[0]
            stats=load_libero_norm_stats(config['norm_stats_path'])
            def snapshot(i):
                r=requests[i];frame=frames[r['observation_step']]
                raw=np.asarray(r['observation_raw_state'],dtype=np.float32)
                return dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],
                            raw_state=raw,state=normalize_state(raw,stats))
            model=base._load_and_check_model(config,torch);result['model_loads']=1
            from flash_rt.subgraphs.pi05.context_action import enable
            enable(model)
            base._infer_actions(model,snapshot(indices[0]),task['instruction'],50,{})
            release_unused_native_expert(model,torch)
            adapter=OriginalOpenPiRTCAdapter(model,torch,config)
            pipe=model._pipe;pl=pipe.pipeline
            original_buf=pl.bufs['vision_x_pooled']
            packed= CudaBuffer.device_empty(512*1152,np.uint16)
            def reset_geometry():
                pl.vision_seq_enc=512;pl.encoder_seq_len=712;pl.total_kv=762
                pl.bufs['vision_x_pooled']=original_buf
            for i in indices:
                reset_geometry()
                data=snapshot(i)
                adapter.bootstrap(data,task['instruction'],{})
                valid=512+pipe.current_prompt_len
                captured_k=pl.attn.enc_K[:,:valid].clone()
                captured_v=pl.attn.enc_V[:,:valid].clone()
                bits=np.empty(512*1152,dtype=np.uint16)
                pl.bufs['vision_x'].download(bits)
                vision=torch.from_numpy(bits).view(torch.bfloat16).reshape(2,16,16,1152)
                solver=requests[i]['rtc_solver_inputs']
                noise=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
                previous=torch.tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
                outputs={};row=dict(request_index=i,observation_step=requests[i]['observation_step'],conditions={})
                for mode in ('full','mean_full_length','mean_compact','select_compact'):
                    tokens=transform_tokens(vision,torch,args.factor,args.pool_view,mode)
                    count=len(tokens);valid=count+pipe.current_prompt_len
                    pl.bufs['vision_x_pooled']=packed
                    pl.vision_seq_enc=count;pl.encoder_seq_len=count+200;pl.total_kv=pl.encoder_seq_len+50
                    packed.upload(tokens.contiguous().view(torch.uint16).numpy())
                    pl.attn.set_fixed_valid_len(valid)
                    with torch.cuda.stream(pipe._graph_torch_stream):
                        pl._copy_lang_embeds_to_encoder_x(stream=pipe._graph_torch_stream.cuda_stream)
                        pl.transformer_encoder(stream=pipe._graph_torch_stream.cuda_stream)
                    pipe._graph_torch_stream.synchronize()
                    if mode=='full':
                        row['manual_vs_captured_kv_max_abs']=max(float((captured_k-pl.attn.enc_K[:,:valid]).abs().max()),
                            float((captured_v-pl.attn.enc_V[:,:valid]).abs().max()))
                        if row['manual_vs_captured_kv_max_abs']!=0:
                            raise RuntimeError('Manual Full context does not equal captured context')
                    adapter.expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                    raw,_=sample_original_rtc(adapter.expert,noise,previous,solver['delay'],solver['executed'],beta=5.,author_order=True)
                    raw=raw.detach().cpu();outputs[mode]=raw
                    diff=raw[:,:7]-outputs['full'][:,:7]
                    item=dict(visual_tokens=count,valid_prefix=valid,finite=bool(torch.isfinite(raw).all()),
                        normalized_action7_rmse=float(diff.square().mean().sqrt()),
                        first_q_action7_rmse=float(diff[:5].square().mean().sqrt()),
                        normalized_action7_max_abs=float(diff.abs().max()),
                        normalized_action7_rmse_per_dim=diff.square().mean(0).sqrt().tolist())
                    if not item['finite']:raise RuntimeError('Nonfinite diagnostic action')
                    row['conditions'][mode]=item
                row['mean_vs_select_action7_rmse']=float((outputs['mean_compact'][:,:7]-outputs['select_compact'][:,:7]).square().mean().sqrt())
                result['cases'].append(row);save()
                print('mechanism_case='+json.dumps(row),flush=True)
                reset_geometry()
                memory=memory_record(torch,'after_case');result.setdefault('memory',[]).append(memory)
                if memory['available_mib']<900:raise RuntimeError('RAM headroom below900MiB')
            reset_geometry()
            result['status']='completed'
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if adapter is not None:adapter.release_graph()
        save()
        metadata=json.loads((run/'run.json').read_text());metadata['status']=result['status']
        _write_json_atomically(run/'run.json',metadata)
        print('MECHANISM_TERMINAL_STATUS='+json.dumps(dict(run=str(run),status=result['status'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
