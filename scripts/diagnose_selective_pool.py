#!/usr/bin/env python3
"""Fixed-input selective pooling:512/416/320 tokens; no rollout."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import median
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--observation-step',type=int,default=140)
    a=p.parse_args()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    import numpy as np
    from scripts import eval_q as base
    from perceptbudget.experiment import create_run,_write_json_atomically
    from perceptbudget.h50_checkpoint import image_input,normalize_state
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import release_unused_native_expert,memory_record
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.openpi_reference import sample_original_rtc
    cfg=json.loads((a.source_run/'run.json').read_text())['config_snapshot']
    cfg=dict(cfg,rtc_audit_inputs=False,rtc_audit_attention_same_input=False,rtc_audit_fused_norm=False)
    rows=json.loads((a.source_run/'rtc/q5_requests.json').read_text())
    request=next(r for r in rows if r['observation_step']==a.observation_step)
    solver=request['rtc_solver_inputs']
    run=create_run(cfg,'M4','selective-pool-fixed-input',repo_root=ROOT)
    result=dict(status='running',source=str(a.source_run),observation_step=a.observation_step,model_loads=0,cases={},
        scope='Same observation,state,noise,prefix,delay; split perception timing includes vision, GPU selection/copies,encoder and host dispatch; not full RTC latency/energy/success')
    def save():_write_json_atomically(run/'summary.json',result)
    print('run_dir='+str(run),flush=True);save();adapter=None;pl=None
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch,_=base._check_cuda_and_native(cfg);torch.use_deterministic_algorithms(True)
            frame=np.load(a.source_run/'rtc/q5_frames.npz')['frames'][a.observation_step]
            raw=np.asarray(request['observation_raw_state'],np.float32)
            snap=dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],raw_state=raw,
                state=normalize_state(raw,load_libero_norm_stats(cfg['norm_stats_path'])))
            if not np.array_equal(snap['state'],np.asarray(request['model_state'],np.float32)):raise ValueError('State mismatch')
            if [hashlib.sha256(x.tobytes()).hexdigest() for x in snap['images']]!=request['context_refresh']['image_input_sha256']:raise ValueError('Image mismatch')
            task=base._load_tasks([cfg['screen']['task_ids'][0]],[cfg['screen']['suites'][0]])[0]
            model=base._load_and_check_model(cfg,torch);result['model_loads']=1
            from flash_rt.subgraphs.pi05.context_action import enable,_run_context
            from flash_rt.subgraphs.capture import capture_graph
            enable(model);base._infer_actions(model,snap,task['instruction'],cfg['H'],{})
            pipe=model._pipe;pl=pipe.pipeline
            if pl.vision_seq_enc!=512 or pl.vision_pool_factor!=1 or pl.vision_num_layers!=27:raise ValueError('Unexpected Full geometry')
            import ctypes
            from flash_rt.core.cuda_buffer import CudaBuffer,_cudart
            from perceptbudget.selective_pool import compact
            original_buffer=pl.bufs['vision_x_pooled']
            original_context=pl._context_graph
            packed=CudaBuffer.device_empty(512*1152,np.uint16)
            raw_features=torch.empty((2,16,16,1152),device='cuda',dtype=torch.bfloat16)
            def copy(dst,src,nbytes):
                ret=_cudart.cudaMemcpyAsync(ctypes.c_void_p(dst),ctypes.c_void_p(src),nbytes,3,ctypes.c_void_p(pipe._graph_torch_stream.cuda_stream))
                if ret:raise RuntimeError('D2D failed:'+str(ret))
            def geometry(count):
                pl.vision_seq_enc=count;pl.encoder_seq_len=count+pl.max_prompt_len
                pl.total_kv=pl.encoder_seq_len+cfg['H']
                pl.attn.set_fixed_valid_len(count+pipe.current_prompt_len)
            vision_graph=capture_graph(pl,pl._graph_stream,lambda:pl.vision_encoder(pipe._graph_torch_stream.cuda_stream))
            encoders={}
            pl.bufs['vision_x_pooled']=packed
            def encode():
                pl._copy_lang_embeds_to_encoder_x(stream=pipe._graph_torch_stream.cuda_stream)
                pl.transformer_encoder(stream=pipe._graph_torch_stream.cuda_stream)
            for count in [512,416,320]:
                geometry(count)
                if count!=512:
                    pl._gemms_autotuned=False
                    pl.autotune_gemms()
                encoders[count]=capture_graph(pl,pl._graph_stream,encode)
            geometry(512);pl.bufs['vision_x_pooled']=original_buffer
            release_unused_native_expert(model,torch)
            adapter=OriginalOpenPiRTCAdapter(model,torch,cfg);adapter.bootstrap(snap,task['instruction'],{})
            baseline_valid=int(pl.attn.dec_devpos.item())
            baseline_k=pl.attn.enc_K[:,:baseline_valid].clone()
            baseline_v=pl.attn.enc_V[:,:baseline_valid].clone()
            z=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
            prev=torch.tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
            outputs={};timings={c:[] for c in encoders};masks={}
            # Capture the selector only, keeping real input dynamic. This removes
            # Python dispatch of its many small kernels, not its GPU computation.
            selector_graphs={};selector_outputs={}
            with torch.cuda.stream(pipe._graph_torch_stream):
                copy(raw_features.data_ptr(),pl.bufs['vision_x'].ptr.value,512*1152*2)
                for count in [416,320]:
                    for _ in range(3):compact(raw_features,(512-count)//3)
                    pipe._graph_torch_stream.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=pipe._graph_torch_stream):
                        tokens,mask=compact(raw_features,(512-count)//3)
                    selector_graphs[count]=graph;selector_outputs[count]=(tokens,mask)
            result['selector_backend']='torch_cuda_graph_fixed_output_size'
            result['encoder_shapes_autotuned']=[512,416,320]
            def context(count):
                geometry(count);pl.bufs['vision_x_pooled']=packed
                vision_graph.replay(pl._graph_stream)
                with torch.cuda.stream(pipe._graph_torch_stream):
                    if count==512:
                        copy(packed.ptr.value,pl.bufs['vision_x'].ptr.value,512*1152*2)
                    else:
                        copy(raw_features.data_ptr(),pl.bufs['vision_x'].ptr.value,512*1152*2)
                        selector_graphs[count].replay()
                        tokens,mask=selector_outputs[count]
                        copy(packed.ptr.value,tokens.data_ptr(),count*1152*2)
                        masks[count]=mask
                    encoders[count].replay(pl._graph_stream)
                pipe._graph_torch_stream.synchronize()
                return count+pipe.current_prompt_len
            for count in encoders:context(count)
            for order in [[512,416,320],[320,416,512]]*4:
                for count in order:
                    begin=time.perf_counter();context(count)
                    timings[count].append((time.perf_counter()-begin)*1000)
            for count in [512,416,320,512]:
                name=str(count) if str(count) not in outputs else '512_repeat'
                valid=context(count)
                if count==512:
                    difference=max(float((baseline_k-pl.attn.enc_K[:,:valid]).abs().max()),float((baseline_v-pl.attn.enc_V[:,:valid]).abs().max()))
                    result['full_split_vs_original_kv_max_abs']=difference
                    if difference!=0:raise ValueError('Split Full differs from original captured context')
                adapter.expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                out,_=sample_original_rtc(adapter.expert,z.clone(),prev.clone(),solver['delay'],solver['executed'],beta=cfg['rtc_beta'],author_order=True)
                out=out.detach().cpu().numpy().copy();outputs[name]=out
                if not np.isfinite(out).all():raise ValueError('Nonfinite actions')
                diff=out[:,:7]-outputs['512'][:,:7];offset=solver['delay']
                result['cases'][name]=dict(visual_tokens=count,state=True,valid_prefix=valid,
                    context_median_ms=median(timings[count][1:]),context_samples_ms=timings[count],
                    all_action7_rmse=float(np.sqrt(np.mean(diff**2))),
                    next10_executable_action7_rmse=float(np.sqrt(np.mean(diff[offset:offset+10]**2))),
                    action7_max_abs=float(np.max(np.abs(diff))))
                if count in masks:
                    selected=masks[count].detach().cpu().numpy()
                    result['cases'][name]['merged_external_blocks']=np.flatnonzero(selected).tolist()
                    # Independent CPU reference for camera/row/column layout.
                    bits=np.empty(512*1152,dtype=np.uint16);pl.bufs['vision_x'].download(bits)
                    features=torch.from_numpy(bits).view(torch.bfloat16).reshape(2,16,16,1152)
                    expected=[]
                    for row in range(16):
                        for col in range(16):
                            merged=selected[(row//2)*8+col//2]
                            if merged and (row%2 or col%2):continue
                            block=features[0,row:row+2,col:col+2].float()
                            token=((block[0,0]+block[0,1]+block[1,0]+block[1,1])/4).to(torch.bfloat16) if merged else features[0,row,col]
                            expected.append(token)
                    expected=torch.cat((torch.stack(expected),features[1].reshape(256,1152)))
                    actual=np.empty(count*1152,np.uint16);packed.download(actual)
                    actual=torch.from_numpy(actual).view(torch.bfloat16).reshape(count,1152)
                    layout_difference=float((actual.float()-expected.float()).abs().max())
                    result['cases'][name]['cpu_layout_max_abs']=layout_difference
                    if layout_difference!=0:raise ValueError('Compact layout differs from CPU reference')
                save();print(name+'='+json.dumps(result['cases'][name]),flush=True)
            np.savez(run/'actions.npz',**outputs)
            if result['cases']['512_repeat']['action7_max_abs']>1e-4:raise ValueError('Full repeat mismatch')
            result.update(status='completed',passed=True,memory=memory_record(torch,'finished'))
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if pl is not None:
            pl.vision_seq_enc=512;pl.encoder_seq_len=512+pl.max_prompt_len;pl.total_kv=pl.encoder_seq_len+cfg['H']
            if 'original_buffer' in locals():pl.bufs['vision_x_pooled']=original_buffer
        save()
        if adapter is not None:adapter.release_graph()
        print('SELECTIVE_POOL_STATUS='+json.dumps(result),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
