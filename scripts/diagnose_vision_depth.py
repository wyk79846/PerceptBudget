#!/usr/bin/env python3
"""Fixed-input vision-depth ablation; shared weights/buffers, no rollout."""
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
    run=create_run(cfg,'M4','vision-depth-fixed-input',repo_root=ROOT)
    result=dict(status='running',source=str(a.source_run),observation_step=a.observation_step,model_loads=0,cases={},
        scope='Same observation,512 visual tokens,state,noise,prefix,delay; context CUDA graph latency only, no full RTC latency/energy/success claim')
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
            graphs={27:pl._context_graph}
            # Native producer graphs, shared allocation/weights. Last-layer fusion
            # is selected by the existing vision_num_layers loop at capture time.
            for depth in [26,24]:
                pl.vision_num_layers=depth
                graphs[depth]=capture_graph(pl,pl._graph_stream,lambda:_run_context(pl,pipe._graph_torch_stream.cuda_stream))
            pl.vision_num_layers=27
            release_unused_native_expert(model,torch)
            adapter=OriginalOpenPiRTCAdapter(model,torch,cfg);adapter.bootstrap(snap,task['instruction'],{})
            valid=int(pl.attn.dec_devpos.item())
            if valid!=512+pipe.current_prompt_len:raise ValueError('Prefix mismatch')
            z=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
            prev=torch.tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
            outputs={};timings={d:[] for d in graphs}
            # Interleave depths to reduce warmup/order bias. Images/text fixed.
            for order in [[27,26,24],[24,26,27]]*4:
                for depth in order:
                    begin=time.perf_counter();graphs[depth].replay(pl._graph_stream);pipe._graph_torch_stream.synchronize()
                    timings[depth].append((time.perf_counter()-begin)*1000)
            for depth in [27,26,24,27]:
                name=str(depth) if str(depth) not in outputs else '27_repeat'
                graphs[depth].replay(pl._graph_stream);pipe._graph_torch_stream.synchronize()
                adapter.expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                out,_=sample_original_rtc(adapter.expert,z.clone(),prev.clone(),solver['delay'],solver['executed'],beta=cfg['rtc_beta'],author_order=True)
                out=out.detach().cpu().numpy().copy();outputs[name]=out
                if not np.isfinite(out).all():raise ValueError('Nonfinite actions')
                diff=out[:,:7]-outputs['27'][:,:7];offset=solver['delay']
                result['cases'][name]=dict(vision_layers=depth,visual_tokens=512,state=True,valid_prefix=valid,
                    context_median_ms=median(timings[depth][1:]),context_samples_ms=timings[depth],
                    all_action7_rmse=float(np.sqrt(np.mean(diff**2))),
                    next10_executable_action7_rmse=float(np.sqrt(np.mean(diff[offset:offset+10]**2))),
                    action7_max_abs=float(np.max(np.abs(diff))))
                save();print(name+'='+json.dumps(result['cases'][name]),flush=True)
            np.savez(run/'actions.npz',**outputs)
            if result['cases']['27_repeat']['action7_max_abs']>1e-4:raise ValueError('Full repeat mismatch')
            result.update(status='completed',passed=True,memory=memory_record(torch,'finished'))
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if pl is not None:pl.vision_num_layers=27
        if adapter is not None:adapter.release_graph()
        save();print('VISION_DEPTH_STATUS='+json.dumps(result),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
