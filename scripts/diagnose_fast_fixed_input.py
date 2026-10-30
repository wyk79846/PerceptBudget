#!/usr/bin/env python3
"""Same recorded observation/noise/prefix/delay: Full vs pooled with/without state.

Offline action sensitivity only, not a rollout or inference latency benchmark.
"""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--observation-step',type=int,default=64)
    args=parser.parse_args()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    import numpy as np
    from perceptbudget.experiment import create_run,_write_json_atomically
    from scripts import eval_q as base
    from perceptbudget.h50_checkpoint import image_input,normalize_state
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.openpi_reference import sample_original_rtc
    from perceptbudget.shared_visual_context import wrist_frontend
    source=json.loads((args.source_run/'run.json').read_text())['config_snapshot']
    cfg=dict(source)
    cfg.update(rtc_audit_inputs=False,rtc_audit_attention_same_input=False,rtc_audit_fused_norm=False)
    requests=json.loads((args.source_run/'rtc/q5_requests.json').read_text())
    matches=[r for r in requests if r['observation_step']==args.observation_step]
    if len(matches)!=1:raise ValueError('Require exactly one recorded request at selected step')
    r=matches[0];solver=r['rtc_solver_inputs']
    if solver['previous_raw'] is None:raise ValueError('Require guided request')
    run=create_run(cfg,'M4','fast-fixed-input',repo_root=ROOT)
    result=dict(status='running',source_run=str(args.source_run),observation_step=args.observation_step,
                delay=solver['delay'],executed=solver['executed'],model_loads=0,cases={},
                scope='Identical observation/noise/previous_raw/delay/executed; action sensitivity, no closed-loop outcome or latency claim')
    def save():_write_json_atomically(run/'summary.json',result)
    print('run_dir='+str(run),flush=True);save();adapter=None
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch,_=base._check_cuda_and_native(cfg);torch.use_deterministic_algorithms(True)
            frame=np.load(args.source_run/'rtc/q5_frames.npz')['frames'][args.observation_step]
            raw=np.asarray(r['observation_raw_state'],np.float32)
            stats=load_libero_norm_stats(cfg['norm_stats_path'])
            snapshot=dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],raw_state=raw,state=normalize_state(raw,stats))
            task=base._load_tasks([source['screen']['task_ids'][0]],[source['screen']['suites'][0]])[0]
            model=base._load_and_check_model(cfg,torch);result['model_loads']=1
            from flash_rt.subgraphs.pi05.context_action import enable
            from flash_rt.api import VLAModel
            enable(model);base._infer_actions(model,snapshot,task['instruction'],cfg['H'],{})
            pooled=VLAModel(wrist_frontend(model._pipe,torch,fast_kind='pool2_agent',no_state=False),'torch')
            enable(pooled);base._infer_actions(pooled,snapshot,task['instruction'],cfg['H'],{})
            release_unused_native_expert(model,torch);release_unused_native_expert(pooled,torch)
            adapter=OriginalOpenPiRTCAdapter(model,torch,cfg)
            # Initialize the frozen expert attention/norm implementation via normal adapter.
            adapter.bootstrap(snapshot,task['instruction'],{})
            noise=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
            previous=torch.tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
            outputs={}
            for name,frontend,state in [('Full',model,True),('Fast',pooled,False),('Fast_state',pooled,True),('Full_repeat',model,True)]:
                adapter.model=frontend;adapter.pipe=frontend._pipe;adapter.pipeline=frontend._pipe.pipeline
                data=dict(snapshot,state=snapshot['state'] if state else None)
                valid=adapter._prepare_context(data,task['instruction'],{},guided=True)
                pl=frontend._pipe.pipeline
                adapter.expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                actions,norms=sample_original_rtc(adapter.expert,noise.clone(),previous.clone(),solver['delay'],solver['executed'],beta=cfg['rtc_beta'],author_order=True)
                out=actions.detach().cpu().numpy().copy();outputs[name]=out
                if not np.isfinite(out).all():raise ValueError('Nonfinite actions')
                diff=out[:,:7]-outputs['Full'][:,:7];start=solver['delay']
                result['cases'][name]=dict(state=state,visual_tokens=pl.vision_seq_enc,valid_prefix=valid,
                    all_action7_rmse=float(np.sqrt(np.mean(diff**2))),
                    next10_executable_action7_rmse=float(np.sqrt(np.mean(diff[start:start+10]**2))),
                    next10_executable_arm6_rmse=float(np.sqrt(np.mean(diff[start:start+10,:6]**2))),
                    action7_max_abs=float(np.max(np.abs(diff))))
                result['cases'][name]['vs_recorded_raw_max_abs']=float(np.max(np.abs(out-np.asarray(r['actions_raw']))))
                save();print(name+'='+json.dumps(result['cases'][name]),flush=True)
            np.savez(run/'actions.npz',**outputs)
            if result['cases']['Full_repeat']['action7_max_abs']>1e-4:raise ValueError('Full repeat changed after pooled cases')
            result.update(status='completed',passed=True)
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if adapter is not None:adapter.release_graph()
        save()
        print('FIXED_INPUT_STATUS='+json.dumps(result),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
