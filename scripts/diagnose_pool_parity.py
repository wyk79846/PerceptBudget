#!/usr/bin/env python3
"""Fixed recorded inputs: pre-memory-fix graphs vs resident shared-KV graphs.

Loads the archived implementation into a private module, never checks out or
patches production/FlashRT. No simulation or closed-loop quality claims.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def pool_check(pipeline, torch, np):
    f=pipeline.vision_pool_factor
    def download(name, count):
        bits=np.empty(count,dtype=np.uint16)
        pipeline.bufs[name].download(bits)
        return torch.from_numpy(bits).view(torch.bfloat16).float()
    original=download('vision_x',2*256*1152).reshape(2,16,16,1152)
    actual=download('vision_x_pooled',2*256//f**2*1152).reshape(2,16//f,16//f,1152)
    # Same FP32 addition order as the native kernel, independent CPU indexing.
    expected=torch.zeros_like(actual)
    for dr in range(f):
        for dc in range(f):
            expected+=original[:,dr::f,dc::f,:]
    expected=(expected/(f*f)).to(torch.bfloat16).float()
    return dict(max_abs=float((expected-actual).abs().max()),
                mismatched_elements=int((expected!=actual).sum()),num_views=2,
                per_view_tokens=256//f**2,scope='native spatial pooling vs CPU reference, view/row/column order')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    from perceptbudget.experiment import create_run,load_config,_write_json_atomically
    from perceptbudget.baseline import validate
    from scripts import eval_q as base
    from perceptbudget.h50_checkpoint import image_input,normalize_state
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.openpi_reference import sample_original_rtc
    config=load_config(repo_root=ROOT)
    validate(config)
    source_cfg=json.loads((args.source_run/'run.json').read_text())['config_snapshot']
    for key in ('H','N','checkpoint_provenance_revision','state_enabled','rtc_beta','native_gemm_policy'):
        if config[key]!=source_cfg[key]:
            raise ValueError('Source baseline mismatch: '+key)
    factor=source_cfg.get('m2_vision_pool_factor',1)
    if factor not in (2,4):
        raise ValueError('Require a recorded dual-view pooling run')
    config.update(m2_profile=True,m2_view='full',m2_vision_pool_factor=factor,
                  rtc_audit_inputs=False,rtc_audit_attention_same_input=False,rtc_audit_fused_norm=False)
    run=create_run(config,'M2','pool-memory-parity',repo_root=ROOT)
    result=dict(status='running',source_run=str(args.source_run),pool_factor=factor,
                archived_revision='b9e7aef^',cases=[],model_loads=0,
                scope='fixed recorded inputs, no simulation, no quality estimate')
    def save():
        _write_json_atomically(run/'summary.json',result)
    print('run_dir='+str(run),flush=True)
    save()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    adapter=None
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch,_=base._check_cuda_and_native(config)
            torch.use_deterministic_algorithms(True)
            requests=json.loads((args.source_run/'rtc/q5_requests.json').read_text())
            distinct={}
            for i,r in enumerate(requests):
                if i and r.get('rtc_solver_inputs',{}).get('previous_raw') is not None:
                    distinct.setdefault(r['rtc_guidance']['valid_prefix_length'],i)
            indices=list(distinct.values())[:3]
            if len(indices)<2:
                raise ValueError('Need at least two recorded exact prefix lengths')
            with np.load(args.source_run/'rtc/q5_frames.npz') as archive:
                frames=archive['frames']
            suite=source_cfg['screen']['suites'][0]
            tid=source_cfg['screen']['task_ids'][0]
            task=base._load_tasks([tid],[suite])[0]
            stats=load_libero_norm_stats(config['norm_stats_path'])
            def snapshot(i):
                r=requests[i]
                frame=frames[r['observation_step']]
                raw=np.asarray(r['observation_raw_state'],dtype=np.float32)
                return dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],
                            raw_state=raw,state=normalize_state(raw,stats))
            model=base._load_and_check_model(config,torch)
            result['model_loads']=1
            from flash_rt.subgraphs.pi05.context_action import enable
            enable(model)
            initial=snapshot(indices[0])
            base._infer_actions(model,initial,task['instruction'],config['H'],{})
            release_unused_native_expert(model,torch)
            adapter=OriginalOpenPiRTCAdapter(model,torch,config)
            adapter.bootstrap(initial,task['instruction'],{})
            lengths=sorted({requests[i]['rtc_guidance']['valid_prefix_length'] for i in indices})
            result['new_bank']=adapter.prepare_step_bank(lengths)
            expert=adapter.expert
            legacy=types.ModuleType('pb_archived_step_graph')
            source=subprocess.check_output(['git','show','b9e7aef^:perceptbudget/rtc_source_step_graph.py'],cwd=ROOT,text=True)
            exec(compile(source,'<archived_pre_memory_fix>','exec'),legacy.__dict__)
            generator=torch.Generator(device='cuda').manual_seed(20261021)
            noise=torch.randn(50,32,device='cuda',generator=generator)
            old=legacy.SourceStepBank(expert,lengths,noise,beta=5.,max_lengths=17)
            result['old_bank_audits']=old.audits
            # Alternate captured lengths and return to earlier entries, refreshing
            # live KV each time. This detects shared-storage overwrite mistakes.
            for i in indices+indices[::-1]:
                r=requests[i]
                adapter.expert=expert
                adapter.bootstrap(snapshot(i),task['instruction'],{})
                pl=model._pipe.pipeline
                valid=int(pl.attn.dec_devpos.item())
                expected=pl.vision_seq_enc+model._pipe.current_prompt_len
                if valid!=expected or valid!=r['rtc_guidance']['valid_prefix_length']:
                    raise RuntimeError('Live pooled prefix differs from recorded/tokenized length')
                pooling=pool_check(pl,torch,np)
                solver=r['rtc_solver_inputs']
                z=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
                previous=torch.tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
                outputs={}
                for name,bank in (('new',adapter.step_bank),('old',old)):
                    bound,graph=bank.select(valid)
                    bound.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                    raw,norms=graph.replay(z,previous,solver['delay'],solver['executed'])
                    outputs[name]=(raw.clone(),norms.clone())
                expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                eager,norms=sample_original_rtc(expert,z,previous,solver['delay'],solver['executed'],beta=5.,author_order=True)
                outputs['eager']=(eager,torch.tensor(norms,device='cuda'))
                row=dict(request_index=i,observation_step=r['observation_step'],valid=valid,
                    encoder_capacity=pl.encoder_seq_len,visual_tokens=pl.vision_seq_enc,
                    tokenized_prompt_length=model._pipe.current_prompt_len,pooling=pooling)
                for left,right in (('new','old'),('new','eager')):
                    row[left+'_vs_'+right]=dict(raw_max_abs=float((outputs[left][0]-outputs[right][0]).abs().max()),
                        vjp_norm_max_abs=float((outputs[left][1]-outputs[right][1]).abs().max()))
                recorded=torch.tensor(r['actions_raw'],device='cuda',dtype=torch.float32)
                row['new_vs_recorded_raw_max_abs']=float((outputs['new'][0]-recorded).abs().max())
                row['passed']=pooling['mismatched_elements']==0 and all(
                    row[k][metric]<1e-4 for k in ('new_vs_old','new_vs_eager')
                    for metric in ('raw_max_abs','vjp_norm_max_abs'))
                result['cases'].append(row)
                save()
                print('parity_case='+json.dumps(row),flush=True)
            result.update(status='completed',passed=all(r['passed'] for r in result['cases']))
            if not result['passed']:
                raise RuntimeError('Parity mismatch; see case outputs')
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if adapter is not None:
            adapter.release_graph()
        save()
        metadata=json.loads((run/'run.json').read_text())
        metadata['status']=result['status']
        _write_json_atomically(run/'run.json',metadata)
        print('PARITY_TERMINAL_STATUS='+json.dumps(dict(run=str(run),status=result['status'],passed=result.get('passed'))),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
