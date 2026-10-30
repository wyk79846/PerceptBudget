#!/usr/bin/env python3
"""Recorded-input check of cross-task RTC graph retention; no rollout or clocks."""
import fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    args=parser.parse_args()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    from scripts import eval_q as base
    from perceptbudget.h50_checkpoint import image_input,normalize_state
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.rtc_source_step_graph import SourceStepBank
    from perceptbudget.openpi_reference import sample_original_rtc
    cfg=load_config(repo_root=ROOT)
    cfg.update(rtc_audit_inputs=False,rtc_audit_attention_same_input=False,rtc_audit_fused_norm=False)
    source=json.loads((args.source_run/'run.json').read_text())['config_snapshot']
    for key in ('H','N','checkpoint_provenance_revision','state_enabled','rtc_beta'):
        if cfg[key]!=source[key]:raise ValueError('Source mismatch '+key)
    run=create_run(cfg,'M4','resident-bank-parity',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    result=dict(status='running',source_run=str(args.source_run),cases=[],scope='same live KV, noise and prefix; fresh graph vs retained graph vs eager; no quality estimate')
    def save():_write_json_atomically(run/'summary.json',result)
    adapter=None
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch,_=base._check_cuda_and_native(cfg);torch.use_deterministic_algorithms(True)
            requests=json.loads((args.source_run/'rtc/q5_requests.json').read_text())
            choices={}
            for i,r in enumerate(requests):
                if r.get('rtc_solver_inputs',{}).get('previous_raw') is not None:
                    choices.setdefault(r['rtc_guidance']['valid_prefix_length'],i)
            indices=list(choices.values())[:3]
            if not indices:raise ValueError('No guided samples')
            with np.load(args.source_run/'rtc/q5_frames.npz') as z:frames=z['frames']
            stats=load_libero_norm_stats(cfg['norm_stats_path'])
            task=base._load_tasks(source['screen']['task_ids'],source['screen']['suites'])[0]
            def snapshot(i):
                r=requests[i];frame=frames[r['observation_step']];raw=np.asarray(r['observation_raw_state'],dtype=np.float32)
                return dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],raw_state=raw,state=normalize_state(raw,stats))
            model=base._load_and_check_model(cfg,torch)
            from flash_rt.subgraphs.pi05.context_action import enable
            enable(model);base._infer_actions(model,snapshot(indices[0]),task['instruction'],cfg['H'],{})
            release_unused_native_expert(model,torch)
            adapter=OriginalOpenPiRTCAdapter(model,torch,cfg)
            adapter.bootstrap(snapshot(indices[0]),task['instruction'],{})
            expert=adapter.expert
            lengths=sorted({requests[i]['rtc_guidance']['valid_prefix_length'] for i in indices})
            noise=torch.randn(cfg['H'],32,device='cuda',dtype=torch.float32)
            # Enable the same packed conditioning and capture stream as production.
            adapter.prepare_step_bank(lengths)
            fresh=adapter.step_bank;adapter.step_bank=None
            resident=SourceStepBank(expert,[lengths[0]],noise,beta=5.,max_lengths=17,stream=adapter._step_bank_stream,resident=True)
            resident.prepare_resident(lengths,noise,beta=5.)
            for i in indices+indices[::-1]:
                adapter.expert=expert;adapter.bootstrap(snapshot(i),task['instruction'],{})
                r=requests[i];valid=r['rtc_guidance']['valid_prefix_length'];solver=r['rtc_solver_inputs']
                pl=model._pipe.pipeline
                if int(pl.attn.dec_devpos.item())!=valid:raise ValueError('Live prefix mismatch')
                z=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
                prev=torch.tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
                outputs={}
                for name,bank in [('fresh',fresh),('resident',resident)]:
                    bound,graph=bank.select(valid);bound.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                    raw,norms=graph.replay(z,prev,solver['delay'],solver['executed']);outputs[name]=(raw.clone(),norms.clone())
                expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                raw,norms=sample_original_rtc(expert,z,prev,solver['delay'],solver['executed'],beta=5.,author_order=True)
                row=dict(request_index=i,valid=valid,retained=(valid==lengths[0]),resident_vs_fresh_raw=float((outputs['resident'][0]-outputs['fresh'][0]).abs().max()),resident_vs_eager_raw=float((outputs['resident'][0]-raw).abs().max()),resident_vs_fresh_norm=float((outputs['resident'][1]-outputs['fresh'][1]).abs().max()))
                row['passed']=all(row[k]<1e-4 for k in ('resident_vs_fresh_raw','resident_vs_eager_raw','resident_vs_fresh_norm'))
                result['cases'].append(row);save();print('PARITY_CASE='+json.dumps(row),flush=True)
            result.update(status='completed',passed=all(r['passed'] for r in result['cases']))
            return 0 if result['passed'] else 1
    except BaseException as e:
        result.update(status='invalid',error=str(e));raise
    finally:
        save()
        if adapter is not None:adapter.release_graph()

if __name__=='__main__':raise SystemExit(main())
