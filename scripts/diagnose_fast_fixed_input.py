#!/usr/bin/env python3
"""Same recorded observation/noise/prefix/delay: Full vs pooled with/without state.

Offline action sensitivity only, not a rollout or inference latency benchmark.
"""
import argparse
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def recorded_frame_index(summary, observation_step):
    """Request-only capture uses a compact frame array, not control-step indices."""
    capture=summary['episode'].get('continue_capture')
    if capture and capture.get('frame_sampling')=='request observations only':
        steps=capture['frame_steps']
        if steps.count(observation_step)!=1:
            raise ValueError('Require one saved request frame at selected observation step')
        return steps.index(observation_step)
    return observation_step


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--observation-step',type=int,default=64)
    selective=parser.add_mutually_exclusive_group()
    selective.add_argument('--selective416',action='store_true',help='Test production selective416 frontend with state')
    selective.add_argument('--selective320',action='store_true',help='Separate state and dual selective320 effects on identical recorded inputs')
    selective.add_argument('--selective464',action='store_true',help='Mild dual-view pooling,8 of64 blocks per camera')
    parser.add_argument('--preserve-positions',action='store_true',help='Probe original image/text/action coordinates with stateless selective320/464')
    parser.add_argument('--stateful-positions',action='store_true',help='Also test complete state with original-position320/464 and explicit expert geometry')
    parser.add_argument('--resident-check',action='store_true',help='Compare resident712-capacity source-step graph with reference on actual inputs')
    parser.add_argument('--mapped-embeddings',action='store_true',help='Before Full_repeat, map frozen vocabulary rows and check all BF16 table values')
    parser.add_argument('--expanded-kv',action='store_true',help='Keep original visual KV positions/multiplicity while computing compact encoder queries and MLPs')
    args=parser.parse_args()
    if args.preserve_positions and (not (args.selective320 or args.selective464) or args.resident_check):
        parser.error('Original-position input probe requires selective320/464 without resident-check')
    if args.stateful_positions and not ((args.selective320 or args.selective464) and args.preserve_positions):
        parser.error('Stateful position probe requires selective320/464 and --preserve-positions')
    if args.expanded_kv and not ((args.selective320 or args.selective464) and args.preserve_positions and not args.stateful_positions):
        parser.error('Expanded KV uses selective320/464, original positions and no action-position offset')
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
    if args.mapped_embeddings:cfg['mapped_prompt_embedding']=False  # Native control before Full_repeat.
    requests=json.loads((args.source_run/'rtc/q5_requests.json').read_text())
    matches=[r for r in requests if r['observation_step']==args.observation_step]
    if len(matches)!=1:raise ValueError('Require exactly one recorded request at selected step')
    r=matches[0];solver=r['rtc_solver_inputs']
    if solver['previous_raw'] is None:raise ValueError('Require guided request')
    run=create_run(cfg,'M4','fast-fixed-input',repo_root=ROOT)
    shutil.copy2(__file__,run/'probe.py')
    result=dict(status='running',source_run=str(args.source_run),observation_step=args.observation_step,
                delay=solver['delay'],executed=solver['executed'],model_loads=0,cases={},
                probe_arguments=dict(selective320=args.selective320,selective416=args.selective416,
                    selective464=args.selective464,preserve_positions=args.preserve_positions,
                    stateful_positions=args.stateful_positions,resident_check=args.resident_check,mapped_embeddings=args.mapped_embeddings,expanded_kv=args.expanded_kv),
                scope='Identical observation/noise/previous_raw/delay/executed; action sensitivity, no closed-loop outcome or latency claim')
    def save():_write_json_atomically(run/'summary.json',result)
    print('run_dir='+str(run),flush=True);save();adapter=None
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch,_=base._check_cuda_and_native(cfg);torch.use_deterministic_algorithms(True)
            saved_summary=json.loads((args.source_run/'summary.json').read_text())
            frame_index=recorded_frame_index(saved_summary,args.observation_step)
            from perceptbudget.trace_frames import read_recorded_frame
            frame=read_recorded_frame(args.source_run/'rtc/q5_frames_spool.npy',frame_index)
            raw=np.asarray(r['observation_raw_state'],np.float32)
            stats=load_libero_norm_stats(cfg['norm_stats_path'])
            snapshot=dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],raw_state=raw,state=normalize_state(raw,stats))
            if not np.array_equal(snapshot['state'],np.asarray(r['model_state'],np.float32)):
                raise ValueError('Recorded state normalization differs')
            import hashlib
            hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in snapshot['images']]
            if hashes!=r['context_refresh']['image_input_sha256']:raise ValueError('Recorded image preprocessing differs')
            result['input_checks']=dict(images_exact=True,state_exact=True,frame_index=frame_index)
            task=base._load_tasks([source['screen']['task_ids'][0]],[source['screen']['suites'][0]])[0]
            model=base._load_and_check_model(cfg,torch);result['model_loads']=1
            from flash_rt.subgraphs.pi05.context_action import enable
            from flash_rt.api import VLAModel
            from perceptbudget.context_preparation import prepare_context_only
            enable(model)
            result['full_context_preparation']=prepare_context_only(model,snapshot,task['instruction'],torch)
            # Both conditions share one fixed-capacity context. Only actual
            # valid prefix tokens attend; no padding token enters the expert.
            # Avoid the legacy state=None exact-length pipeline reconstruction.
            import types
            kind='selective464' if args.selective464 else 'selective416' if args.selective416 else 'selective320' if args.selective320 else 'pool2_agent'
            frontend=wrist_frontend(model._pipe,torch,fast_kind=kind,no_state=False,
                preserve_positions=args.preserve_positions,expand_kv=args.expanded_kv)
            def fixed_prompt(self,state,prompt_len):self._set_prompt_fixed(prompt_len)
            frontend._set_prompt_per_length=types.MethodType(fixed_prompt,frontend)
            model._pipe._set_prompt_per_length=types.MethodType(fixed_prompt,model._pipe)
            pooled=VLAModel(frontend,'torch')
            enable(pooled)
            result['fast_context_preparation']=prepare_context_only(pooled,snapshot,task['instruction'],torch)
            release_unused_native_expert(model,torch);release_unused_native_expert(pooled,torch)
            adapter=OriginalOpenPiRTCAdapter(model,torch,cfg)
            active_geometry=dict(fast=False)
            if args.expanded_kv:
                result['position_policy']=dict(policy='full_visual_multiplicity_and_coordinates_v1',
                    action_position_offset=0,expert_visual_tokens=512,encoder_compute_visual_tokens=320 if args.selective320 else 464,
                    rtc_sampler_changed=False)
            elif args.preserve_positions:
                from perceptbudget.selective_pipeline import align_expert_positions
                result['position_policy']=align_expert_positions(adapter.expert,
                    464 if args.selective464 else 320,200 if args.stateful_positions else 32,
                    active_fast=(lambda:active_geometry['fast']) if args.stateful_positions else None)
            # Initialize the frozen expert attention/norm implementation via normal adapter.
            adapter.bootstrap(snapshot,task['instruction'],{})
            noise=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
            previous=torch.tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
            if args.resident_check:
                if not args.selective416:raise ValueError('Resident check requires selective416')
                full_valid=512+model._pipe.current_prompt_len
                adapter._dual_bank_preparing=True
                try:adapter.prepare_step_bank([full_valid,full_valid-96])
                finally:adapter._dual_bank_preparing=False
            outputs={}
            conditions=[('Full',model,True),('Fast_state',pooled,True),('Full_repeat',model,True)] if args.resident_check else [('Full',model,True),('Fast',pooled,False),('Fast_state',pooled,True),('Full_no_state',model,False),('Full_repeat',model,True)]
            if args.preserve_positions and not (args.stateful_positions or args.expanded_kv):conditions=[c for c in conditions if c[0]!='Fast_state']
            for name,frontend,state in conditions:
                active_geometry['fast']=not name.startswith('Full')
                if name=='Full_repeat' and args.mapped_embeddings:
                    from perceptbudget.mapped_embedding import install_mapped_prompt_embeddings
                    from perceptbudget.rtc_memory import memory_record
                    result['before_mapped_embedding']=memory_record(torch,'before_mapped_embedding')
                    result['mapped_embedding']=install_mapped_prompt_embeddings(
                        [model._pipe,pooled._pipe],cfg['model_path'],
                        checkpoint_prefix=cfg.get('checkpoint_key_prefix',''),verify_all=True)
                    result['after_mapped_embedding']=memory_record(torch,'after_mapped_embedding')
                adapter.model=frontend;adapter.pipe=frontend._pipe;adapter.pipeline=frontend._pipe.pipeline
                data=dict(snapshot,state=snapshot['state'] if state else None)
                valid=adapter._prepare_context(data,task['instruction'],{},guided=True)
                pl=frontend._pipe.pipeline
                expected=(512 if name.startswith('Full') or args.expanded_kv else 464 if args.selective464 else 416 if args.selective416 else 320)+frontend._pipe.current_prompt_len
                if valid!=expected:raise ValueError('Live valid prefix differs from actual tokens')
                if not name.startswith('Full'):
                    from scripts.diagnose_pool_parity import pool_check
                    if args.selective416 or args.selective320 or args.selective464:
                        from perceptbudget.selective_pool import compact,compact_both
                        bits=np.empty(512*1152,dtype=np.uint16);pl.bufs['vision_x'].download(bits)
                        features=torch.from_numpy(bits).view(torch.bfloat16).reshape(2,16,16,1152)
                        count=464 if args.selective464 else 416 if args.selective416 else 320
                        expected_tokens,_=compact(features,32) if args.selective416 else compact_both(features,8 if args.selective464 else 32)
                        bits=np.empty(count*1152,dtype=np.uint16);pl.bufs['vision_x_pooled'].download(bits)
                        actual=torch.from_numpy(bits).view(torch.bfloat16).reshape(count,1152)
                        layout=dict(max_abs=float((actual.float()-expected_tokens.float()).abs().max()),mismatched_elements=int((actual!=expected_tokens).sum()),camera_token_counts=[160,256] if args.selective416 else [232,232] if args.selective464 else [160,160])
                    else:layout=pool_check(pl,torch,np,'agent')
                    result.setdefault('pooling_checks',{})[name]=layout
                    if layout['mismatched_elements']:raise ValueError('Native pooling differs from CPU reference')
                adapter.expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                actions,norms=sample_original_rtc(adapter.expert,noise.clone(),previous.clone(),solver['delay'],solver['executed'],beta=cfg['rtc_beta'],author_order=True)
                out=actions.detach().cpu().numpy().copy();outputs[name]=out
                if not np.isfinite(out).all():raise ValueError('Nonfinite actions')
                diff=out[:,:7]-outputs['Full'][:,:7];start=solver['delay']
                result['cases'][name]=dict(state=state,visual_tokens=pl.vision_seq_enc,valid_prefix=valid,
                    expert_visual_tokens=512 if args.expanded_kv else pl.vision_seq_enc,
                    all_action7_rmse=float(np.sqrt(np.mean(diff**2))),
                    next10_executable_action7_rmse=float(np.sqrt(np.mean(diff[start:start+10]**2))),
                    next10_executable_arm6_rmse=float(np.sqrt(np.mean(diff[start:start+10,:6]**2))),
                    action7_max_abs=float(np.max(np.abs(diff))))
                result['cases'][name]['vs_recorded_raw_max_abs']=float(np.max(np.abs(out-np.asarray(r['actions_raw']))))
                if args.resident_check:
                    bound,graph=adapter.step_bank.select(valid)
                    if graph is None:raise ValueError('Missing resident graph')
                    bound.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                    graph_raw,_=graph.replay(noise,previous,solver['delay'],solver['executed'])
                    graph_out=graph_raw.detach().cpu().numpy().copy()
                    result['cases'][name]['resident_vs_reference_max_abs']=float(np.max(np.abs(graph_out-out)))
                    result['cases'][name]['resident_vs_recorded_max_abs']=float(np.max(np.abs(graph_out-np.asarray(r['actions_raw']))))
                    if result['cases'][name]['resident_vs_reference_max_abs']>1e-4:raise ValueError('Resident graph differs from reference on real input')
                save();print(name+'='+json.dumps(result['cases'][name]),flush=True)
            result['factorial_pairs']={}
            pairs=([('Full','Full_no_state'),('Full','Fast'),('Full_no_state','Fast')] if args.preserve_positions and not (args.stateful_positions or args.expanded_kv) else
                [('Full','Fast_state')] if args.resident_check else [('Full','Full_no_state'),('Full','Fast_state'),('Full_no_state','Fast'),('Fast_state','Fast')])
            for a,b in pairs:
                delta=outputs[a][:,:7]-outputs[b][:,:7]
                result['factorial_pairs'][a+'_vs_'+b]=dict(all_action7_rmse=float(np.sqrt(np.mean(delta**2))),next10_executable_action7_rmse=float(np.sqrt(np.mean(delta[solver['delay']:solver['delay']+10]**2))))
            np.savez(run/'actions.npz',**outputs)
            if args.mapped_embeddings:
                result['mapped_full_raw_max_abs_all32']=float(np.max(np.abs(outputs['Full_repeat']-outputs['Full'])))
                if result['mapped_full_raw_max_abs_all32']!=0:
                    raise ValueError('Mapped embedding changed Full RTC output')
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
