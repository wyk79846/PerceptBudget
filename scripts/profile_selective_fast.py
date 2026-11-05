#!/usr/bin/env python3
"""Fixed-input Full/oldFast/dual-selectiveFast probe, one shared checkpoint.

No scheduler changes or task-success claims. Energy is module VDD_IN during
repeated hot requests, including input preparation, full context and RTC VJP.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class TuneOnce:
    """Keep shared native BF16 choices unchanged between probe frontends."""
    def __init__(self,native):self.native,self.shapes,self.skipped=native,set(),0
    def __getattr__(self,name):return getattr(self.native,name)
    def autotune_bf16_nn(self,*args,**kwargs):
        shape=tuple(args[3:6])
        if len(shape)!=3:raise ValueError('Expected positional native BF16 shape')
        if shape in self.shapes:self.skipped+=1;return
        result=self.native.autotune_bf16_nn(*args,**kwargs)
        self.shapes.add(shape)
        return result


def clock_costs(infer, records, torch, run, write, *, visual_tokens=320,preserve_positions=False,full_attention=False):
    """Hot request/hold costs after clock acknowledgement; one resident model."""
    from perceptbudget.three_frequency import ThreeClocks
    from perceptbudget.power import PowerMonitor
    from perceptbudget.device_clock_screen import energy_window,frequency_check
    clock=ThreeClocks(run);monitor=PowerMonitor(.02,max_gap_seconds=.1)
    result=dict(status='running',cells={},rows=[],restored=False,
        fast_input_contract=dict(visual_mode='selective'+str(visual_tokens),visual_tokens=visual_tokens,state_enabled=full_attention,
            expert_visual_tokens=512 if full_attention else visual_tokens,
            position_policy='full_visual_attention_state_v1' if full_attention else 'original_camera_and_text_coordinates_v1' if preserve_positions else 'contiguous_v1'),
        scope='Fixed-input request energy after frequency acknowledgement; shared entry cost cancels in the same-time same-frequency mode comparison')
    entered=False
    try:
        clock.__enter__();entered=True;monitor.start();time.sleep(.2)
        for profile in ('baseline','gpu_emc'):
            if clock.profile!=profile:
                clock.switch(profile)
                end=time.monotonic()+3.
                while clock.pending is not None:
                    clock.poll()
                    if time.monotonic()>end:raise TimeoutError('Cost probe clock acknowledgement')
                    time.sleep(.002)
            for label in ('Full','DualFast'):infer(label,records[0])
            time.sleep(.3);monitor.sample();a=time.monotonic()
            time.sleep(.6);b=time.monotonic();monitor.sample()
            hold=energy_window(monitor,a,b)
            if hold['energy_j'] is None:raise RuntimeError('Hold power coverage incomplete')
            cell=dict(hold_power_w=hold['energy_j']/(b-a),modes={})
            for label in ('Full','DualFast','DualFast','Full'):
                times=[];time.sleep(.1);monitor.sample();a=time.monotonic()
                for record in records:
                    tick=time.monotonic();infer(label,record);times.append(time.monotonic()-tick)
                b=time.monotonic();time.sleep(.1);monitor.sample()
                energy=energy_window(monitor,a,b)
                if energy['energy_j'] is None:raise RuntimeError('Request power coverage incomplete')
                actual=frequency_check(clock.lease.events,profile,a,b)
                if not actual['matches']:raise RuntimeError('Cost frequency mismatch')
                mode='Full' if label=='Full' else 'Fast'
                row=dict(profile=profile,mode=mode,samples_seconds=times,energy_j=energy['energy_j'],count=len(times),actual=actual)
                result['rows'].append(row)
                sample=cell['modes'].setdefault(mode,dict(samples_seconds=[],energy_j=0.,count=0))
                sample['samples_seconds'].extend(times);sample['energy_j']+=energy['energy_j'];sample['count']+=len(times)
                print('fast_clock_cost='+json.dumps(row),flush=True)
            for sample in cell['modes'].values():
                sample['mean_seconds']=sum(sample['samples_seconds'])/sample['count']
                sample['mean_energy_j']=sample['energy_j']/sample['count']
            result['cells'][profile]=cell;write(run/'ordinary_fast_costs.json',result)
        result['status']='completed'
    finally:
        if entered:clock.close();result['restored']=True
        monitor.stop();write(run/'clock_cost_power.json',[asdict(r) for r in monitor.readings])
        write(run/'ordinary_fast_costs.json',result)
    return result


def kernel_checks(torch,np):
    from perceptbudget.selective_pool import compact,compact_both
    from perceptbudget.selective_pool_cuda import load
    lib=load();rows=[]
    generator=torch.Generator().manual_seed(7)
    for tied in (False,True):
        features=(torch.ones(2,16,16,1152) if tied else torch.randn(2,16,16,1152,generator=generator)).to(torch.bfloat16)
        gpu=features.cuda()
        for count,reference,cameras in ((416,lambda:compact(features,32),1),(320,lambda:compact_both(features),2),(464,lambda:compact_both(features,8),2)):
            gpu.copy_(features)
            scores=torch.empty(64*cameras,device='cuda',dtype=torch.float32)
            mask=torch.empty(64*cameras,device='cuda',dtype=torch.int32)
            mapping=torch.empty(count if cameras==2 else 160,device='cuda',dtype=torch.int32)
            output=torch.empty(count,1152,device='cuda',dtype=torch.bfloat16)
            rope=None
            if cameras==2:
                canonical=torch.arange(768,device='cuda',dtype=torch.int16)[:,None].expand(768,256).contiguous()
                rope=torch.empty(count+32,256,device='cuda',dtype=torch.bfloat16)
            def positions_exact(selected):
                if rope is None:return True
                positions=[]
                for camera in range(2):
                    for index in range(256):
                        r,c=divmod(index,16)
                        if not selected[camera,(r//2)*8+c//2] or (r%2==0 and c%2==0):
                            positions.append(camera*256+index)
                positions+=list(range(512,544))
                expected_positions=torch.tensor(positions,dtype=torch.int16)[:,None].expand(-1,256)
                return torch.equal(rope.cpu().view(torch.int16),expected_positions)
            function=getattr(lib,'selective_pool'+str(count))
            def call():
                code=function(gpu.data_ptr(),scores.data_ptr(),mask.data_ptr(),mapping.data_ptr(),output.data_ptr(),torch.cuda.current_stream().cuda_stream)
                if code:raise RuntimeError('Selector launch failed:'+str(code))
                if rope is not None:
                    code=lib.selective_rope(canonical.data_ptr(),mapping.data_ptr(),rope.data_ptr(),count,count+32,torch.cuda.current_stream().cuda_stream)
                    if code:raise RuntimeError('Original-position gather launch failed:'+str(code))
            call();torch.cuda.synchronize()
            expected,selected=reference()
            error=float((output.cpu().float()-expected.float()).abs().max())
            same_mask=torch.equal(mask.cpu().bool().reshape(selected.shape),selected)
            initial_positions=positions_exact(selected)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            # Change the content after capture: graph must select fresh tokens.
            changed=features.flip(1).contiguous()
            gpu.copy_(changed);graph.replay();torch.cuda.synchronize()
            check,changed_selected=compact(changed,32) if count==416 else compact_both(changed,8 if count==464 else 32)
            replay_error=float((output.cpu().float()-check.float()).abs().max())
            row=dict(visual_tokens=count,tied=tied,max_abs=error,mask_equal=same_mask,replay_changed_input_max_abs=replay_error,
                original_positions_exact=initial_positions,graph_fresh_positions_exact=positions_exact(changed_selected))
            rows.append(row)
            if error or replay_error or not same_mask or not initial_positions or not row['graph_fresh_positions_exact']:
                raise RuntimeError('Native selector/position parity failed: '+str(row))
    return rows


def rollout_pair(cfg,models,adapter,torch,np,task,records,run,cache):
    """Fixedq30, Full versus Full/Fast alternation; identical capture and clocks."""
    import copy,hashlib
    from unittest.mock import patch
    import mujoco
    import robosuite.utils.binding_utils as binding
    from scripts import eval_q as base,probe_delay_buffer as probe
    from scripts.audit_rtc_system import audit
    from perceptbudget.device_quality import install_render_buffer_cleanup
    from perceptbudget.experiment import create_run,_write_json_atomically as write
    from perceptbudget.rtc_source_step_graph import h50_state_prefix_lengths
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.rtc_visual_alternation import FreshViewRTCAdapter
    from perceptbudget.libero_input import load_libero_norm_stats
    report=json.loads((cache/'environment_cache.json').read_text())
    if report.get('status')!='passed' or report.get('version')!=mujoco.__version__:
        raise ValueError('Require a passed matching-version environment cache')
    base._patch_egl_cleanup();install_render_buffer_cleanup(shared_egl=True)
    _,envclass=base._libero_api()
    def factory(cls,xml):
        key=hashlib.sha256((mujoco.__version__+'\0'+xml).encode()).hexdigest()
        path=cache/'environment_models'/(key+'.mjb')
        if not path.is_file():raise RuntimeError('Environment absent from supplied binary cache')
        return cls(mujoco.MjModel.from_binary_path(str(path)))
    cfg=copy.deepcopy(cfg)
    cfg.update(candidate_q=[30],m4_enabled=False,m3_fast_enabled=False,m3_scheduler_mode=None,
        m4_acceptance_group=None,m4_acceptance_manifest=None,
        m3_hold_high_clocks=True,m3_critical_wait=False,m4_lean_continue=True,
        m2_visual_alternation=True,m2_visual_fast_mode='selective320',m2_fast_no_state=True,
        m2_full_per_fast=1,m2_split_delay=True,m2_delay_modes=['full','selective320'],
        rtc_initial_delay_steps=10,rtc_initial_delays_by_mode={'full':10,'selective320':7},rtc_audit_inputs=True)
    cfg['screen']=dict(cfg.get('screen',{}),suites=[task['suite']],task_ids=[task['task_id']])
    if isinstance(adapter,FreshViewRTCAdapter):adapter._activate('full')
    full_valid=OriginalOpenPiRTCAdapter._prepare_context(adapter,records[0]['snapshot'],task['instruction'],{},guided=True)
    lengths=h50_state_prefix_lengths(full_valid,task['instruction'])
    if isinstance(adapter,FreshViewRTCAdapter):visual=adapter
    else:
        if adapter.step_bank is not None:adapter.step_bank.release()
        adapter.step_bank=None;adapter._resident_step_bank=None
        visual=FreshViewRTCAdapter.__new__(FreshViewRTCAdapter);visual.__dict__.update(adapter.__dict__)
    visual.config=cfg;visual._full_model=models['Full'];visual._wrist_model=models['DualFast'];visual._guided_requests=0
    fast=models['DualFast']._pipe
    models['DualFast'].set_prompt(task['instruction'],state=None)
    fast_valid=fast.pipeline.vision_seq_enc+fast.current_prompt_len
    cfg['source_step_prefixes']=lengths+[fast_valid]
    visual.prepare_step_bank(cfg['source_step_prefixes'])
    env=None;episodes=[]
    try:
        with patch.object(binding.MjSim,'from_xml_string',classmethod(factory)):
            env=base._make_env(envclass,task);env._pb_lock_fixtures=True
            for label,enabled in [('Full_q30',False),('DualFast_alternating_q30',True)]:
                cfg['m2_visual_fast_enabled']=enabled
                child=create_run(cfg,'M4','dual-fast-q30',repo_root=ROOT)
                (child/'rtc').mkdir()
                with (child/'rtc/trace.jsonl').open('w') as stream:
                    ep=probe.run_episode(env,task,30,models['Full'],load_libero_norm_stats(cfg['norm_stats_path']),cfg,np,torch,child/'rtc',stream,
                        rtc_adapter=visual,condition=label)
                write(child/'summary.json',dict(status='completed',episode=ep,suite=task['suite'],task_id=task['task_id']))
                checked=audit(child);write(child/'system_audit.json',checked)
                if not checked['passed']:raise RuntimeError('Closed-loop audit failed: '+str(checked['failures'][:5]))
                meta=json.loads((child/'run.json').read_text());meta['status']='completed';write(child/'run.json',meta)
                episodes.append(dict(condition=label,run=str(child),outcome=ep['status'],steps=ep['steps'],seconds=ep['real_seconds'],
                    energy_j=ep['including_drain_power']['total_energy_mj']/1000,
                    underflow_steps=len(ep['underflow_steps']),audit_checks=checked['checks']))
                write(run/f"rollout_{task['suite']}_{task['task_id']}.json",dict(status='running',episodes=episodes,scope='One task; fixedq30, Full versus alternating Full/DualFast; high clocks, no dynamic scheduling change'))
                print('rollout='+json.dumps(episodes[-1]),flush=True)
    finally:
        if env is not None:env.close()
    write(run/f"rollout_{task['suite']}_{task['task_id']}.json",dict(status='completed',episodes=episodes,scope='One task; fixedq30, no success-rate inference'))
    return visual,episodes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-only',action='store_true')
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--request-ids',nargs='+',type=int,default=[1,2,3])
    parser.add_argument('--block-requests',type=int,default=8)
    parser.add_argument('--rollout-pair',action='store_true',help='After fixed-input probe, one Full/alternating-Fast q30 task pair')
    parser.add_argument('--environment-cache',type=Path,help='Read-only passed MuJoCo binary cache for --rollout-pair')
    parser.add_argument('--rollout-only',action='store_true',help='Skip fixed-input comparisons and power blocks; run paired tasks with one model')
    parser.add_argument('--rollout-tasks',nargs='+',help='Explicit suite:task pairs, e.g. libero_spatial:0 libero_object:0 libero_goal:3')
    parser.add_argument('--author-audit-only',action='store_true',help='Compare current RTC graph with author function on saved inputs; no power blocks or rollout')
    parser.add_argument('--clock-costs',action='store_true',help='Measure Full/DualFast at HIGH and MID for ordinary-stage substitution')
    parser.add_argument('--visual-tokens',type=int,choices=(320,464),default=320,help='Dual-camera selective token budget')
    parser.add_argument('--preserve-positions',action='store_true',help='Measure selective input with original camera/text/action positions')
    parser.add_argument('--full-attention',action='store_true',help='Keep all original queries/KV and complete state; pool projection/MLP rows')
    args=parser.parse_args()
    if args.full_attention:args.preserve_positions=True
    if not args.kernel_only and args.source_run is None:parser.error('--source-run is required')
    if not 4<=args.block_requests<=20:parser.error('Use4..20 requests per power block')
    if args.rollout_pair and args.environment_cache is None:parser.error('--rollout-pair needs --environment-cache')
    if (args.rollout_only or args.rollout_tasks) and not args.rollout_pair:parser.error('Rollout options require --rollout-pair')
    if args.author_audit_only and (args.kernel_only or args.rollout_pair):parser.error('Author audit is a standalone fixed-input comparison')
    if args.clock_costs and (args.kernel_only or args.rollout_pair or args.author_audit_only):parser.error('Clock costs are a standalone measurement')
    if args.visual_tokens!=320 and args.rollout_pair:parser.error('Use464 for fixed inputs and clock costs; rollout uses run_m4')
    if args.preserve_positions and not (args.clock_costs or args.author_audit_only):parser.error('Original positions use clock-cost or author-audit mode')
    if args.full_attention and not (args.clock_costs or args.author_audit_only):parser.error('Full attention is a fixed-input/clock probe first')
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    import numpy as np
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically as write
    from scripts import eval_q as base
    cfg=load_config(repo_root=ROOT)
    if args.source_run:
        cfg.update(json.loads((args.source_run/'run.json').read_text())['config_snapshot'])
    cfg.update(m2_visual_fast_mode='selective'+str(args.visual_tokens),m2_fast_no_state=not args.full_attention,
        m3_ordinary_fast_visual_mode='selective'+str(args.visual_tokens),m3_ordinary_fast_positions=args.preserve_positions,
        m3_ordinary_fast_full_attention=args.full_attention)
    with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=create_run(cfg,'M4','dual-selective-fast',repo_root=ROOT)
        shutil.copy2(__file__,run/'probe.py')
        print('run_dir='+str(run),flush=True)
        result=dict(status='running',model_loads=0,clocks_changed=False,source=str(args.source_run),
            cases=[],power_blocks=[],probe_arguments=dict(request_ids=args.request_ids,block_requests=args.block_requests,
                rollout_only=args.rollout_only,rollout_tasks=args.rollout_tasks,environment_cache=str(args.environment_cache),author_audit_only=args.author_audit_only),
            scope=('Paired fixedq30 task attempts; Full versus alternating Full/DualFast, high clocks, no production policy change'
                if args.rollout_pair else 'Fixed-input comparison only; no task execution or energy-saving claim for closed-loop deployment'))
        adapter=monitor=None
        def save():write(run/'summary.json',result)
        save()
        try:
            torch,_=base._check_cuda_and_native(cfg);torch.use_deterministic_algorithms(True)
            result['kernel_checks']=kernel_checks(torch,np);save()
            if args.kernel_only:
                result.update(status='completed',kernel_only=True);return 0
            from perceptbudget.h50_checkpoint import image_input,normalize_state,decode_actions
            from perceptbudget.trace_frames import read_recorded_frame
            from perceptbudget.m4_acceptance import request_path
            from perceptbudget.rtc_memory import release_unused_native_expert,memory_record
            from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
            from perceptbudget.shared_visual_context import wrist_frontend
            from perceptbudget.power import PowerMonitor
            from perceptbudget.device_clock_screen import energy_window
            from flash_rt.subgraphs.pi05.context_action import enable
            from flash_rt.api import VLAModel
            cfg.update(m3_dvac_enabled=False,rtc_audit_inputs=False,rtc_audit_attention_same_input=False,rtc_audit_fused_norm=False)
            requests=json.loads(request_path(args.source_run).read_text())
            source_summary=json.loads((args.source_run/'summary.json').read_text())
            task=base._load_tasks([source_summary['task_id']],[source_summary['suite']])[0]
            path=request_path(args.source_run)
            archive=path.with_name(path.name.replace('_requests.json','_frames.npz'))
            with np.load(archive,allow_pickle=False) as saved:
                steps=saved['frame_steps'].tolist() if 'frame_steps' in saved else None
            stats=json.loads(Path(cfg['norm_stats_path']).read_text());records=[]
            for pid in args.request_ids:
                request=next(r for r in requests if r['prediction_id']==pid)
                if request['rtc_solver_inputs']['previous_raw'] is None:raise ValueError('Require guided requests')
                step=request['observation_step'];index=steps.index(step) if steps is not None else step
                frame=read_recorded_frame(path.with_name(path.name.replace('_requests.json','_frames_spool.npy')),index)
                state=normalize_state(request['observation_raw_state'],stats)
                if not np.array_equal(state,np.asarray(request['model_state'],np.float32)):raise ValueError('Source state mismatch')
                records.append(dict(request=request,snapshot=dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],state=state)))
            def mem(label):
                row=memory_record(torch,label);result.setdefault('memory',[]).append(row);save()
                if row['available_mib']<900:raise RuntimeError('Probe RAM below900MiB; stop without reloading')
            model=base._load_and_check_model(cfg,torch);result['model_loads']=1
            if model._pipe.pipeline is not None:raise ValueError('Install probe tuning guard before first pipeline')
            tuning=TuneOnce(model._pipe.gemm);model._pipe.gemm=tuning
            from perceptbudget.context_preparation import prepare_context_only
            enable(model)
            result['full_context_preparation']=prepare_context_only(model,records[0]['snapshot'],task['instruction'],torch)
            models={'Full':model}
            fast_kind='selective'+str(args.visual_tokens)
            peers=[('DualFast',fast_kind)] if args.rollout_only or args.author_audit_only or args.clock_costs else [('OldFast','pool2_agent'),('DualFast',fast_kind)]
            for label,kind in peers:
                frontend=wrist_frontend(model._pipe,torch,fast_kind=kind,no_state=not args.full_attention,
                    no_state_prompt_capacity=None if args.full_attention else cfg.get('m4_fast_prompt_capacity',20),
                    preserve_positions=args.preserve_positions,expand_kv=args.full_attention)
                peer=VLAModel(frontend,'torch');enable(peer)
                snap=dict(records[0]['snapshot'],state=records[0]['snapshot']['state'] if args.full_attention else None)
                result.setdefault('fast_context_preparation',{})[label]=prepare_context_only(peer,snap,task['instruction'],torch)
                models[label]=peer;mem(label+'_ready')
            result['geometry']={label:dict(visual_tokens=m._pipe.pipeline.vision_seq_enc,state=label=='Full' or args.full_attention,views=m._pipe.pipeline.num_views,
                camera_tokens=[256,256] if label=='Full' else [64,256] if label=='OldFast' else [args.visual_tokens//2]*2) for label,m in models.items()}
            for m in models.values():release_unused_native_expert(m,torch)
            mem('native_experts_released')
            adapter=OriginalOpenPiRTCAdapter(model,torch,cfg)
            if args.preserve_positions and not args.full_attention:
                from perceptbudget.selective_pipeline import align_expert_positions
                result['position_policy']=align_expert_positions(adapter.expert,args.visual_tokens,cfg.get('m4_fast_prompt_capacity',20))
            adapter.bootstrap(records[0]['snapshot'],task['instruction'],{})
            def collect_rollouts():
                nonlocal adapter
                result['runtime_clock_control_used']=True
                result['clock_scope']='Fixed HIGH during episodes, per-episode finite helper restoration; no new frequency policy'
                tasks=[task] if not args.rollout_tasks else []
                for token in args.rollout_tasks or []:
                    suite,tid=token.rsplit(':',1)
                    tasks.append(base._load_tasks([int(tid)],[suite])[0])
                if len({(t['suite'],t['task_id']) for t in tasks})!=len(tasks):raise ValueError('Duplicate rollout task')
                for target in tasks:
                    adapter,episodes=rollout_pair(cfg,models,adapter,torch,np,target,records,run,args.environment_cache)
                    result.setdefault('rollout_pairs',[]).append(dict(suite=target['suite'],task_id=target['task_id'],episodes=episodes))
                    save()
            if args.rollout_only:
                collect_rollouts()
                result.update(status='completed',production_default_changed=False,rollout_only=True)
                return 0
            def context(label,record):
                mdl=models[label];pipe=mdl._pipe;pl=pipe.pipeline
                with torch.cuda.stream(pipe._graph_torch_stream):
                    mdl.set_prompt(task['instruction'],state=record['snapshot']['state'] if label=='Full' or args.full_attention else None)
                    pipe._fill_img_buf({'images':record['snapshot']['images']})
                    pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf,pl.input_images_buf,pipe._graph_torch_stream.cuda_stream)
                    pl._context_graph.replay(pl._graph_stream)
                pipe._graph_torch_stream.synchronize()
                return int(pl.attn.dec_devpos.item())
            lengths=sorted({context(label,r) for r in records for label in models})
            adapter.prepare_step_bank(lengths);mem('source_graphs_ready')
            for record in records:
                solver=record['request']['rtc_solver_inputs']
                record['noise']=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
                record['previous']=torch.tensor(solver['previous_raw'],device='cuda',dtype=torch.float32)
            def infer(label,record):
                valid=context(label,record)
                expert,graph=adapter.step_bank.select(valid)
                if graph is None:raise RuntimeError('Unprepared expert prefix')
                pl=models[label]._pipe.pipeline
                expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                s=record['request']['rtc_solver_inputs']
                raw,_=graph.replay(record['noise'],record['previous'],s['delay'],s['executed'])
                return raw.detach().cpu().numpy().copy()
            if args.clock_costs:
                if args.preserve_positions:
                    from perceptbudget.rtc_author_source import sample_author_source
                    result['positioned_author_audit']=[]
                    record=records[0];s=record['request']['rtc_solver_inputs']
                    for label in ('Full','DualFast'):
                        raw=infer(label,record)
                        valid=int(models[label]._pipe.pipeline.attn.dec_devpos.item())
                        expert,_=adapter.step_bank.select(valid)
                        author,_,_,sha=sample_author_source(expert,record['noise'],record['previous'],s['delay'],s['executed'],
                            source=Path(cfg['output_root'])/'rtc-author-reference/src/model.py',beta=cfg['rtc_beta'])
                        error=float(np.max(np.abs(raw-author.detach().cpu().numpy())))
                        result['positioned_author_audit'].append(dict(mode=label,max_abs=error,author_source_sha256=sha))
                        if error>1e-4:raise RuntimeError('Positioned RTC graph differs from author sampling loop')
                result['clock_costs']=clock_costs(infer,records,torch,run,write,visual_tokens=args.visual_tokens,
                    preserve_positions=args.preserve_positions,full_attention=args.full_attention)
                result.update(status='completed',clocks_changed=True,production_default_changed=False)
                return 0
            if args.author_audit_only:
                from perceptbudget.rtc_author_source import sample_author_source
                from perceptbudget.openpi_reference import sample_original_rtc
                source=Path(cfg['output_root'])/'rtc-author-reference/src/model.py'
                result['scope']='Same current frozen expert/context and recorded noise/old tail; author sampler body versus eager and optimized graph. Not author checkpoint or JAX/XLA deployment reproduction.'
                result['author_audit']=[]
                for record in records:
                    s=record['request']['rtc_solver_inputs']
                    for label in models:
                        raw=infer(label,record)
                        valid=models[label]._pipe.pipeline.vision_seq_enc+models[label]._pipe.current_prompt_len
                        expert,graph=adapter.step_bank.select(valid)
                        reference,ref_norms=sample_original_rtc(expert,record['noise'],record['previous'],s['delay'],s['executed'],beta=cfg['rtc_beta'],author_order=True)
                        author,trace,mask,sha=sample_author_source(expert,record['noise'],record['previous'],s['delay'],s['executed'],source=source,beta=cfg['rtc_beta'])
                        graph_trace=[]
                        got,_=graph.replay(record['noise'],record['previous'],s['delay'],s['executed'],trace=graph_trace)
                        errors={key:max(float((a[key]-z[key]).abs().max()) for a,z in zip(graph_trace,trace))
                            for key in ('input','velocity','clean','error','vjp','output')}
                        error=float(np.max(np.abs(raw-author.detach().cpu().numpy())))
                        row=dict(request_id=record['request']['prediction_id'],condition=label,valid=valid,delay=s['delay'],executed=s['executed'],
                            graph_vs_author_max_abs=error,eager_vs_author_max_abs=float((reference-author).abs().max()),per_step_max_abs=errors,
                            graph_repeat_max_abs=float(np.max(np.abs(raw-got.detach().cpu().numpy()))),author_source_sha256=sha)
                        result['author_audit'].append(row);save();print('author_audit='+json.dumps(row),flush=True)
                        if max(error,*errors.values(),row['eager_vs_author_max_abs'],row['graph_repeat_max_abs'])>1e-4:
                            raise RuntimeError('Current real-input RTC differs from author sampler')
                result.update(status='completed',production_default_changed=False,author_audit_passed=True)
                return 0
            outputs={}
            for record in records:
                pid=record['request']['prediction_id'];start=record['request']['rtc_solver_inputs']['delay']
                for label in ('Full','OldFast','DualFast','Full'):
                    raw=infer(label,record);key=label+'_'+str(pid)
                    if key in outputs:
                        if not np.array_equal(raw,outputs[key]):raise RuntimeError('Full repeat changed')
                        continue
                    outputs[key]=raw
                    diff=decode_actions(raw,stats)-decode_actions(outputs['Full_'+str(pid)],stats)
                    action=decode_actions(raw,stats)
                    switches=np.flatnonzero(np.signbit(action[:-1,6])!=np.signbit(action[1:,6]))+1
                    result['cases'].append(dict(request_id=pid,observation_step=record['request']['observation_step'],condition=label,
                        next10_arm6_rmse=float(np.sqrt(np.mean(diff[start:start+10,:6]**2))),
                        all50_arm6_rmse=float(np.sqrt(np.mean(diff[:,:6]**2))),
                        next10_gripper_max_abs=float(np.max(np.abs(diff[start:start+10,6]))),
                        gripper_switch_indices=switches.tolist()))
                    save()
            # Validate native packing against CPU reference on actual SigLIP features.
            from perceptbudget.selective_pool import compact_both
            pl=models['DualFast']._pipe.pipeline
            bits=np.empty(512*1152,np.uint16);pl.bufs['vision_x'].download(bits)
            expected,_=compact_both(torch.from_numpy(bits).view(torch.bfloat16).reshape(2,16,16,1152),8 if args.visual_tokens==464 else 32)
            packed=np.empty(args.visual_tokens*1152,np.uint16);pl.bufs['vision_x_pooled'].download(packed)
            if not torch.equal(expected,torch.from_numpy(packed).view(torch.bfloat16).reshape(args.visual_tokens,1152)):
                raise RuntimeError('Live selective packing differs from CPU reference')
            result['live_layout_exact']=True
            monitor=PowerMonitor(.02,max_gap_seconds=.1);monitor.start();time.sleep(.3)
            for block,order in enumerate((('Full','OldFast','DualFast'),('DualFast','OldFast','Full'))):
                for label in order:
                    record=records[0];infer(label,record);time.sleep(.3);monitor.sample()
                    started=time.monotonic();latencies=[]
                    for _ in range(args.block_requests):
                        tick=time.monotonic();infer(label,record);latencies.append(time.monotonic()-tick)
                    stopped=time.monotonic();time.sleep(.2);monitor.sample()
                    energy=energy_window(monitor,started,stopped)
                    if energy['energy_j'] is None:raise RuntimeError('Power coverage incomplete')
                    row=dict(block=block,condition=label,requests=args.block_requests,
                        median_ms=float(np.median(latencies)*1000),seconds=stopped-started,energy=energy,
                        mean_j_per_request=energy['energy_j']/args.block_requests)
                    result['power_blocks'].append(row);save();print('power_block='+json.dumps(row),flush=True)
            np.savez_compressed(run/'raw_actions.npz',**outputs)
            result['autotune_guard']=dict(unique_shapes=sorted(tuning.shapes),duplicate_tunes_skipped=tuning.skipped,
                scope='Same native per-shape winners shared inside this probe process; not cross-process algorithm pinning')
            if args.rollout_pair:
                monitor.stop();write(run/'power.json',[asdict(r) for r in monitor.readings]);monitor=None
                collect_rollouts()
            result.update(status='completed',production_default_changed=False)
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            if monitor is not None:
                monitor.stop();write(run/'power.json',[asdict(r) for r in monitor.readings])
            if adapter is not None:adapter.release_graph()
            save();print('DUAL_FAST_STATUS='+result['status'],flush=True)
            meta=json.loads((run/'run.json').read_text());meta['status']=result['status'];write(run/'run.json',meta)
    return 0


if __name__=='__main__':raise SystemExit(main())
