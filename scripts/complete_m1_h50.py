"""Six planned disturbance episodes and power windows with one resident model.

No main-scan records are changed. Context and suffix load exactly once; only
environment and exact-shape graph banks are reset between the two tasks.
"""
import argparse
import copy
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import eval_q as base, probe_delay_buffer as probe
from scripts.diagnose_rtc_handoff import sync_episode
from perceptbudget.disturbance import make_hook
from scripts.probe_m1_h50_power import reference_config, guided_request, measure_block
from perceptbudget.power_windows import block_schedule, measurement_summary
from scripts.run_m1_h50 import compare_initial
from perceptbudget.experiment import create_run


def plan():
    return [(t,m,q) for t in (0,3) for m,q in
            (('sync_no_rtc',5),('actual_history',5),('actual_history',40))]


def pending_cases(episodes):
    completed=[(e['task_id'],e['mode'],e['q']) for e in episodes]
    if len(completed)!=len(set(completed)) or any(c not in plan() for c in completed):
        raise ValueError('duplicate or unexpected resident episode')
    return [c for c in plan() if c not in completed]


def cleanup(torch):
    import ctypes
    gc.collect();torch.cuda.empty_cache()
    ctypes.CDLL(None).malloc_trim(0)


def calibrate(env,task,np):
    obs,_=base._prepare_env_episode(env,task,np)
    samples=[]
    for _ in range(12):
        started=time.monotonic()
        probe.snapshot_scene(env,obs)
        np.concatenate((obs['agentview_image'],obs['robot0_eye_in_hand_image']),axis=1)
        obs,_,_,_=env.step(np.zeros(7,dtype=np.float32))
        probe.snapshot_scene(env,obs);env.check_success()
        time.sleep(max(0.,.05-(time.monotonic()-started)))
        samples.append(time.monotonic()-started)
    obs,_=base._prepare_env_episode(env,task,np)
    return obs,dict(samples_seconds=samples,median_seconds=float(np.median(samples)),
                    suite=task['suite'],task_id=task['task_id'])


def initial_info(run,episode):
    import hashlib,struct
    return dict(run=str(run),initial_hashes={
        **{k:episode[k] for k in ('initial_sim_state_sha256','initial_frames_sha256')},
        'initial_raw_state_sha256':hashlib.sha256(struct.pack('<8d',*episode['initial_raw_state'])).hexdigest()})


def power_windows(root,adapter,env,task,stats,np,torch):
    from perceptbudget.power import PowerMonitor
    from dataclasses import asdict
    power_config=copy.deepcopy(adapter.config)
    power_config.update(rtc_audit_inputs=False,rtc_audit_attention_same_input=False,
                        rtc_audit_fused_norm=False,execution_mode='m1_h50_resident_power')
    adapter.config=power_config
    obs,_=base._prepare_env_episode(env,task,np)
    snapshot=base._make_snapshot(obs,stats,True)
    first={};adapter.bootstrap(snapshot,task['instruction'],first)
    previous=np.asarray(first['actions_raw'],dtype=np.float32)[5:].copy()
    valid=first['rtc_guidance']['valid_prefix_length']
    power_config['source_step_prefixes']=[valid]
    power_config['screen'].update(suites=['libero_spatial'],task_ids=[task['task_id']])
    graph_audit=adapter.prepare_step_bank([valid])
    guided_request(adapter,snapshot,task['instruction'],previous,-1)
    result=dict(status='running',blocks=[],requests=[],resident_model_reused=True,
        power_rail='INA3221 VDD_IN module input',poll_interval_seconds=.02,
        scope='fixed snapshot, full context + guided RTC VJP; simulation/rendering active; no power or clock changes',
        device_before=probe.device_state(),config_snapshot=power_config,graph_audit=graph_audit,
        limitations=['fixed snapshot power calibration, not task success or implemented DVFS',
                    '20ms polling is not the sensor hardware update frequency; drain is separate from idle'])
    def save():base._write_json_atomic(root/'power.json',result)
    def tick():
        started=time.monotonic();env.step(np.zeros(7,dtype=np.float32))
        time.sleep(max(0.,.05-(time.monotonic()-started)))
    def predict(index):
        request=guided_request(adapter,snapshot,task['instruction'],previous,index)
        if 'rtc_solver_inputs' in request:raise RuntimeError('power measurement must not retain solver audit arrays')
        result['requests'].append(request)
    monitor=PowerMonitor(poll_interval=.02);monitor.start();save()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for index,mode in enumerate(block_schedule()):
                block=measure_block(mode,index,executor,predict,tick)
                monitor.sample();block['device']=probe.device_state()
                block['drain_power']=asdict(monitor.integrate_report(block['drain_start_monotonic'],block['drain_stop_monotonic']))
                result['blocks'].append(block);save()
                print(f'power_block={index+1}/7 mode={mode}',flush=True)
        monitor.stop()
        result.update(measurement_summary(monitor.readings,result['blocks']),status='completed',device_after=probe.device_state())
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        monitor.stop()
        result['raw_readings']=[dict(timestamp_monotonic=r.timestamp,power_mw=r.power_mw,error=r.error) for r in monitor.readings]
        save()
    return result


def run(batch,resume=None):
    import numpy as np
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import memory_record,release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.rtc_source_step_graph import h50_state_prefix_lengths
    from scripts.audit_rtc_system import audit
    from perceptbudget.measurement import timed
    index=json.loads((batch/'index.json').read_text())
    if index.get('status')!='completed' or index.get('completed')!=300:
        raise ValueError('complete the main 300 episodes first')
    records=index['cells']
    reference=Path(records['libero_spatial_task00_q5_actual_history']['result']['run'])
    config=reference_config(reference)
    config.update(execution_mode='m1_h50_resident_completion',release_rtc_graph_before_artifacts=False,disk_backed_frames=True,
                  source_prefix_coverage_rule='h50_state8_tokenizer_bounds_v1')
    # reference_config turns audit recording off for power; episodes require it.
    config['rtc_audit_inputs']=True
    os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root=resume if resume else create_run(config,'M1','h50-resident-completion',repo_root=ROOT)
        print('completion_dir='+str(root),flush=True)
        if resume:
            summary=json.loads((root/'summary.json').read_text())
            if summary['main_batch']!=str(batch) or summary.get('expected')!=6:
                raise ValueError('resident resume batch mismatch')
            for row in summary['episodes']:
                if (row['task_id'],row['mode'],row['q']) not in plan():raise ValueError('unexpected saved resident case')
                child=Path(row['run'])
                if json.loads((child/'summary.json').read_text()).get('status')!='completed' or not audit(child)['passed']:
                    raise ValueError('saved resident episode failed resume audit')
            if summary.get('status')=='completed':return root
            backup=root/f'summary_before_resume_{time.time_ns()}.json'
            base._write_json_atomic(backup,summary)
            summary.setdefault('exception_restarts',[]).append(dict(
                prior_summary=str(backup),reason='guarded exit after4 episodes, renderer/reset memory pressure; resume missing only',
                completed_before=len(summary['episodes']),source_commit=subprocess.check_output(
                    ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
            summary['status']='loading'
        else:
            summary=dict(status='loading',main_batch=str(batch),episodes=[],expected=6,
                         model_load_count=0,suffix_load_count=0,task_graph_audits=[],memory=[])
        def save():
            base._write_json_atomic(root/'summary.json',summary)
            base._save_run_status(root,summary['status'])
        summary['egl_cleanup']=base._patch_egl_cleanup();save()
        torch,_=base._check_cuda_and_native(config);torch.use_deterministic_algorithms(True)
        def mem(stage):
            cleanup(torch);r=memory_record(torch,stage);summary['memory'].append(r);save()
            print('resident_memory='+json.dumps(r),flush=True)
            if r['available_mib']<900:raise RuntimeError('resident memory headroom below900MiB; stop without repeated model loading')
            return r
        stats=load_libero_norm_stats(config['norm_stats_path'])
        env=adapter=model=None
        active_child=None
        try:
            from flash_rt.subgraphs.pi05.context_action import enable
            mem('before_model_load')
            model=base._load_and_check_model(config,torch);summary['model_load_count']+=1;enable(model)
            mem('after_model_load')
            _,cls=base._libero_api()
            root_expert=None
            remaining=pending_cases(summary['episodes'])
            for task_id in sorted({s[0] for s in remaining}) or [3]:
                if adapter is not None:
                    torch.cuda.synchronize();adapter.step_bank=None;adapter.expert=root_expert
                    cleanup(torch);mem('old_task_bank_released_before_environment')
                task=base._load_tasks([task_id],['libero_spatial'])[0]
                env=base._make_env(cls,task);env._pb_lock_fixtures=True
                obs,calibration=calibrate(env,task,np)
                snapshot=base._make_snapshot(obs,stats,True)
                if adapter is None:
                    base._infer_actions(model,snapshot,task['instruction'],50,{})
                    summary['native_release']=release_unused_native_expert(model,torch)
                    mem('after_native_release')
                    adapter=OriginalOpenPiRTCAdapter(model,torch,config);root_expert=adapter.expert
                    mem('after_suffix_load')
                    summary['suffix_load_count']+=1
                    gate=json.loads((Path(config['verified_reference_run'])/'summary.json').read_text())
                    if root_expert.provenance!=gate['original_source_hashes']:raise RuntimeError('source gate mismatch')
                first={};adapter.bootstrap(snapshot,task['instruction'],first)
                lengths=h50_state_prefix_lengths(first['rtc_guidance']['valid_prefix_length'],task['instruction'])
                adapter.config=config
                summary['task_graph_audits'].append(dict(task_id=task_id,**adapter.prepare_step_bank(
                    lengths,progress=lambda r:mem(f'task{task_id}_prefix{r["valid"]}_prepared'))))
                previous=np.asarray(first['actions_raw'],dtype=np.float32)[5:]
                timing=timed(torch,lambda:adapter.predict(snapshot,task['instruction'],previous,delay=5,executed=5,request={}),count=3)
                config['rtc_initial_delay_steps']=math.ceil(timing['median_ms']/1000/calibration['median_seconds'])
                config['rtc_initial_control_period_seconds']=calibration['median_seconds']
                config['rtc_initial_control_period_source']='same-environment 12 zero-action ticks; reset before episode'
                mem(f'task{task_id}_prepared')
                summary['status']='running';save()
                for tid,mode,q in [s for s in remaining if s[0]==task_id]:
                    cfg=copy.deepcopy(config)
                    cfg.update(candidate_q=[q],diagnostic_condition=mode,execution_mode='diagnostic_'+mode,
                        rtc_fixed_delay_estimate=0 if mode=='sync_no_rtc' else None,
                        rtc_paper_schedule=mode=='actual_history',
                        rtc_initial_delay_steps=0 if mode=='sync_no_rtc' else config['rtc_initial_delay_steps'],
                        source_step_prefixes=lengths,disturbance=dict(target='akita_black_bowl_1',step=30,delta_xyz_m=[.02,0,0]))
                    cfg['screen'].update(suites=['libero_spatial'],task_ids=[task_id])
                    adapter.config=cfg
                    child=create_run(cfg,'M1','h50-resident-nudge',repo_root=ROOT);directory=child/'rtc';directory.mkdir()
                    active_child=child
                    base._write_json_atomic(child/'control_period_calibration.json',calibration)
                    events=[]
                    mem(f'before_task{task_id}_{mode}_q{q}')
                    with (directory/'trace.jsonl').open('w') as stream:
                        args=(env,task,q,model,stats,cfg,np,torch,directory,stream)
                        kwargs=dict(rtc_adapter=adapter,step_hook=make_hook(True,events),refresh_policy='latest_due',pending_arm_gain=1.)
                        episode=(sync_episode(probe.run_episode,*args,**kwargs) if mode=='sync_no_rtc' else
                                 probe.run_episode(*args,**kwargs))
                    result=dict(status='validating',episode=episode,disturbance_events=events,
                        suite=task['suite'],task_id=task_id,max_steps=task['max_steps'],diagnostic_condition=mode,
                        resident_model_reused=True,shared_graph_bank=True)
                    base._write_json_atomic(child/'summary.json',result)
                    if episode['status'] not in ('success','timeout','failure'):
                        raise RuntimeError('unexpected episode termination')
                    if len(events)!=1:raise RuntimeError('exactly one planned disturbance required')
                    clean=records[f'libero_spatial_task{task_id:02d}_q{q}_{mode}']['result']
                    result['initial_pair_audit']=compare_initial(clean,initial_info(child,episode))
                    checked=audit(child);base._write_json_atomic(child/'system_audit.json',checked)
                    if not checked['passed']:raise RuntimeError('resident episode record audit failed')
                    requests=json.loads((directory/f'q{q}_requests.json').read_text())
                    if any(r['rtc_guidance'].get('variable_length_fallback') for r in requests):
                        raise RuntimeError('unexpected slow-path graph miss')
                    result['steady_inference_median_ms']=float(np.median([
                        (r['inference_ended_monotonic']-r['inference_started_monotonic'])*1000 for r in requests[1:]]))
                    result['status']='completed'
                    base._write_json_atomic(child/'summary.json',result)
                    base._save_run_status(child,'completed')
                    summary['episodes'].append(dict(task_id=task_id,mode=mode,q=q,run=str(child),
                        outcome=episode['status'],steps=episode['steps'],real_seconds=episode['real_seconds'],
                        clean_run=clean['run'],audit_passed=True))
                    del requests,episode,result
                    mem(f'after_task{task_id}_{mode}_q{q}')
                    print('completion_progress='+str(len(summary['episodes']))+'/6',flush=True)
                    active_child=None
                env.close();env=None;cleanup(torch);mem(f'task{task_id}_environment_closed')
            # The same weights and final task's graph bank continue into power calibration.
            torch.cuda.synchronize();adapter.step_bank=None;adapter.expert=root_expert
            cleanup(torch);mem('bank_released_before_fixed_input_power')
            env=base._make_env(cls,dict(task,max_steps=10000));env._pb_lock_fixtures=True
            summary['status']='power_running';save()
            summary['power']=power_windows(root,adapter,env,task,stats,np,torch)
            summary['status']='completed';save()
            print('completion_finished='+str(root),flush=True)
        except BaseException as exc:
            if active_child is not None:
                path=active_child/'summary.json'
                child_result=json.loads(path.read_text()) if path.exists() else {}
                child_result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
                base._write_json_atomic(path,child_result)
            summary.update(status='invalid',error=f'{type(exc).__name__}: {exc}');save();raise
        finally:
            if env is not None:env.close()
            if adapter is not None:adapter.release_graph()
            save()
        return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-batch',type=Path,required=True)
    parser.add_argument('--resume',type=Path,help='continue only missing cases after an exceptional guarded exit')
    parser.add_argument('--guarded-worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.guarded_worker:
        command=[sys.executable,str(ROOT/'scripts/guarded_run.py'),'--',sys.executable,'-u',
            str(Path(__file__).resolve()),'--main-batch',str(args.main_batch),'--guarded-worker']
        if args.resume:command+=['--resume',str(args.resume)]
        return subprocess.call(command)
    parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    if not any(Path(os.fsdecode(p)).name=='guarded_run.py' for p in parent if p):
        parser.error('standard 768MiB guard required')
    run(args.main_batch,args.resume)


if __name__=='__main__':raise SystemExit(main())
