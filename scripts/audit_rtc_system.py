"""Independent saved-record system audit; does not call production queue/history helpers."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def decode_raw(raw,stats,profile=None):
    import numpy as np
    if profile=='lerobot_h50_quantiles_v044':
        x=np.asarray(raw,dtype=np.float32)[...,:7]
        lo=np.asarray(stats['actions']['q01'],dtype=np.float32)
        hi=np.asarray(stats['actions']['q99'],dtype=np.float32)
        delta=hi-lo
        delta=np.where(delta==0,np.float32(1e-8),delta)
        return (x+1)*delta/2+lo
    x=np.clip(np.asarray(raw,dtype=np.float32),-1,1)
    low=np.asarray(stats['actions']['q01'],dtype=np.float32)
    high=np.asarray(stats['actions']['q99'],dtype=np.float32)
    n=min(x.shape[-1],len(low))
    x[...,:n]=(x[...,:n]+1)/2*(high[:n]-low[:n]+1e-6)+low[:n]
    return x[...,:7]


def audit(run):
    import numpy as np
    cfg=json.loads((run/'run.json').read_text())['config_snapshot']
    synchronous=cfg.get('diagnostic_condition')=='sync_no_rtc' or cfg.get('execution_mode')=='diagnostic_sync_no_rtc'
    stats=json.loads(Path(cfg['norm_stats_path']).read_text())
    stats=stats.get('norm_stats',stats)
    q=cfg.get('candidate_q',[5])[0]
    requests=json.loads((run/f'rtc/q{q}_requests.json').read_text())
    events=[json.loads(s) for s in (run/'rtc/trace.jsonl').read_text().splitlines()]
    controls=[r for r in events if r['event']=='control_step']
    commits=[r for r in events if r['event']=='prediction_committed']
    by_step={r['step']:r for r in controls}
    by_id={r['prediction_id']:r for r in requests}
    committed={r['prediction_id']:r for r in commits}
    failures=[]
    count=0
    def check(name,ok,where=None):
        nonlocal count
        count+=1
        if not ok:failures.append(dict(check=name,where=where))
    eq=lambda a,b:np.array_equal(np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32))
    check('sequential_control_steps',[r['step'] for r in controls]==list(range(len(controls))))
    check('unique_request_ids',len(by_id)==len(requests))
    check('unique_commit_ids',len(committed)==len(commits))
    check('single_bootstrap',requests[0]['observation_step']==0 and commits[0]['commit_step']==0)
    if cfg.get('m2_single_reuse'):
        expert_ids=[r['prediction_id'] for r in requests if r.get('kv_recompute',{}).get('mode')=='expert_only']
        check('single_fast_request',expert_ids==[1])
        check('single_fast_committed',1 in committed)
        check('single_fast_actually_executed',any(r.get('source',{}).get('prediction_id')==1 for r in controls))
    history=[] if synchronous else [cfg['rtc_initial_delay_steps']]
    mode_histories=({mode:[cfg['rtc_initial_delays_by_mode'][mode]] for mode in cfg.get('m2_delay_modes', ('full','expert_only'))}
                    if cfg.get('m2_split_delay') else None)
    previous=None
    slacks=[]
    for request in requests:
        pid=request['prediction_id'];origin=request['observation_step']
        g=request['rtc_guidance'];solver=request['rtc_solver_inputs']
        if synchronous or not origin:
            check('bootstrap_without_guidance',g.get('guidance_enabled') is False
                  and solver['previous_raw'] is None and solver['delay']==solver['executed']==0
                  and g['delay_estimate_steps']==g['executed_steps']==0,pid)
            if synchronous:
                check('sync_without_delay_history','delay_history_steps' not in g,pid)
        check('finite_raw32',np.asarray(request['actions_raw']).shape==(cfg['H'],32)
              and np.isfinite(request['actions_raw']).all(),pid)
        check('finite_noise32',np.asarray(solver['noise']).shape==(cfg['H'],32)
              and np.isfinite(solver['noise']).all(),pid)
        check('action_postprocessing',eq(decode_raw(request['actions_raw'],stats,cfg.get('checkpoint_profile')),request['actions']),pid)
        scene=by_step[origin]['before']
        check('observation_raw_state_matches_control',eq(request['observation_raw_state'],scene['raw_state']),pid)
        low=np.asarray(stats['state']['q01'],dtype=np.float32)
        high=np.asarray(stats['state']['q99'],dtype=np.float32)
        if cfg.get('checkpoint_profile')=='lerobot_h50_quantiles_v044':
            delta=high-low
            delta=np.where(delta==0,np.float32(1e-8),delta)
            normalized=2*(np.asarray(scene['raw_state'],dtype=np.float32)-low)/delta-1
            normalized=np.pad(normalized,(0,32-len(normalized)))
        else:
            normalized=(np.asarray(scene['raw_state'],dtype=np.float32)-low)/(high-low+1e-6)*2-1
        check('state_normalization',eq(request['model_state'],normalized),pid)
        if cfg.get('m2_visual_alternation'):
            from perceptbudget.rtc_visual_alternation import scheduled_visual_mode,visual_geometry
            visual=request.get('context_refresh',{})
            expected_mode=scheduled_visual_mode(pid,cfg.get('m2_visual_fast_mode','wrist'),
                cfg.get('m2_visual_fast_enabled'),cfg.get('m2_full_per_fast',1))
            geometry=visual_geometry(expected_mode)
            check('fresh_visual_schedule',visual.get('mode')==expected_mode,pid)
            check('fresh_visual_inputs',visual.get('current_images_encoded') is True
                  and visual.get('current_state_encoded') == (expected_mode=='full' or not cfg.get('m2_fast_no_state',False)),pid)
            if cfg.get('m2_fast_no_state',False):
                check('fresh_visual_state_prompt',g.get('state_in_prompt') == (expected_mode=='full'),pid)
            check('fresh_visual_source',visual.get('source_observation_step')==origin
                  and visual.get('source_prediction_id')==pid,pid)
            check('fresh_visual_geometry',visual.get('num_views')==geometry['num_views']
                  and visual.get('visual_tokens')==geometry['visual_tokens']
                  and visual.get('camera_indices')==([1] if expected_mode=='wrist' else [0,1]),pid)
            check('fresh_visual_pool_factor',visual.get('vision_pool_factor',1)==geometry['vision_pool_factor'],pid)
            if expected_mode=='pool2_agent':
                check('fresh_visual_pooled_camera',visual.get('pooled_view')=='agent',pid)
            check('fresh_visual_no_stale_kv','kv_recompute' not in request,pid)
        if cfg.get('m2_expert_recompute'):
            k=request.get('kv_recompute',{})
            mode=k.get('mode')
            check('kv_mode_declared',mode in ('full','expert_only'),pid)
            src=by_id.get(k.get('source_prediction_id'),{})
            check('kv_source_is_full',src.get('kv_recompute',{}).get('mode')=='full',pid)
            check('kv_source_state',eq(k.get('source_raw_state'),src.get('observation_raw_state')),pid)
            check('kv_source_step',k.get('source_observation_step')==src.get('observation_step'),pid)
            check('kv_source_valid_length',g.get('valid_prefix_length')==src.get('rtc_guidance',{}).get('valid_prefix_length'),pid)
            check('kv_nominal_expiry',k.get('nominal_expiry_step')==src.get('observation_step',0)+cfg['H'],pid)
            check('kv_current_inputs_encoded',k.get('current_images_encoded')==(mode=='full')
                  and k.get('current_state_encoded')==(mode=='full'),pid)
            if mode=='expert_only':
                previous_request=by_id.get(pid-1,{})
                check('kv_reuse_enabled',bool(cfg.get('m2_reuse_enabled')),pid)
                check('kv_no_consecutive_reuse',previous_request.get('kv_recompute',{}).get('mode')=='full',pid)
        check('input_timestamp_order',request['observation_monotonic']<=request['input_ready_monotonic']
            <=request['inference_started_monotonic']<=request['inference_ended_monotonic']<=request['ready_monotonic'],pid)
        if origin:
            check('observation_after_previous_action',request['observation_monotonic']>=by_step[origin-1]['action_end_monotonic'],pid)
            old=by_id[previous['prediction_id']]
            s=origin-old['observation_step']
            if not synchronous:
                check('old_target_alignment',g.get('guidance_enabled') is True
                    and solver['executed']==s==g['executed_steps']
                    and eq(solver['previous_raw'],old['actions_raw'][s:]),pid)
            check('next_request_counter_rule',origin==max(old['observation_step']+q,previous['commit_step']),pid)
            check('no_overlapping_inference',request['inference_started_monotonic']>=old['inference_ended_monotonic'],pid)
            check('guidance_delay_matches_solver',solver['delay']==g['delay_estimate_steps'],pid)
            if synchronous:
                check('sync_request_interval',origin==old['observation_step']+q,pid)
            elif cfg.get('rtc_paper_schedule'):
                estimate=cfg.get('rtc_fixed_delay_estimate')
                expected_history=history
                if mode_histories is not None:
                    mode=request.get('context_refresh' if cfg.get('m2_visual_alternation') else 'kv_recompute',{}).get('mode')
                    check('planned_delay_mode_matches_execution',request.get('planned_context_mode' if cfg.get('m2_visual_alternation') else 'planned_kv_mode')==mode,pid)
                    check('delay_history_mode',g.get('delay_history_mode')==mode
                        and g.get('delay_estimate_mode')=='mode_history_max',pid)
                    expected_history=mode_histories.get(mode,[])
                check('independent_history_max' if estimate is None else 'fixed_guidance_estimate',
                      g['delay_estimate_steps']==(max(expected_history) if estimate is None else estimate)
                      and g['delay_history_steps']==expected_history,pid)
            else:
                check('controlled_d5',g['delay_estimate_steps']==cfg['controlled_delay_steps'],pid)
        if pid not in committed:
            check('only_terminal_request_uncommitted',request is requests[-1],pid)
            continue
        c=committed[pid];at=c['commit_step'];actual=at-origin
        check('commit_after_ready',c['commit_monotonic']>=request['ready_monotonic'],pid)
        check('exact_absolute_crop',c['expired_new_actions']==actual
              and c['new_prediction_remaining_actions']==cfg['H']-actual,pid)
        if synchronous:
            check('sync_zero_control_delay',actual==0,pid)
            if 'actual_delay_steps' in g:
                check('reported_actual_delay',g['actual_delay_steps']==actual,pid)
        else:
            check('reported_actual_delay',g['actual_delay_steps']==actual,pid)
        if origin and not synchronous:
            for step in range(origin,at):
                source=by_step[step]['source']
                old=by_id[previous['prediction_id']]
                exhausted=step>=old['observation_step']+cfg['H']
                check('old_actions_during_inference',source is None if exhausted else
                      source is not None and source['prediction_id']==previous['prediction_id'],(pid,step))
            slacks.append(g['delay_estimate_steps']-actual)
            history=(history+[actual])[-cfg.get('rtc_delay_buffer_size',10):]
            if mode_histories is not None:
                mode=request['context_refresh' if cfg.get('m2_visual_alternation') else 'kv_recompute']['mode']
                mode_histories[mode]=(mode_histories[mode]+[actual])[-cfg.get('rtc_delay_buffer_size',10):]
        previous=c
    underflows=[]
    for step,row in by_step.items():
        candidates=[c for c in commits if c['commit_step']<=step]
        active=max(candidates,key=lambda c:c['commit_step'])
        request=by_id[active['prediction_id']]
        offset=step-request['observation_step']
        check('control_timestamp_order',row['tick_monotonic']<=row['action_start_monotonic']<=row['action_end_monotonic'],step)
        if step:
            check('sequential_action_timestamps',by_step[step-1]['action_end_monotonic']<=row['tick_monotonic'],step)
        check('commit_before_action',active['commit_monotonic']<=row['action_start_monotonic'],step)
        if offset>=cfg['H']:
            underflows.append(step)
            check('exhausted_chunk_zero_underflow',row['source'] is None
                  and eq(row['action'],np.zeros(7)) and row['applied_arm_gain']==1
                  and not row.get('late_hold',False),step)
            continue
        check('action_is_latest_committed_chunk',row['source'] is not None
            and row['source']['prediction_id']==active['prediction_id']
            and row['source']['offset']==offset
            and row['source']['origin_step']==request['observation_step']
            and row['applied_arm_gain']==1 and eq(row['action'],request['actions'][offset]),step)
        if cfg.get('m2_expert_recompute'):
            check('kv_execution_before_original_expiry',
                step<request.get('kv_recompute',{}).get('nominal_expiry_step',-1),step)
    summary_path=run/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        episode=summary.get('episode',summary)
        if 'underflow_steps' in episode:
            check('reported_underflow_steps',episode['underflow_steps']==underflows)
        if 'initial_raw_state' in episode:
            check('initial_raw_state',eq(episode['initial_raw_state'],controls[0]['before']['raw_state']))
        if 'initial_sim_state_sha256' in episode:
            import hashlib
            digest=hashlib.sha256(np.asarray(controls[0]['before']['sim_state'],dtype=np.float64).tobytes()).hexdigest()
            check('initial_sim_state_sha256',episode['initial_sim_state_sha256']==digest)
    controlled=cfg.get('controlled_delay_steps')
    check('no_controlled_wait_in_actual',controlled is not None or all('controlled_wait_seconds' not in r for r in requests))
    return dict(run=str(run),passed=not failures,checks=count,failures=failures,
        controls=len(controls),requests=len(requests),commits=len(commits),
        estimate_minus_actual=slacks,
        audit_mode='sync_no_rtc' if synchronous else 'async_rtc',
        scope_id='m1_sync_no_rtc_records_v1' if synchronous else 'm1_async_rtc_records_v1',
        supply_quality=dict(passed=not underflows,underflow_steps=underflows,
                            classification='supply_quality_issue' if underflows else 'no_underflow'),
        scope='recorded state/normalization/inference timestamps/Algorithm1 counters/crop/physical actions; not all-input or full-model proof')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=[audit(run) for run in args.runs]
    from scripts import eval_q as base
    base._write_json_atomic(args.output,result)
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if all(r['passed'] for r in result) else 1)
