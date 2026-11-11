"""Independent saved-record system audit; does not call production queue/history helpers."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def latest_commit_at_step(commits, step):
    # Critical waiting can finish several predictions before the same env.step.
    # A step number alone does not order these submissions.
    return max((c for c in commits if c['commit_step']<=step),
               key=lambda c:(c['commit_step'],c['commit_monotonic'],c['prediction_id']))


def phase_active_at_decision(phase,timing,events):
    # A copied clock-recovery plan may be submitted on a later control step.
    steps=[e['step'] for e in events if e['event']=='m3_decision'
           and e.get('decision_monotonic')==timing.get('decision_monotonic')]
    return bool(steps) and phase['completed_step']<=min(steps)<phase['stop_step']


def phase_history_at_decision(commits,by_id,step,now):
    """Rebuild every still-active typed response, without the runtime helper."""
    result=[]
    for commit in commits:
        if commit['commit_monotonic']>now or commit['commit_step']>step:continue
        request=by_id[commit['prediction_id']]
        response=request.get('m3_timing',{}).get('refresh_phase_response')
        if response and commit['commit_step']<response['stop_step'] and step<response['stop_step']:
            result.append(dict(response,completed_prediction_id=commit['prediction_id'],
                completed_step=commit['commit_step']))
    return result


def point_history_at_decision(commits, by_id, step, now, cfg):
    """Rebuild completed event responses independently of the controller."""
    from perceptbudget.action_events import EventConfig, detect_events
    config = EventConfig(**cfg['m3_events'])
    points = []
    for commit in commits:
        if commit['commit_monotonic'] > now or commit['commit_step'] > step:
            continue
        request = by_id[commit['prediction_id']]
        timing = request.get('m3_timing', {})
        target = timing.get('critical_step')
        if target is None or commit['commit_step'] > target:
            continue
        kinds = timing.get('candidate_event_kinds') or []
        if cfg.get('m3_event_coverage_mode', 'event_only') == 'event_only':
            if timing.get('reason') != 'action_event' or (
                    timing.get('arrival_lateness_seconds') and cfg.get('m3_feedback_policy') != 'overlap_pacing_v2'):
                continue
        else:
            if commit['commit_step'] < request['observation_step']:
                continue
            events = detect_events(request['actions'], config,
                start_offset=max(0, commit['commit_step']-request['observation_step']))
            kinds = [kind for kind in kinds if not (cfg.get('m3_ignore_turn') and kind == 'turn')
                and any(e['kind'] == kind and abs(request['observation_step']+e['offset']-target) <= config.match_steps
                        for e in events)]
        points.extend(dict(kind=kind, target_step=target) for kind in kinds
            if target+config.match_steps >= step)
    return points


def turn_identity_from_record(event,phases,by_id,config):
    """Independent reconstruction from saved Full outputs, not runtime chunks."""
    valid=[p for p in phases if p['start_step']<=event['target_step']<p['stop_step']
        and event['kind'] in p['kinds']]
    phase=max(valid,key=lambda p:p['completed_prediction_id'])
    reference=phase['completed_prediction_id'];anchor=max(phase['target_step'],phase['completed_step'])
    def vector(pid,step):
        r=by_id[pid];i=step-r['observation_step'];n=config.get('confirm_steps',2)
        values=r['actions'][i:i+n] if i>=0 else []
        if len(values)!=n:return None
        result=[sum(a[k] for a in values)/n for k in range(3)]
        return result if sum(v*v for v in result)>=config.get('minimum_motion',.1)**2 else None
    a=vector(reference,anchor);b=vector(event['prediction_id'],event['target_step']);angle=None
    if a is not None and b is not None:
        dot=sum(x*y for x,y in zip(a,b));length=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
        angle=math.degrees(math.acos(max(-1.,min(1.,dot/length))))
    limit=config.get('turn_degrees',60.)
    same=angle is not None and angle+1e-9<limit
    return dict(kind=event['kind'],prediction_id=event['prediction_id'],target_step=event['target_step'],
        reference_prediction_id=reference,reference_action_step=anchor,
        reference_direction=a,new_direction=b,angle_degrees=angle,threshold_degrees=limit,
        same_event=same,reason='direction_unavailable' if angle is None else 'same_turn_direction' if same else 'new_turn_direction')


def ordinary_turn_from_record(event,source,proof,controls,config,phases,approach_delta=None):
    """Reconstruct local command evidence independently from saved records."""
    at=event['target_step'];n=config.get('confirm_steps',2);begin=proof['source_step']
    def vector(first):
        values=[]
        for step in range(first,first+n):
            index=step-source['observation_step']
            a=(source['actions'][index] if step>=begin and 0<=index<len(source['actions'])
                else controls.get(step,{}).get('action'))
            if a is None:return None
            values.append(a)
        v=[sum(a[k] for a in values)/n for k in range(3)]
        return v if math.sqrt(sum(x*x for x in v))>=config.get('minimum_motion',.1) else None
    a=vector(at-n);b=vector(at);angle=None
    if a is not None and b is not None:
        angle=math.degrees(math.acos(max(-1.,min(1.,sum(x*y for x,y in zip(a,b))/
            math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))))))
    intervals=[[p['start_step'],p['stop_step']] for p in phases if p['start_step']<=at<p['stop_step']]
    cues=proof['result']['events'];gap=config.get('match_steps',5)
    grip=min((abs(e['target_step']-at) for e in cues if e['kind'].startswith('gripper_')),default=None)
    slow=min((abs(e['target_step']-at) for e in cues if e['kind']=='slowdown'),default=None)
    approach=None
    if approach_delta is not None:
        down_before=-a[2]/math.sqrt(sum(v*v for v in a)) if a else None
        down_after=-b[2]/math.sqrt(sum(v*v for v in b)) if b else None
        index=at-source['observation_step'];window=source['actions'][index:index+n] if index>=0 else []
        opened=len(window)==n and all(x[6]<=-config.get('gripper_threshold',.2) for x in window)
        required=bool(opened and down_before is not None and down_after is not None
            and down_after>=.5 and down_after-down_before+1e-9>=approach_delta)
        approach=dict(minimum_increase=approach_delta,down_before=down_before,down_after=down_after,
                      gripper_open=opened,requires_full=required)
    reason=('downward_approach' if approach and approach['requires_full'] else
        'protected_motion_phase' if intervals else 'near_gripper' if grip is not None and grip<=gap else
        'near_slowdown' if slow is not None and slow<=gap else 'direction_unavailable' if angle is None else
        'abrupt_turn' if angle+1e-9>=config.get('turn_degrees',60.) else 'ordinary_bend')
    return dict(kind='turn',prediction_id=event['prediction_id'],target_step=at,source_step=begin,
        before_direction=a,after_direction=b,local_angle_degrees=angle,protected_intervals=intervals,
        nearest_gripper_steps=grip,nearest_slowdown_steps=slow,ordinary=reason=='ordinary_bend',reason=reason,
        **({'full_approach':approach} if approach is not None else {}))


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


def phase_completion_matches(request,commit):
    if request.get('m3_fast_choice',{}).get('mode')!='Full':return False
    done=request.get('m3_timing',{}).get('key_phase_response_completed')
    if commit is None:return done is None  # terminal drain, not a fulfilled response
    return done==dict(prediction_id=request['prediction_id'],commit_step=commit['commit_step'],commit_monotonic=commit['commit_monotonic'])


def audit_service_formula(formula,decision,costs,previous_mode):
    """Independent V4 reconstruction; Continue is not inference energy."""
    errors=[]
    def test(ok,name):
        if not ok:errors.append(name)
    def close(a,b):return abs(a-b)<1e-7
    now=decision['decision_monotonic'];end=decision['deadline'];eta=decision['margin_seconds']
    period=decision['estimated_control_period_seconds'];profile=decision['current_profile']
    cap=formula.get('latest_start_cap');rows=formula['candidates']
    test(len(rows)==2 and [r['mode'] for r in rows]==['Full','Fast'],'candidate_set')
    require_full=not formula['observation_step'] or decision.get('fast_full_only',False) or decision['reason'] in ('action_event','key_phase')
    for r in rows:
        mode=r['mode'];entry='gpu_emc' if mode=='Fast' else profile
        cell=max((c for c in costs['service'][mode].values() if c['from_profile']==entry),key=lambda c:c['energy_j'])
        steps=math.ceil((r['service_seconds']+eta)/period-1e-9) if mode=='Fast' else 0
        bound=cap-steps*period if cap is not None and mode=='Fast' else cap
        raw=end-r['service_seconds']-eta;latest=min(raw,bound) if bound is not None else raw
        wait=max(0.,latest-now);down=costs.get('transitions',{}).get('down',{})
        down_time=down.get('max_readback_seconds',down.get('seconds',0.))
        low_ok=mode!='Fast' or profile=='gpu_emc' or wait>=decision.get('minimum_low_idle_seconds',.2)+down_time
        phase_allowed=True
        if mode=='Fast' and decision.get('fast_phase_guard') is not None:
            guard=decision['fast_phase_guard']
            begin=formula['observation_step']+math.floor(max(0.,latest-now)/period+1e-9)
            stop=begin+guard['horizon']
            overlaps=[p for p in guard['intervals'] if p['start_step']<stop and begin<p['stop_step']]
            phase_allowed=not overlaps
            gate=r.get('key_phase_gate',{})
            test(gate.get('start_step')==begin and gate.get('stop_step')==stop
                 and gate.get('prediction_id')==guard['prediction_id'] and gate.get('overlaps')==overlaps
                 and gate.get('allowed')==phase_allowed,'key_phase_forecast_overlap')
        allowed=mode=='Full' or not require_full and previous_mode!='Fast' and low_ok and phase_allowed
        test(r['quality_allowed']==allowed and r['low_continue_admissible']==low_ok,'quality_filter')
        test(r['entry_profile']==entry and r['compute_profile']=='baseline','service_entry_and_compute_state')
        test(close(r['latest_start'],latest) and close(r['formula_latest_start'],raw) and r['advance_steps']==steps,'fast_advance_once')
        test(r['feasible']==(latest>=now-1e-9),'feasibility_filter')
        test(close(r['energy_j'],cell['energy_j']) and close(r['service_energy_j'],cell['energy_j']) and r['energy_scope']=='complete_service_only','service_energy_only')
        test(r['continue_before']['activity']==r['continue_after']['activity']=='Continue' and close(r['continue_before']['seconds'],wait)
             and r['continue_after'].get('forecast') is False,'continue_separate')
        test(r['additional_transition_energy_j']==0.,'recovery_once')
    eligible=[r for r in rows if r['quality_allowed'] and r['feasible']]
    chosen=min(eligible,key=lambda r:(r['service_energy_j'],r['mode']!='Full')) if eligible else rows[0]
    test(formula['mode']==chosen['mode'] and formula['selected']==chosen,'chosen_service')
    test(formula['reason']==('feasible_minimum_service_energy' if eligible else 'deadline_unreachable'),'fallback')
    test(close(decision['service_seconds'],chosen['service_seconds']) and close(decision['latest_start'],chosen['latest_start']),'selected_budget_applied')
    return errors


def audit_fast_formula(formula, decision, costs, previous_mode):
    """Independently recompute the recorded two-mode deadline/energy ledger."""
    if formula.get('policy')=='m3_service_ledger_fast_advance_v4':
        return audit_service_formula(formula,decision,costs,previous_mode)
    failures=[]
    def test(ok,name):
        if not ok:failures.append(name)
    def close(a,b):return abs(a-b)<1e-7
    def hold(profile,age,length):
        edges=(0.,.2,.5,1.,float('inf'))
        return sum(max(0.,min(age+length,b)-max(age,a))*costs['waits'][profile][i]
                   for i,(a,b) in enumerate(zip(edges,edges[1:])))
    now=decision['decision_monotonic'];end=decision['deadline'];eta=decision['margin_seconds']
    profile=decision['current_profile'];cap=formula.get('latest_start_cap')
    rows=formula.get('candidates',[])
    test(len(rows)==2 and {r['mode'] for r in rows}=={'Full','Fast'},'candidate_set')
    critical=decision.get('critical_step')
    # decision stores the chunk's current observation step through the caller.
    step=formula['observation_step']
    critical_time=now+(critical-step)*decision['estimated_control_period_seconds'] if critical is not None else None
    require_full=(decision.get('fast_full_only',False) or not step or decision['reason']=='action_event' or critical_time is not None and critical_time<=end+1e-9)
    for r in rows:
        mode=r['mode'];entry=[c for c in costs['service'][mode].values() if c['from_profile']==profile]
        cell=max(entry,key=lambda c:c['energy_j'])
        latest=end-r['service_seconds']-eta
        if cap is not None:latest=min(latest,cap)
        base_latest=latest
        reserve_steps=0
        if mode=='Fast' and decision.get('fast_reserve_full',False):
            reserve_steps=math.ceil((decision['fast_full_reserve_service_seconds']+eta)/decision['estimated_control_period_seconds']-1e-9)
            latest-=reserve_steps*decision['estimated_control_period_seconds']
        if decision.get('fast_reserve_full',False):
            test(r.get('full_reserve_steps')==reserve_steps and close(r.get('base_latest_start',float('inf')),base_latest),'full_reservation')
        start=max(now,latest);before=start-now;after=max(0.,end-start-cell['mean_seconds'])
        tail_energy=hold('baseline',0.,after)
        tail_path='high_hold'
        if costs.get('continue_energy_model')=='measured_roundtrip_v1':
            down,up=(costs['transitions'][k] for k in ('down','up'))
            low_seconds=after-down['seconds']-up['seconds']
            if low_seconds>=decision.get('minimum_low_idle_seconds',.2):
                transition=0.
                for t,age in ((down,0.),(up,down['seconds']+low_seconds)):
                    adjustment=max(0.,hold(t['source'],age,t['seconds'])-hold(t['source'],t.get('reference_quiet_age_seconds',0.),t['seconds']))
                    transition+=t['energy_j']+adjustment
                low_energy=transition+hold('gpu_emc',down['seconds'],low_seconds)
                if low_energy<tail_energy:tail_energy,tail_path=low_energy,'low_hold_roundtrip'
            plan=r.get('continue_plan',{})
            test(r.get('tail_activity')=='Continue' and plan.get('activity')=='Continue' and plan.get('path')==tail_path
                 and close(plan.get('energy_j',float('inf')),tail_energy),'continue_tail_plan')
        energy=hold(profile,formula['quiet_age_seconds'],before)+cell['energy_j']+tail_energy
        test(close(r['latest_start'],latest) and close(r['formula_latest_start'],end-r['service_seconds']-eta),'mode_latest_start')
        test(close(r['window_end'],end) and close(formula['window_start'],now)
             and close(r['prepare_start'],start) and close(r['wait_seconds'],before)
             and close(r['tail_seconds'],after) and close(r['mean_service_seconds'],cell['mean_seconds']),'same_window')
        test(close(r['energy_j'],energy) and close(r['service_energy_j'],cell['energy_j'])
             and close(r['wait_energy_j'],hold(profile,formula['quiet_age_seconds'],before))
             and close(r['tail_energy_j'],tail_energy)
             and r['additional_transition_energy_j']==0.,'absolute_energy_once')
        quality=mode=='Full' or not require_full and previous_mode!='Fast'
        test(r['quality_allowed']==quality,'quality_filter')
        test(r['feasible']==(latest>=now-1e-9 and start+cell['mean_seconds']<=end+1e-9),'feasibility_filter')
    allowed=[r for r in rows if r['quality_allowed'] and r['feasible']]
    chosen=(min(allowed,key=lambda r:(r['energy_j'],r['mode']!='Full')) if allowed else
            next(r for r in rows if r['mode']=='Full') if decision.get('fast_reserve_full',False) else
            min((r for r in rows if r['quality_allowed']),key=lambda r:r['service_seconds']))
    test(formula['mode']==chosen['mode'] and formula['selected']==chosen,'chosen_minimum')
    test(formula['reason']==('feasible_minimum_energy' if allowed else 'deadline_unreachable'),'missed_deadline_recorded')
    test(close(decision['service_seconds'],chosen['service_seconds']) and close(decision['latest_start'],chosen['latest_start']),'selected_budget_applied')
    return failures


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
    m3=cfg.get('m3_scheduler_mode') in ('deadline_high','deadline_dvfs')
    age_limit=cfg['m3_timing']['maximum_observation_age_seconds'] if m3 else None
    decisions={r['step']:r for r in events if r['event']=='m3_decision'}
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
    visual_samples=[e for e in events if e.get('event')=='visual_monitor']
    cnn_settings=cfg.get('visual_monitor',{}).get('kind')=='cnn_occupancy'
    for sample in visual_samples:
        control=by_step.get(sample['step']-1,{})
        decision=decisions.get(sample.get('decision_step'),{})
        acquired=sample.get('acquisition',{})
        period=cfg.get('visual_monitor',{}).get('interval',5)
        scheduled=sample['step']%period==0
        if cnn_settings:
            earlier=[s for s in visual_samples if s['step']<sample['step']
                     and s.get('close_step')==sample.get('close_step')]
            prior=earlier[-1] if earlier else None
            scheduled=(sample['step']==sample.get('close_step') if prior is None
                       else sample['step']-prior['step']>=period)
            check('cnn_sample_closed_cycle',scheduled and sample['eligible']
                and sample.get('input_kind')=='raw_wrist_rgb' and sample.get('device')=='cpu'
                and sample.get('inference_active')==control.get('inference_active')
                and sample.get('inference_active_at_action')==control.get('inference_active_at_action')
                and control.get('action',[0]*7)[6]>=.2,sample.get('sample_id'))
            probabilities=np.asarray(sample.get('probabilities',[]))
            valid=(probabilities.shape==(2,) and np.all(np.isfinite(probabilities))
                and np.all(probabilities>=0) and np.all(probabilities<=1)
                and abs(float(probabilities.sum())-1)<1e-5)
            prediction=(('held','empty')[int(probabilities.argmax())]
                if valid and probabilities.max()>=sample['confidence_threshold'] else 'unknown')
            check('cnn_confidence_rejection',valid and sample.get('predicted_state')==prediction,
                sample.get('sample_id'))
            carrying=sample.get('carrying_window',False)
            bad=(prior.get('confirmation_count',0) if prior else 0)+1 if carrying and prediction=='empty' else 0
            previous_alarm=any(s.get('alarm') for s in earlier)
            check('cnn_temporal_alarm',sample.get('confirmation_count')==bad
                and sample['alarm']==bool(not previous_alarm and bad>=cfg['visual_monitor']['confirmation']),
                sample.get('sample_id'))
            close_step=sample.get('close_step',0)
            active=[r for r in controls if close_step<=r['step']+1<=sample['step']]
            check('cnn_carry_window_from_arm_not_objects',bool(active)
                and carrying==any(r['after']['raw_state'][2]-sample['close_z_m']>=.02 for r in active)
                and not any(r['action'][6]<=-.2 for r in active),sample.get('sample_id'))
        elif not scheduled:
            at=sample['step']-1
            preceding=[by_step[i] for i in sorted(by_step) if i<=at]
            opened_at=max((r['step'] for r in preceding if r['action'][6]<=-.2),default=-1)
            close=next((r for r in preceding if r['step']>opened_at and r['action'][6]>=.2),None)
            recent_closed=len(preceding)>=2 and all(r['action'][6]>=.2 for r in preceding[-2:])
            opened=[abs(r['after']['raw_state'][6]-r['after']['raw_state'][7])
                for r in preceding if close and r['step']<close['step'] and r['action'][6]<=-.2]
            width=abs(control['after']['raw_state'][6]-control['after']['raw_state'][7])
            proof=sample.get('grasp_check',{}).get('state_candidate') or {}
            scheduled=(cfg.get('visual_monitor',{}).get('grasp_check') is True and recent_closed
                and bool(opened) and width<=.05*max(opened)
                and control['after']['raw_state'][2]-close['before']['raw_state'][2]>=.02
                and proof.get('close_step')==close['step']+1
                and sample.get('sampling_trigger')=='grasp_check')
            forced_count=sum(e.get('event')=='visual_monitor' and e.get('sampling_trigger')=='grasp_check'
                and (e.get('grasp_check',{}).get('state_candidate') or {}).get('close_step')==proof.get('close_step')
                for e in events)
            check('visual_grasp_extra_camera_budget',forced_count<=2,sample.get('sample_id'))
        if not cnn_settings:
            check('visual_sample_real_continue',bool(cfg.get('visual_monitor'))
                and scheduled
                and sample['eligible'] and not control.get('inference_active_at_action',True)
                and control.get('action',[0]*7)[6]>=.2 and not decision.get('start_now',True)
                and decision.get('wait_seconds',0)>0,sample.get('sample_id'))
        check('visual_camera_after_control_without_physics_step',
            acquired.get('sim_time')==control.get('after',{}).get('sim_time')
            and acquired.get('physics_advanced') is False
            and acquired.get('policy_observation_replaced') is False
            and acquired.get('start_monotonic',-1)>=control.get('action_end_monotonic',float('inf'))
            and acquired.get('stop_monotonic',-1)>=acquired.get('start_monotonic',float('inf')),
            sample.get('sample_id'))
    group=cfg.get('m3_formal_group')
    if group is not None:
        # Historical runs before the explicit selector used variance_only.
        formal_signal=cfg.get('m3_formal_signal','variance_only')
        active=group!='A_fixed_q5'
        expected_variance=active and formal_signal!='events_only'
        expected_events=active and formal_signal!='variance_only'
        check('formal_group_declared',group in ('A_fixed_q5','B_high','C_independent','D_joint'))
        check('formal_trigger_declared',formal_signal in ('events_only','variance_only','events_and_variance'))
        check('formal_same_model_settings',cfg['H']==50 and cfg['N']==5 and cfg['precision']=='bfloat16'
              and cfg['state_enabled'] is True and q==5)
        check('formal_signal_switches',cfg.get('m3_dvac_enabled')==expected_variance
              and cfg.get('m3_action_events_enabled')==expected_events and cfg.get('m3_merge_slowdown_steps') is None)
        check('formal_request_watchdog',cfg.get('m3_request_timeout_seconds')==3. and bool(cfg.get('m3_watchdog_path')))
    check('sequential_control_steps',[r['step'] for r in controls]==list(range(len(controls))))
    if cfg.get('m3_ordinary_fast_policy') in ('ordinary_fast_feedback_buffer_v2','ordinary_fast_feedback_buffer_v3','critical_fast_feedback_buffer_v1','full_fast_alternating_state_v1','full_fast_alternating_no_state_v1'):
        check('fast_request_only_capture_enabled',cfg.get('m4_lean_continue') is True
              and cfg.get('m4_request_only_images') is True)
        for row in controls:
            active=row.get('inference_active_at_action')
            source=by_id.get(row.get('inference_prediction_id_at_action')) if active else None
            mode=source.get('m3_fast_choice',{}).get('mode','Full') if source else None
            check('fast_inflight_mode_provenance',row.get('inference_mode_at_action')==mode
                  and (not active or source is not None),row['step'])
            check('fast_no_unused_inflight_images',row.get('camera_render_skipped') is True,row['step'])
    check('unique_request_ids',len(by_id)==len(requests))
    check('unique_commit_ids',len(committed)==len(commits))
    check('single_bootstrap',requests[0]['observation_step']==0 and commits[0]['commit_step']==0)
    if cfg.get('m2_single_reuse'):
        expert_ids=[r['prediction_id'] for r in requests if r.get('kv_recompute',{}).get('mode')=='expert_only']
        check('single_fast_request',expert_ids==[1])
        check('single_fast_committed',1 in committed)
        check('single_fast_actually_executed',any(r.get('source',{}).get('prediction_id')==1 for r in controls))
    history=[] if synchronous else [cfg['rtc_initial_delay_steps']]
    frequency_histories=({'baseline':[cfg['rtc_initial_delay_steps']],
        'gpu_emc':[cfg['m3_low_full_initial_delay']]} if cfg.get('m3_low_full')
        and cfg.get('m3_low_full_policy')!='original_m3_slot_v3' and not cfg.get('m3_ordinary_fast_context') else None)
    mode_histories=({mode:[cfg['rtc_initial_delays_by_mode'][mode]] for mode in cfg.get('m2_delay_modes', ('full','expert_only'))}
                    if cfg.get('m2_split_delay') else None)
    if (cfg.get('m4_enabled') or cfg.get('m3_ordinary_fast_context')) and mode_histories is not None:
        mode_histories={key:[value] for key,value in cfg['rtc_initial_delays_by_compute'].items()}
    service_history=None
    if cfg.get('m3_fast_policy') in ('m3_two_mode_deadline_energy_v2','m3_full_fast_execution_window_v3','m3_service_ledger_fast_advance_v4'):
        service_history={'Full':{h:list(v) for h,v in cfg['m3_service_samples'].items()},
            'Fast':{h:list(cfg['m3_fast_cost_model']['service']['Fast'][h]['samples_seconds'])
                    for h in ('hot','short','long')}}
        if cfg.get('m3_fast_selective416'):
            probability=cfg['m3_timing']['service_quantile']
            def initial_q(values):
                values=sorted(values);return values[math.ceil(len(values)*probability)-1]
            recovery_q=max(initial_q(v) for v in service_history['Full'].values())
            if cfg.get('m3_recovery_samples'):recovery_q=max(recovery_q,initial_q(cfg['m3_recovery_samples']))
            overhead=max(0.,recovery_q-initial_q(service_history['Full']['hot']))
            service_history['Fast']={h:[v+(overhead if h!='hot' else 0.) for v in cfg['m3_fast_service_calibration_seconds']] for h in ('hot','short','long')}
        service_history['Full']['recovery']=list(cfg.get('m3_recovery_samples',[]))
        service_history['Fast']['recovery']=[]
    previous=None
    slacks=[]
    from perceptbudget.critical_wait import PhaseWait, wait_target, rtc_prefix_plan
    from perceptbudget.action_events import EventConfig
    wait_guard=PhaseWait()
    critical_waits={e['prediction_id']:e for e in events if e['event']=='critical_wait'}
    visual_waits={e['prediction_id']:e for e in events if e['event']=='visual_monitor_wait'}
    for request in requests:
        pid=request['prediction_id'];origin=request['observation_step']
        if cfg.get('safe_failure_probe'):
            safe=request.get('safe_failure_probe',{})
            earlier=[r.get('safe_failure_probe',{}) for r in requests if r['prediction_id']<pid]
            cumulative=sum(r.get('probability',0.) for r in earlier)+safe.get('probability',0.)
            threshold=safe.get('threshold')
            exceeded=threshold is not None and safe.get('cumulative_score',-1.)>=threshold
            check('safe_cumulative_score',safe.get('request_index')==pid
                and 0.<=safe.get('probability',-1.)<=1.
                and abs(safe.get('cumulative_score',-1.)-cumulative)<1e-4
                and safe.get('expert_forwards')==cfg['N'],pid)
            check('safe_first_alarm_only',safe.get('exceeded')==exceeded
                and safe.get('new_alarm')==(exceeded and not any(r.get('exceeded') for r in earlier)),pid)
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
            if cfg.get('m3_ordinary_fast_context'):
                choice=request.get('m3_fast_choice',{})
                declared=choice.get('mode');decision=request.get('m3_fast_decision',{})
                expected='Full'
                if origin and cfg.get('m3_ordinary_fast'):
                    f=decision['m3_fast_formula'];profile=decision['compute_profile']
                    c=cfg['ordinary_fast_costs']['cells'][profile]
                    prior=by_id[pid-1].get('m3_fast_choice',{}).get('mode','Full')
                    from perceptbudget.action_events import EventConfig,detect_events
                    from perceptbudget.key_phases import absolute_intervals
                    source=by_id[previous['prediction_id']];ec=EventConfig(**cfg['m3_events'])
                    phases=absolute_intervals(source['actions'],source['observation_step'],ec)
                    active=[p for p in phases if p['start_step']<=origin<p['stop_step']]
                    forecast=detect_events(source['actions'],ec,start_offset=max(0,previous['commit_step']-source['observation_step']))
                    near=[dict(e,target_step=source['observation_step']+e['offset']) for e in forecast
                        if source['observation_step']+e['offset']>=origin and
                        (source['observation_step']+e['offset']-origin)*decision['estimated_control_period_seconds']<=decision['service_seconds']+decision['margin_seconds']]
                    ordinary=decision['reason']=='buffer' and not active and not near
                    alternating=cfg.get('m3_ordinary_fast_policy') in ('full_fast_alternating_state_v1','full_fast_alternating_no_state_v1')
                    if alternating:
                        check('alternating_state_contract',f['policy']==cfg['m3_ordinary_fast_policy']
                            and cfg.get('m3_fast_alternation') is True and cfg.get('m2_fast_no_state')==
                            (cfg['m3_ordinary_fast_policy']=='full_fast_alternating_no_state_v1'),pid)
                    if cfg.get('m3_ordinary_fast_policy') in ('ordinary_fast_feedback_buffer_v2','ordinary_fast_feedback_buffer_v3','critical_fast_feedback_buffer_v1'):
                        raw=decision['critical_forecast']['result']['events']
                        near=[dict(e,prediction_id=previous['prediction_id']) for e in raw
                              if e['kind'].startswith('gripper_') and abs(e['target_step']-origin)<=ec.match_steps]
                        ordinary=decision['reason']=='buffer' and not near
                        if cfg['m3_ordinary_fast_policy'] in ('ordinary_fast_feedback_buffer_v3','critical_fast_feedback_buffer_v1'):
                            protected=[p for p in active if any(a['kind'].startswith('gripper_') for a in p.get('anchors',[]))]
                            ordinary=ordinary and not protected
                            check('ordinary_gripper_phase_protection',f.get('protected_gripper_phases')==protected,pid)
                        check('ordinary_feedback_policy',f['policy']==cfg['m3_ordinary_fast_policy']
                              and cfg.get('m3_feedback_policy')=='overlap_pacing_v2'
                              and f.get('gripper_margin_steps')==ec.match_steps,pid)
                    # Alternation deliberately ignores ordinary-stage admission.
                    # Reconstruct its order below, independently of that unused
                    # legacy diagnostic (which can name a different old chunk).
                    if not alternating:
                        check('ordinary_phase_gate',f['ordinary']==ordinary and f['active_phases']==active,pid)
                    check('ordinary_previous_mode',f['previous_mode']==prior,pid)
                    schedule=f.get('selected_schedule') if decision.get('fast_start_budget') else f.get('full_schedule')
                    check('ordinary_selected_schedule_recorded',bool(schedule)
                        and all(decision[k]==v for k,v in schedule.items()),pid)
                    if decision.get('fast_start_budget'):
                        check('ordinary_fast_budget_opt_in',cfg.get('m3_fast_own_budget') is True
                            and decision['reason']=='action_event'
                            and f.get('timing_basis')=='critical_fast_own_budget_v1',pid)
                    window=max(0.,decision['deadline']-decision['margin_seconds']-max(decision['decision_monotonic'],decision['latest_start']))
                    energy=[]
                    for proposal,mode in zip(f['candidates'],('Full','Fast')):
                        cell=c['modes'][mode];tail=max(0.,window-cell['mean_seconds'])
                        total=cell['mean_energy_j']+tail*c['hold_power_w'];energy.append(total)
                        check('ordinary_same_window_energy',proposal['mode']==mode and abs(proposal['service_energy_j']-cell['mean_energy_j'])<1e-8
                            and abs(proposal['continue_energy_j']-tail*c['hold_power_w'])<1e-8
                            and abs(proposal['comparable_window_energy_j']-total)<1e-8,pid)
                    estimate=max(mode_histories[cfg.get('m3_ordinary_fast_visual_mode','selective320')+':'+profile])
                    check('ordinary_fast_delay_history',f['fast_delay_steps']==estimate,pid)
                    allowed=(ordinary and prior!='Fast' and not decision['deadline_already_unreachable']
                        and estimate<=decision['rtc_delay_steps'] and c['modes']['Fast']['mean_seconds']<=c['modes']['Full']['mean_seconds'])
                    expected='Fast' if allowed and energy[1]<energy[0] else 'Full'
                    if cfg.get('m3_ordinary_fast_policy')=='critical_fast_feedback_buffer_v1':
                        expected='Fast' if (not ordinary and prior!='Fast'
                            and estimate<=decision['rtc_delay_steps']
                            and c['modes']['Fast']['mean_seconds']<=c['modes']['Full']['mean_seconds']) else 'Full'
                    if alternating:
                        ratio=cfg.get('m3_fast_full_per_fast',1)
                        streak=0
                        for earlier in range(pid-1,-1,-1):
                            if by_id[earlier].get('m3_fast_choice',{}).get('mode','Full')!='Full':break
                            streak+=1
                        expected='Fast' if streak>=ratio else 'Full'
                        check('alternating_cadence',ratio in (1,2) and f.get('full_per_fast',1)==ratio
                            and f.get('full_streak',streak)==streak,pid)
                    if (cfg.get('m3_full_approach_delta') is not None and decision.get('reason')=='action_event'
                            and decision.get('critical_step') in decision.get('full_approach_target_steps',[])):
                        expected='Full'
                    if cfg.get('m3_fast_every_event'):
                        force_event=decision['reason']=='action_event'
                        check('critical_event_fast_explicit_override',f.get('event_fast_override')==force_event,pid)
                        if force_event:expected='Fast'
                    if decision.get('safe_refresh'):
                        proof=decision['safe_refresh']
                        alarm=by_id[proof['prediction_id']].get('safe_failure_probe',{})
                        check('safe_full_response_opt_in',bool(cfg.get('safe_failure_probe'))
                            and cfg.get('safe_probe_mode')=='refresh' and alarm.get('new_alarm') is True
                            and proof==request.get('safe_failure_response')
                            and f.get('reason')=='safe_failure_probe',pid)
                        expected='Full'
                    if decision.get('visual_refresh'):
                        proof=decision['visual_refresh']
                        observed=next((e for e in events if e.get('event')=='visual_monitor'
                            and e.get('sample_id')==proof['sample_id']),{})
                        check('visual_full_response_opt_in',cfg.get('visual_monitor',{}).get('mode')=='refresh'
                            and observed.get('alarm') is True
                            and proof==request.get('visual_monitor_response')
                            and f.get('reason')=='visual_monitor',pid)
                        expected='Full'
                    check('ordinary_choice_formula',f['mode']==expected,pid)
                check('ordinary_explicit_mode',declared==expected,pid)
                event_exception=cfg.get('m3_fast_every_event') and decision.get('reason')=='action_event'
                check('ordinary_no_consecutive_fast',declared!='Fast' or event_exception
                    or by_id[pid-1]['m3_fast_choice']['mode']=='Full',pid)
                expected_mode=cfg.get('m3_ordinary_fast_visual_mode','selective320') if declared=='Fast' else 'full'
            if cfg.get('m3_fast_enabled'):
                choice=request.get('m3_fast_choice',{})
                declared=choice.get('mode')
                previous_mode=by_id[pid-1].get('m3_fast_choice',{}).get('mode') if pid else 'Full'
                decision=request.get('m3_fast_decision',{})
                if origin:
                    recorded={k:v for k,v in decisions.get(origin,{}).items() if k not in ('event','step')}
                    check('m3_fast_decision_bound_to_trace',decision==recorded,pid)
                if not origin:
                    expected_choice='Full'
                else:
                    if cfg.get('m3_fast_policy') in ('m3_two_mode_deadline_energy_v2','m3_full_fast_execution_window_v3','m3_service_ledger_fast_advance_v4'):
                        formula=decision['m3_fast_formula']
                        if cfg.get('m3_fast_policy') in ('m3_full_fast_execution_window_v3','m3_service_ledger_fast_advance_v4'):
                            check('fast_advance_policy_bound_to_config',bool(cfg.get('m3_fast_advance'))==(formula.get('policy')=='m3_service_ledger_fast_advance_v4'),pid)
                            check('full_only_flag_bound_to_config',decision.get('fast_full_only',False)==bool(cfg.get('m3_fast_full_only',False)),pid)
                            check('full_reserve_flag_bound_to_config',decision.get('fast_reserve_full',False)==bool(cfg.get('m3_fast_reserve_full',False)),pid)
                            window=decision['fast_execution_window']
                            check('fast_window_belongs_to_current_chunk',previous is not None and window['prediction_id']==previous['prediction_id'],pid)
                            expected_end=window['start_step']+math.ceil((window['full_recovery_seconds']+window['margin_seconds'])/window['period_seconds']-1e-9)
                            check('fast_execution_window_end',window['end_step']==expected_end,pid)
                            check('fast_critical_outside_committed_window',decision.get('critical_step') is None or decision['critical_step']>expected_end,pid)
                            check('fast_excluded_events_inside_window',all(e['target_step']<=expected_end for e in decision['fast_ignored_events']),pid)
                        if cfg.get('m3_fast_key_phases'):
                            from perceptbudget.key_phases import absolute_intervals
                            from perceptbudget.action_events import EventConfig
                            guard=decision.get('fast_phase_guard',{})
                            phase_source=by_id[previous['prediction_id']]
                            expected=absolute_intervals(phase_source['actions'],phase_source['observation_step'],EventConfig(**cfg['m3_events']))
                            check('key_phases_bound_to_active_chunk',guard.get('prediction_id')==previous['prediction_id']
                                  and guard.get('intervals')==expected
                                  and guard.get('horizon')==cfg['m3_timing']['maximum_action_offset'],pid)
                        if cfg.get('m3_key_phase_response_policy')=='full_once_v1':
                            check('key_phase_response_policy',decision.get('key_phase_response_policy')=='full_once_v1',pid)
                            phases=decision.get('fast_phase_guard',{}).get('intervals',[])
                            remaining=[p for p in phases if p['stop_step']>formula['observation_step']] if previous_mode=='Fast' else []
                            expected_response=None
                            if remaining:
                                p=min(remaining,key=lambda p:(p['start_step'],p['stop_step']))
                                expected_response=dict(prediction_id=previous['prediction_id'],start_step=p['start_step'],stop_step=p['stop_step'],target_step=max(formula['observation_step'],p['start_step']),state='active' if p['start_step']<=formula['observation_step'] else 'upcoming')
                            check('key_phase_response_from_active_fast',decision.get('key_phase_response')==expected_response,pid)
                            if expected_response:
                                check('key_phase_response_completion_or_terminal_drain',phase_completion_matches(request,committed.get(pid)),pid)
                        errors=audit_fast_formula(formula,decision,cfg['m3_fast_cost_model'],previous_mode)
                        probability=cfg['m3_timing']['service_quantile']
                        if decision.get('fast_reserve_full',False):
                            full_samples=service_history['Full']
                            recovery_q=max(sorted(v)[math.ceil(len(v)*probability)-1] for v in full_samples.values() if v)
                            check('fast_reserved_full_budget_from_history',abs(decision['fast_full_reserve_service_seconds']-recovery_q)<1e-8,pid)
                        for proposal in formula['candidates']:
                            samples=service_history[proposal['mode']]
                            h='recovery' if proposal['mode']=='Fast' and cfg.get('m3_fast_advance') else decision['service_budget_history']
                            histories=[h] if h!='recovery' else ['hot','short','long','recovery']
                            quantile=max(sorted(samples[x])[math.ceil(len(samples[x])*probability)-1]
                                         for x in histories if samples[x])
                            check('m3_fast_conditional_service_quantile',abs(proposal['service_seconds']-quantile)<1e-8,pid)
                        for error in errors:check('m3_fast_formula_'+error,False,pid)
                        check('m3_fast_formula_record',not errors,pid)
                        expected_choice=formula['mode']
                    else:
                        # Read-only compatibility for the old heuristic records.
                        window=math.ceil((decision['service_seconds']+decision['margin_seconds'])/decision['estimated_control_period_seconds']-1e-9)
                        critical=decision.get('critical_step')
                        near=critical is not None and origin<=critical<=origin+window
                        expected_choice='Full' if decision['reason']=='action_event' or near or previous_mode=='Fast' else 'Fast'
                check('m3_fast_only_visual_change',not cfg.get('m4_enabled') and cfg.get('m3_action_events_enabled') and not cfg.get('m3_dvac_enabled'),pid)
                check('m3_fast_explicit_choice',declared==expected_choice,pid)
                expected_mode='full' if declared=='Full' else cfg['m2_visual_fast_mode']
            if cfg.get('m4_enabled'):
                declared=request.get('m4_service_mode')
                check('m4_explicit_service_mode',declared in ('Full','Fast'),pid)
                expected_mode='full' if declared=='Full' else cfg['m2_visual_fast_mode']
                check('m4_full_bootstrap',pid!=0 or declared=='Full',pid)
                if declared=='Fast' and pid>0:
                    check('m4_no_consecutive_fast',by_id[pid-1].get('m4_service_mode')!='Fast',pid)
            for proof in request.get('m4_event_coverage',[]):
                from perceptbudget.action_events import EventConfig,detect_events
                ec=EventConfig(**cfg['m3_events']);commit=committed.get(pid,{})
                close=proof.get('close_target_step',-1000);slow=proof.get('target_step',-1000)
                timing=request.get('m3_timing') or {}
                extracted={(e['kind'],origin+e['offset']) for e in detect_events(request['actions'],ec,
                    start_offset=max(0,commit.get('commit_step',0)-origin))}
                check('m4_local_close_slowdown_proof',cfg.get('m4_cover_close_slowdown') is True
                    and request.get('m4_service_mode')=='Full' and pid>0
                    and timing.get('arrival_lateness_seconds',1)<=0
                    and 'gripper_close' in timing.get('candidate_event_kinds',[])
                    and abs(close-timing.get('critical_step',-1000))<=ec.match_steps
                    and 0<slow-close<=ec.match_steps
                    and commit.get('commit_step',float('inf'))<=close
                    and ('gripper_close',close) in extracted and ('slowdown',slow) in extracted,pid)
            geometry=visual_geometry(expected_mode)
            check('fresh_visual_schedule',visual.get('mode')==expected_mode,pid)
            check('fresh_visual_inputs',visual.get('current_images_encoded') is True
                  and visual.get('current_state_encoded') == (expected_mode=='full' or not cfg.get('m2_fast_no_state',False)),pid)
            check('fresh_visual_state_prompt',g.get('state_in_prompt') ==
                  (expected_mode=='full' or not cfg.get('m2_fast_no_state',False)),pid)
            check('fresh_visual_source',visual.get('source_observation_step')==origin
                  and visual.get('source_prediction_id')==pid,pid)
            check('fresh_visual_geometry',visual.get('num_views')==geometry['num_views']
                  and visual.get('visual_tokens')==geometry['visual_tokens']
                  and visual.get('camera_indices')==([1] if expected_mode=='wrist' else [0,1]),pid)
            check('fresh_visual_pool_factor',visual.get('vision_pool_factor',1)==geometry['vision_pool_factor'],pid)
            if expected_mode=='selective416':
                check('selective416_state_and_compression',cfg.get('m3_fast_selective416') is True and not cfg.get('m2_fast_no_state')
                    and visual.get('compression')=='redundancy_selected_32_of64_blocks'
                    and visual.get('external_tokens')==160 and visual.get('wrist_tokens')==256
                    and g.get('state_in_prompt') is True,pid)
            if expected_mode=='pool2_agent':
                if cfg.get('m3_external_token_selector')=='divprune':
                    indices=visual.get('external_token_indices',[])
                    motion=bool(cfg.get('m3_external_motion_tokens'))
                    check('fresh_visual_external_divprune',
                        visual.get('compression')==('motion16_divprune48_external_v1' if motion else 'divprune_cosine_max_min_external_v1')
                        and visual.get('external_tokens')==64 and visual.get('wrist_tokens')==256
                        and len(indices)==64 and indices==sorted(set(indices))
                        and all(type(i) is int and 0<=i<256 for i in indices),pid)
                    if motion:
                        seeds=visual.get('motion_token_indices',[])
                        fulls=[i for i,q in by_id.items() if i<pid and q.get('context_refresh',{}).get('mode')=='full']
                        reference=max(fulls) if fulls else None
                        check('fresh_visual_motion16',visual.get('motion_tokens_requested')==16
                            and visual.get('motion_tokens_used')==len(seeds)
                            and len(seeds) in (0,16) and len(set(seeds))==len(seeds)
                            and set(seeds)<=set(indices)
                            and visual.get('motion_score_method')=='mean_abs_rgb_patch16_v1'
                            and visual.get('motion_reference_prediction_id')==reference
                            and reference is not None
                            and visual.get('motion_reference_observation_step')==by_id[reference]['observation_step'],pid)
                else:check('fresh_visual_pooled_camera',visual.get('pooled_view')=='agent',pid)
            if expected_mode=='selective320':
                check('selective320_camera_counts',visual.get('pooled_view')=='both'
                    and visual.get('compression')=='redundancy_selected_32_of64_blocks_per_camera'
                    and visual.get('external_tokens')==160 and visual.get('wrist_tokens')==160,pid)
            if expected_mode=='selective464':
                check('selective464_camera_counts',visual.get('pooled_view')=='both'
                    and visual.get('compression')=='redundancy_selected_8_of64_blocks_per_camera'
                    and visual.get('external_tokens')==232 and visual.get('wrist_tokens')==232,pid)
            if cfg.get('m3_ordinary_fast_positions'):
                offset=512-visual['visual_tokens'] if expected_mode!='full' else 0
                check('selective_original_position_input',visual.get('position_coordinates')==(
                    'original_camera_and_text_coordinates_v1' if expected_mode!='full' else 'contiguous_v1')
                    and visual.get('action_position_offset')==offset
                    and visual.get('action_position_start')==request['rtc_guidance']['valid_prefix_length']+offset,pid)
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
                if g.get('visual_stop_replan'):
                    stopped=visual_waits.get(pid,{})
                    settings=cfg.get('visual_monitor',{})
                    response=request.get('visual_monitor_response',{})
                    begin=stopped.get('start_monotonic',float('inf'))
                    end=stopped.get('stop_monotonic',-1.)
                    check('visual_stop_replan_without_failed_tail',settings.get('mode')=='refresh'
                        and settings.get('response')=='stop_and_replan' and bool(response)
                        and g.get('guidance_enabled') is False and solver['previous_raw'] is None
                        and not g.get('vjp_norms') and solver['delay']==g['delay_estimate_steps']==0
                        and solver['executed']==s==g['executed_steps']
                        and g.get('old_tail_policy')=='discard_failed_reference'
                        and request['m3_fast_choice']['mode']=='Full',pid)
                    check('visual_stop_no_old_actions_until_ready',stopped.get('step')==origin==response.get('sample_step')
                        and stopped.get('resume_step')==origin and stopped.get('old_actions_executed')==0
                        and stopped.get('simulation_frozen') is True and end>=begin
                        and request['observation_monotonic']>=begin
                        and end>=request['ready_monotonic']
                        and not any(begin<=r['action_start_monotonic']<end for r in controls),pid)
                elif cfg.get('rtc_disable_guidance'):
                    check('unguided_without_old_target_or_vjp',g.get('guidance_enabled') is False
                        and solver['previous_raw'] is None and not g.get('vjp_norms')
                        and solver['executed']==s==g['executed_steps']
                        and g.get('solver_backend')=='original_openpi_eager_unguided',pid)
                else:
                    check('old_target_alignment',g.get('guidance_enabled') is True
                        and solver['executed']==s==g['executed_steps']
                        and eq(solver['previous_raw'],old['actions_raw'][s:]),pid)
            if m3:
                d=decisions.get(origin,{})
                t=request.get('m3_timing',{})
                if cfg.get('m3_three_frequency'):
                    f=t.get('three_frequency_formula',{})
                    candidates=f.get('candidates',[])
                    valid=len(candidates)==2 and [r['profile'] for r in candidates]==['gpu_emc','baseline']
                    check('three_frequency_candidates_recorded',valid,pid)
                    if valid:
                        now=t['decision_monotonic'];end=t['deadline']-t['margin_seconds']
                        for row in candidates:
                            latest=min(end-row['service_seconds'],row['rtc_cap'],row['first_cap'])
                            check('three_frequency_candidate_feasibility',
                                abs(row['latest_start']-latest)<1e-8 and row['feasible']==(latest>=now-1e-9),pid)
                        expected=candidates[0] if candidates[0]['feasible'] else candidates[1]
                        policy='medium_first_v1'
                        budget_policy=f.get('budget_policy',f.get('policy'))
                        if budget_policy=='critical_fast_own_budget_v1':
                            policy=budget_policy
                            check('fast_service_budget_opt_in',cfg.get('m3_fast_own_budget') is True
                                and t['reason']=='action_event',pid)
                            original=f.get('full_candidates',[])
                            check('fast_service_full_reference',len(original)==2,pid)
                            for row,full in zip(candidates,original):
                                profile=row['profile']
                                if row['mode']=='Fast':
                                    from perceptbudget.deadline_controller import percentile
                                    delay=max(mode_histories[cfg.get('m3_ordinary_fast_visual_mode','selective320')+':'+profile])
                                    prior=percentile(cfg['ordinary_fast_costs']['cells'][profile]['modes']['Fast']['samples_seconds'],
                                        cfg['m3_timing']['service_quantile'])
                                    period=t['estimated_control_period_seconds']
                                    transition=(full['switch_seconds']+full['ack_reserve_seconds']
                                        if t['current_profile']!=profile else 0.)
                                    service=min(full['service_seconds'],max(prior,delay*period)+period+transition)
                                    check('fast_service_uses_own_history',row['fast_delay_steps']==delay
                                        and abs(row['fast_prior_seconds']-prior)<1e-8
                                        and abs(row['service_seconds']-service)<1e-8,pid)
                                else:
                                    check('fast_service_full_unchanged',all(row[k]==v for k,v in full.items()),pid)
                        if cfg.get('m3_full_energy'):
                            from perceptbudget.full_energy import audit_record,POLICY
                            policy=POLICY;energy=f.get('energy',{})
                            check('full_energy_original_trigger',f.get('baseline_selected')==expected
                                  and t.get('energy_prepare_anchor',{}).get('latest_start')==expected['latest_start'],pid)
                            if energy:
                                check('full_energy_feasibility_bound',len(energy['candidates'])==len(candidates)
                                    and all(e['profile']==c['profile'] and e['feasible']==c['feasible']
                                            for e,c in zip(energy['candidates'],candidates)),pid)
                                errors=audit_record(energy,cfg['m3_full_energy_costs'],current_profile=t['current_profile'],
                                    minimum_idle=cfg['m3_timing']['minimum_low_idle_seconds'])
                                check('full_energy_calibration_and_segments',not errors,(pid,errors))
                                check('full_energy_common_window',energy['window_start']==t['decision_monotonic']
                                      and abs(energy['window_end']-(t['decision_monotonic']+
                                          t['admitted_remaining_actions']*t['estimated_control_period_seconds']))<1e-8
                                      and energy.get('window_basis')=='current_admitted_buffer',pid)
                                if energy['selected']:
                                    expected=next(c for c in candidates if c['profile']==energy['profile'])
                            check('full_energy_choice_present',bool(energy),pid)
                        if f.get('policy')=='critical_event_high_v1':
                            anchor=t.get('critical_high_prepare_anchor',{})
                            supplement=t.get('critical_freshness')
                            anchor_ok=(not anchor and not f.get('prepare_anchor')
                                and f.get('supplement_prepare_rule')=='selected_mode_service_budget'
                                and cfg.get('m3_action_signal_settings',{}).get('critical_freshness_max_age_steps') is not None
                                if supplement else f.get('prepare_anchor')==anchor
                                and anchor.get('profile')==expected['profile']
                                and anchor.get('latest_start')==expected['latest_start'])
                            check('critical_high_original_trigger',cfg.get('m3_critical_high') is True
                                and t['reason']=='action_event' and f.get('baseline_selected')==expected
                                and f.get('budget_policy')==policy
                                and anchor_ok,pid)
                            expected=candidates[1];policy='critical_event_high_v1'
                        elif cfg.get('m3_critical_high') and t['reason']=='action_event':
                            check('critical_high_event_must_use_high',False,pid)
                        check('three_frequency_profile_selection',f['selected']==expected
                            and request.get('m4_compute_profile')==expected['profile']
                            and request.get('m3_full_frequency')==expected['profile']
                            and request.get('m3_three_frequency_policy')==policy
                            and abs(t['service_seconds']-expected['service_seconds'])<1e-8,pid)
                if cfg.get('m3_low_full') and not cfg.get('m3_three_frequency'):
                    profile=request.get('m4_compute_profile')
                    check('m3_full_profile_declared',profile in ('baseline','gpu_emc')
                          and request.get('m3_full_frequency')==profile,pid)
                    formula=t.get('low_full_formula')
                    if formula:
                        from perceptbudget.m4_candidates import hold_energy
                        costs=cfg['m3_low_full_costs']
                        now=t['decision_monotonic'];end=t['deadline']-t['margin_seconds']
                        for row in formula['candidates']:
                            cell=next(c for c in costs['cells'] if c['profile']==row['profile']
                                and c['history']==row['history'] and c['from_profile']=='gpu_emc')
                            if formula.get('policy')=='original_m3_slot_v3':
                                latest=t['latest_start']
                                prepare=max(now,latest)
                                request_at=prepare+(1 if t['path']=='low_continue_recovery' else 0)*t['estimated_control_period_seconds']
                                begin=prepare if row['profile']=='baseline' else request_at
                                slot_end=end if row['profile']=='baseline' else min(end,
                                    request_at+t['rtc_delay_steps']*t['estimated_control_period_seconds']-t['margin_seconds'])
                                feasible=begin+row['service_seconds']<=slot_end+1e-9
                                before=max(0.,begin-now)
                                check('m3_original_request_slot',formula['original_latest_start']==t['latest_start']
                                    and formula['original_rtc_delay_steps']==t['rtc_delay_steps']
                                    and request.get('target_handoff_step') is None,pid)
                            else:
                                cap=formula['rtc_low_cap'] if row['profile']=='gpu_emc' else row['latest_start']
                                latest=min(end-row['service_seconds'],cap)
                                before=max(0.,latest-now)
                                feasible=latest>=now-1e-9
                            tail=max(0.,end-(now+before+cell['mean_service_seconds']))
                            energy=(hold_energy(costs['waits'],'gpu_emc',formula['quiet_age_seconds'],before)
                                +cell['mean_absolute_service_energy_j']+hold_energy(costs['waits'],row['profile'],0.,tail))
                            check('m3_full_common_window_energy',abs(row['energy_j']-energy)<1e-8
                                and abs(row['latest_start']-latest)<1e-8
                                and row['feasible']==feasible,pid)
                        choices=[r for r in formula['candidates'] if r['quality_allowed'] and r['feasible']]
                        if formula.get('policy') in ('deadline_first_v2','original_m3_slot_v3','shared_deadline_v4'):
                            low=next(r for r in formula['candidates'] if r['profile']=='gpu_emc')
                            chosen=low if low['quality_allowed'] and low['feasible'] else formula['candidates'][0]
                            check('m3_full_deadline_first_policy',cfg.get('m3_low_full_policy')==formula['policy']
                                and formula.get('energy_used_for_selection') is False,pid)
                        else:
                            chosen=min(choices,key=lambda r:(r['energy_j'],r['profile']!='baseline')) if choices else formula['candidates'][0]
                        check('m3_full_selected_policy',formula['selected']==chosen
                            and chosen['profile']==profile,pid)
                    if profile=='gpu_emc':
                        from perceptbudget.key_phases import absolute_intervals
                        from perceptbudget.action_events import EventConfig,detect_events
                        ec=EventConfig(**cfg.get('m3_events',{}))
                        phases=absolute_intervals(old['actions'],old['observation_step'],ec)
                        qlow=formula['selected']['service_seconds'] if formula else t['service_seconds']
                        period=t['estimated_control_period_seconds']
                        if 'low_full_dispatch_lead_seconds' in t:
                            ahead=t['latest_start']-t['decision_monotonic']
                            check('m3_low_full_last_feasible_tick',
                                cfg.get('m3_low_full_dispatch_policy')=='before_next_control_tick'
                                and t['low_full_dispatch_lead_seconds']==period
                                and -1e-9<=ahead<=period+1e-9 and t['start_now'],pid)
                        imminent=[e for e in detect_events(old['actions'],ec)
                            if 0<=(old['observation_step']+e['offset']-origin)*period<=qlow+t['margin_seconds']]
                        check('m3_low_full_only_ordinary_stage',bool(formula) and formula['ordinary']
                            and old['prediction_id']!=0 and t['reason']!='action_event'
                            and not any(p['start_step']<=origin<p['stop_step'] for p in phases)
                            and not imminent,pid)
                # Confirmed visual stops deliberately bypass the normal q gate:
                # an in-flight request can have started only1-2 actions ago.
                # Still require a later observation and the old commit first.
                minimum=1 if g.get('visual_stop_replan') else q
                check('m3_request_minimum_interval',origin>=max(old['observation_step']+minimum,previous['commit_step']),pid)
                check('m3_request_triggered_by_recorded_deadline',
                      d.get('start_now') is True and d.get('inflight') is False
                      and t.get('deadline')==d.get('deadline') and t.get('reason')==d.get('reason'),pid)
                policy=cfg.get('m3_signal_policy','earliest')
                if cfg.get('m3_critical_forecast'):
                    from perceptbudget.critical_forecast import forecast
                    from perceptbudget.action_events import EventConfig
                    proof=d.get('critical_forecast',{});at=proof.get('source_step',-1)
                    ec=EventConfig(**cfg['m3_events'])
                    past_steps=list(range(max(0,at-ec.confirm_steps),at))
                    source_decisions=[e for e in events if e['event']=='m3_decision'
                        and e['step']==at and e.get('decision_monotonic')==proof.get('source_monotonic')]
                    valid=(proof.get('version')==cfg['m3_critical_forecast']=='remaining_v1'
                        and proof.get('prediction_id')==old['prediction_id']
                        and previous['commit_step']<=at<=origin and bool(source_decisions)
                        and proof.get('history_steps')==past_steps
                        and all(i in by_step for i in past_steps))
                    check('remaining_forecast_source',valid,pid)
                    if valid:
                        offset=at-old['observation_step']
                        replay=forecast(old['actions'][offset:cfg['m3_timing']['maximum_action_offset']],
                            previous_actions=[by_step[i]['action'] for i in past_steps],
                            current_step=at,now=proof['source_monotonic'],
                            control_period=proof['source_period_seconds'],config=ec)
                        check('remaining_forecast_replay',proof.get('result')==replay
                            and source_decisions[0]['estimated_control_period_seconds']==proof['source_period_seconds'],pid)
                        if cfg.get('m3_feedback_policy'):
                            from perceptbudget.feedback_refresh import review_cues
                            report=d.get('feedback_review',{})
                            previous_ids=[i for i in by_id if i<old['prediction_id']]
                            reference_id=max(previous_ids,default=None)
                            reference_request=by_id.get(reference_id)
                            reference_proofs=[e['critical_forecast'] for e in events if e['event']=='m3_decision'
                                and e.get('critical_forecast',{}).get('prediction_id')==reference_id]
                            reference=(dict(prediction_id=reference_id,origin=reference_request['observation_step'],
                                actions=reference_request['actions'],events=reference_proofs[0]['result']['events'])
                                if reference_request and reference_proofs else None)
                            expected=review_cues(replay['events'],prediction_id=old['prediction_id'],
                                origin=old['observation_step'],source_step=at,period=proof['source_period_seconds'],
                                response_seconds=report['response_seconds'],margin_seconds=t['margin_seconds'],
                                minimum_steps=cfg['m3_feedback_minimum_steps'],actions=old['actions'],
                                reference=reference,config=ec,policy=cfg['m3_feedback_policy'],
                                full_approach_steps=([r['target_step'] for r in d.get('ordinary_turn_review',[])
                                    if r.get('full_approach',{}).get('requires_full')]
                                    if cfg.get('m3_full_approach_delta') is not None else None))
                            suppressed=d.get('feedback_suppressed',[])
                            allowed={(e['kind'],e['target_step']) for e in expected['events'] if e['suppress']}
                            check('feedback_window_fixed_and_replayed',
                                cfg['m3_feedback_policy']==report.get('policy')
                                and report['policy'] in ('overlap_pacing_v1','overlap_pacing_v2')
                                and report==expected and report==source_decisions[0].get('feedback_review'),pid)
                            check('feedback_suppression_scope',all((e['kind'],e['target_step']) in allowed
                                for e in suppressed) and not any(e['target_step']==d.get('critical_step')
                                and e['kind'] in d.get('candidate_event_kinds',[])
                                for e in suppressed),pid)
                        signal_candidate=None
                        if cfg.get('m3_action_signal_settings'):
                            from perceptbudget.action_signal_experiment import prepare
                            signal=d.get('action_signal_review',{})
                            signal_reference=(dict(origin=reference_request['observation_step'],
                                actions=reference_request['actions']) if reference_request else None)
                            signal_source=source_decisions[0]
                            signal_phases=(signal_source.get('refresh_phase_history',[])
                                if cfg.get('m3_phase_coverage_memory') else
                                [signal_source['refresh_phase_covered']]
                                if signal_source.get('refresh_phase_covered') else [])
                            packet=prepare(actions=old['actions'],origin=old['observation_step'],
                                source_step=at,history={i:by_step[i]['action'] for i in past_steps},
                                reference=signal_reference,config=ec,settings=cfg['m3_action_signal_settings'],
                                period=proof['source_period_seconds'],response_seconds=report['response_seconds'],
                                margin_seconds=t['margin_seconds'],
                                prediction_count=sum(i<=old['prediction_id'] for i in by_id),
                                covered_phases=signal_phases,
                                covered_events=point_history_at_decision(commits,by_id,at,proof['source_monotonic'],cfg))
                            check('action_signal_budget_and_features',signal==packet['review']
                                and signal==source_decisions[0].get('action_signal_review'),pid)
                            if 'action_revision' in d.get('candidate_event_kinds',[]):
                                check('action_signal_request_interval_budget',
                                    origin>=signal['earliest_new_request_step'],pid)
                            signal_candidate=packet['review']['candidate']
                        critical=d.get('critical_step')
                        freshness=d.get('critical_freshness')
                        if freshness:
                            from perceptbudget.critical_wait import covered_by_phase
                            limit=cfg.get('m3_action_signal_settings',{}).get('critical_freshness_max_age_steps')
                            prior_freshness=[q.get('m3_timing',{}).get('critical_freshness')
                                for q in by_id.values() if q['prediction_id']<pid]
                            covered=any(covered_by_phase(freshness,p)
                                for p in d.get('refresh_phase_history',[]))
                            if not covered:
                                covered=any(q.get('m3_timing',{}).get('critical_step') is not None
                                    and freshness['kind'] in q['m3_timing'].get('candidate_event_kinds',[])
                                    and abs(q['m3_timing']['critical_step']-critical)<=ec.match_steps
                                    and any(c['prediction_id']==q['prediction_id']
                                        and c['commit_monotonic']<=d['decision_monotonic'] for c in commits)
                                    for q in by_id.values() if q['prediction_id']<pid)
                            check('critical_freshness_source_and_age',type(limit) is int
                                and freshness['source_observation_step']==old['observation_step']
                                and freshness['source_age_steps']==critical-old['observation_step']>limit
                                and freshness['maximum_age_steps']==limit and covered
                                and freshness['target_step']==critical
                                and any(e['target_step']==critical and e['kind']==freshness['kind'] for e in replay['events']),pid)
                            wait_steps=cfg.get('m3_action_signal_settings',{}).get('critical_freshness_max_wait_steps',0)
                            expected_wait=max(0.,t['decision_monotonic']+t['service_seconds']+t['margin_seconds']-t['deadline'])
                            wait_allowed=(freshness.get('wait_policy')=='critical_freshness_short_wait_v1'
                                and type(wait_steps) is int and 0<wait_steps<=2
                                and freshness.get('maximum_wait_steps')==wait_steps
                                and abs(freshness.get('estimated_wait_seconds',-1.)-expected_wait)<1e-8
                                and 0.<expected_wait<=wait_steps*t['estimated_control_period_seconds']+1e-8)
                            check('critical_freshness_once_and_timely',not any(p and p['kind']==freshness['kind']
                                and p['start_step']<=critical<p['stop_step'] for p in prior_freshness)
                                and origin<critical and (expected_wait<=1e-8 or wait_allowed)
                                and request.get('m4_compute_profile')=='baseline'
                                and freshness['selected_mode']==request.get('m3_fast_choice',{}).get('mode','Full'),pid)
                            if freshness.get('pause_policy'):
                                correction=cfg.get('m3_action_signal_settings',{}).get('critical_freshness_correction_steps',0)
                                at=min(e['step'] for e in events if e.get('event')=='m3_decision'
                                    and e.get('decision_monotonic')==d['decision_monotonic'])
                                pause=max(at,critical-correction)
                                expected_pause=max(0.,t['decision_monotonic']+t['service_seconds']+t['margin_seconds']
                                    -(t['decision_monotonic']+(pause-at)*t['estimated_control_period_seconds']))
                                check('critical_freshness_pre_grasp_correction',type(correction) is int and 0<correction<=2
                                    and freshness['kind']=='gripper_close'
                                    and freshness['pause_policy']=='pre_grasp_correction_v1'
                                    and freshness['correction_steps']==correction and freshness['pause_step']==pause
                                    and abs(freshness['estimated_pause_seconds']-expected_pause)<1e-8,pid)
                        originals=[e for e in events if e['event']=='m3_decision'
                            and e.get('decision_monotonic')==d['decision_monotonic']]
                        decision_step=min(e['step'] for e in originals)
                        expected_time=d['decision_monotonic']+(critical-decision_step)*d['estimated_control_period_seconds'] if critical is not None else None
                        check('remaining_forecast_current_arrival',d.get('critical_time_monotonic')==expected_time
                            and (critical is None or any(e['target_step']==critical for e in replay['events'])
                                 or signal_candidate is not None and signal_candidate['target_step']==critical),pid)
                        if cfg.get('m3_phase_event_source'):
                            from perceptbudget.key_phases import intervals_for_timing
                            check('remaining_phase_event_source',
                                cfg['m3_phase_event_source']==d.get('phase_event_source')
                                and d.get('phase_event_source') in ('remaining_v1','remaining_v2'),pid)
                            shared=intervals_for_timing(old['actions'],old['observation_step'],ec,d)
                            phase=next((p for p in shared if p['end_observed'] and critical is not None
                                and p['start_step']<=critical<p['stop_step']),None)
                            expected_response=(dict(start_step=phase['start_step'],stop_step=phase['stop_step'],
                                source_prediction_id=old['prediction_id'],target_step=critical,
                                kinds=d['candidate_event_kinds']) if phase and d['reason']=='action_event' else None)
                            if expected_response and cfg['m3_phase_event_source']=='remaining_v2':
                                expected_response['event_source']='remaining_v2'
                            check('remaining_phase_response_replay',
                                d.get('refresh_phase_response')==expected_response
                                and t.get('refresh_phase_response')==expected_response,pid)
                            if cfg.get('m3_ordinary_turn_policy'):
                                expected_reviews=[ordinary_turn_from_record(dict(e,prediction_id=old['prediction_id']),
                                    old,proof,by_step,cfg['m3_events'],shared,approach_delta=cfg.get('m3_full_approach_delta'))
                                    for e in replay['events'] if e['kind']=='turn']
                                check('ordinary_turn_evidence_replay',
                                    cfg['m3_ordinary_turn_policy']==d.get('ordinary_turn_policy')=='context_v1'
                                    and cfg.get('m3_event_identity')=='turn_direction_v1'
                                    and d.get('ordinary_turn_review')==expected_reviews,pid)
                                if cfg.get('m3_full_approach_delta') is not None:
                                    check('full_approach_targets',d.get('full_approach_target_steps')==[
                                        r['target_step'] for r in expected_reviews
                                        if r.get('full_approach',{}).get('requires_full')],pid)
                                restored={(r['prediction_id'],r['target_step'])
                                    for r in d.get('refresh_phase_identity_review',[]) if not r['same_event']}
                                suppressed=d.get('ordinary_turn_suppressed',[])
                                check('ordinary_turn_suppression',all(e['kind']=='turn'
                                    and any(r['ordinary'] and r['target_step']==e['target_step']
                                        and r['prediction_id']==e['prediction_id'] for r in expected_reviews)
                                    and (e['prediction_id'],e['target_step']) not in restored
                                    for e in suppressed),pid)
                                check('ordinary_turn_retains_critical_signal',not any(
                                    e['target_step']==d.get('critical_step')
                                    and (not cfg.get('m3_action_signal_settings')
                                         or e['kind'] in d.get('candidate_event_kinds',[])) for e in suppressed),pid)
                if cfg.get('m3_phase_refresh'):
                    from perceptbudget.key_phases import intervals_for_timing
                    from perceptbudget.action_events import EventConfig
                    from perceptbudget.critical_wait import covered_by_phase
                    phase=d.get('refresh_phase_covered')
                    coverages=[phase] if phase else []
                    if cfg.get('m3_phase_coverage_memory'):
                        origins=[e['step'] for e in events if e['event']=='m3_decision'
                            and e.get('decision_monotonic')==d['decision_monotonic']]
                        expected_history=phase_history_at_decision(commits,by_id,min(origins),d['decision_monotonic'])
                        coverages=d.get('refresh_phase_history',[])
                        check('m3_refresh_phase_history_complete',
                            cfg['m3_phase_coverage_memory']=='all_active_v1'
                            and cfg.get('m3_phase_event_source')=='remaining_v2'
                            and coverages==expected_history,pid)
                    for phase in coverages:
                        completed=by_id.get(phase['completed_prediction_id'],{})
                        response=completed.get('m3_timing',{}).get('refresh_phase_response')
                        matched=[c for c in commits if c['prediction_id']==phase['completed_prediction_id']]
                        source=by_id.get(phase['source_prediction_id'],{})
                        phases=intervals_for_timing(source.get('actions',[]),source.get('observation_step',0),
                            EventConfig(**cfg.get('m3_events',{})),completed.get('m3_timing',{}))
                        check('m3_refresh_phase_provenance',bool(response) and bool(matched)
                            and all(phase.get(k)==response.get(k) for k in response)
                            and phase['completed_step']==matched[0]['commit_step']
                            and phase_active_at_decision(phase,d,events)
                            and any(p['end_observed'] and p['start_step']==phase['start_step']
                                and p['stop_step']==phase['stop_step'] for p in phases),pid)
                    reviews={}
                    if cfg.get('m3_event_identity'):
                        saved=d.get('refresh_phase_identity_review',[])
                        check('m3_turn_identity_version',cfg['m3_event_identity']=='turn_direction_v1'
                            and cfg.get('m3_phase_coverage_memory')=='all_active_v1',pid)
                        for review in saved:
                            key=(review['prediction_id'],review['target_step'])
                            cue=next((e for e in d['critical_forecast']['result']['events']
                                if e['kind']=='turn' and e['target_step']==review['target_step']),None)
                            valid=(cue is not None and review['kind']=='turn'
                                and review['prediction_id']==d['critical_forecast']['prediction_id']
                                and any(covered_by_phase(review,p) for p in coverages))
                            check('m3_turn_identity_source',valid and key not in reviews,pid)
                            if valid:
                                expected=turn_identity_from_record(review,coverages,by_id,cfg['m3_events'])
                                check('m3_turn_identity_replay',review==expected,pid)
                            reviews[key]=review
                    check('m3_refresh_phase_suppression',all(any(covered_by_phase(e,p) for p in coverages)
                        and (not cfg.get('m3_event_identity') or e['kind']!='turn'
                            or reviews.get((e['prediction_id'],e['target_step']),{}).get('same_event') is True)
                        for e in d.get('refresh_phase_suppressed',[])),pid)
                check('m3_signal_policy_matches_config',policy in ('earliest','latest')
                      and d.get('signal_policy','earliest')==policy,pid)
                if group is not None:
                    check('formal_runtime_signal_switches',d.get('action_events_enabled')==expected_events
                          and d.get('dvac_enabled')==expected_variance,pid)
                    expected_policy='independent' if group=='C_independent' else 'joint'
                    expected_budget='recovery' if group=='D_joint' and d.get('path')=='low_continue_recovery' else 'hot'
                    if (cfg.get('m3_low_full') and request.get('m4_compute_profile')=='gpu_emc'
                            and cfg.get('m3_low_full_policy')!='original_m3_slot_v3'):expected_budget='low'
                    if cfg.get('m3_three_frequency') and request.get('m4_compute_profile')=='baseline':
                        expected_budget='recovery' if d.get('current_profile')!='baseline' else 'hot'
                    check('formal_device_service_budget',d.get('dvfs_policy')==expected_policy
                          and d.get('service_budget_history')==expected_budget,pid)
                if 'signal_policy' in d:
                    required=('decision_monotonic','estimated_control_period_seconds','admitted_remaining_actions')
                    complete=all(k in d for k in required)
                    check('m3_signal_inputs_recorded',complete,pid)
                    if complete and policy in ('earliest','latest'):
                        originals=[e for e in events if e['event']=='m3_decision'
                                   and e.get('decision_monotonic')==d['decision_monotonic']]
                        original_step=min(e['step'] for e in originals)
                        now=d['decision_monotonic'];period=d['estimated_control_period_seconds']
                        limits=[(now+d['admitted_remaining_actions']*period,'buffer')]
                        signals=[(now+(d[key]-original_step)*period,reason)
                            for key,reason in (('critical_step','action_event'),('dvac_step','denoising_variance'))
                            if d.get(key) is not None]
                        if signals:
                            boundary=(min if policy=='earliest' else max)(x[0] for x in signals)
                            limits.append(min(x for x in signals if x[0]==boundary))
                        if age_limit is not None:
                            limits.append((old['observation_monotonic']+age_limit,'observation_age'))
                        if d.get('key_phase_response'):
                            target=d['key_phase_response']['target_step']
                            limits.append((now+(target-original_step)*period,'key_phase'))
                        expected_deadline,expected_reason=min(limits)
                        if d.get('grasp_lead'):
                            lead=d['grasp_lead'];steps=cfg.get('m3_grasp_lead_steps',0)
                            physical_deadline=expected_deadline
                            target=d['critical_step']-steps
                            expected_deadline=min(expected_deadline,now+(target-original_step)*period)
                            check('m3_grasp_correction_deadline',steps>0 and expected_reason=='action_event'
                                and 'gripper_close' in d.get('candidate_event_kinds',[])
                                and lead.get('policy')=='gripper_close_correction_lead_v1'
                                and lead.get('lead_steps')==steps
                                and lead.get('physical_critical_step')==d['critical_step']
                                and lead.get('refresh_target_step')==target
                                and abs(lead.get('lead_seconds',-1)-steps*period)<1e-8
                                and abs(lead.get('physical_deadline',-1)-physical_deadline)<1e-8,pid)
                        check('m3_signal_deadline_and_hard_bounds',
                              abs(d['deadline']-expected_deadline)<1e-8 and d['reason']==expected_reason,pid)
                expected_start=d['deadline']-d['service_seconds']-d['margin_seconds'] if d else float('nan')
                expected_constraint='service_deadline'
                if d.get('rtc_latest_submit_step') is not None:
                    # Recovery acknowledges HIGH on the following control tick;
                    # the copied plan still belongs to the original decision.
                    originals=[e for e in events if e['event']=='m3_decision'
                               and e.get('decision_monotonic')==d.get('decision_monotonic')]
                    decision_step=min(e['step'] for e in originals)
                    period=d['estimated_control_period_seconds']
                    delay=max(max(v) for v in mode_histories.values()) if cfg.get('m4_enabled') or cfg.get('m3_fast_enabled') else g.get('uncapped_delay_estimate_steps',g['delay_estimate_steps'])
                    if frequency_histories is not None:
                        delay=(max(frequency_histories[request['m4_compute_profile']])
                            if cfg.get('m3_low_full_rtc_policy')=='per_profile_v2'
                            else max(max(v) for v in frequency_histories.values()))
                        if cfg.get('rtc_fixed_delay_estimate') is not None:delay=cfg['rtc_fixed_delay_estimate']
                    if cfg.get('m3_ordinary_fast_context'):
                        delay=max(mode_histories['full:'+request['m4_compute_profile']])
                    latest_submit=old['observation_step']+cfg['H']-delay
                    changing=(d.get('current_profile')!=request.get('m4_compute_profile')) if cfg.get('m3_three_frequency') else d['path']=='low_continue_recovery'
                    guard=int(changing)+math.ceil(d['margin_seconds']/period-1e-9)
                    prefix_start=d['decision_monotonic']+(latest_submit-decision_step-guard)*period
                    if cfg.get('m3_fast_advance') and request.get('m3_fast_choice',{}).get('mode')=='Fast':
                        prefix_start-=math.ceil((d['service_seconds']+d['margin_seconds'])/period-1e-9)*period
                    check('m3_rtc_prefix_bound_inputs',d['rtc_delay_steps']==delay
                          and d['rtc_latest_submit_step']==latest_submit
                          and d['rtc_submit_guard_steps']==guard,pid)
                    if prefix_start<expected_start:expected_constraint='rtc_prefix'
                    expected_start=min(expected_start,prefix_start)
                if d.get('first_refresh_cap'):
                    cap=d['first_refresh_cap']
                    originals=[e for e in events if e['event']=='m3_decision'
                               and e.get('decision_monotonic')==d.get('decision_monotonic')]
                    decision_step=min(e['step'] for e in originals)
                    recovery_ticks=int(d.get('current_profile')!=request.get('m4_compute_profile')) if cfg.get('m3_three_frequency') else (1 if d['path']=='low_continue_recovery' else 0)
                    periodic=cfg.get('m3_max_refresh_interval')
                    request_step=old['observation_step']+periodic if periodic is not None else 30
                    allowed=(20,30,50) if cfg.get('m3_ordinary_turn_policy')=='context_v1' else (30,)
                    valid_cap=(periodic in allowed and cap.get('scope')=='every_observation'
                        and cap.get('source_prediction_id')==old['prediction_id']
                        and cap.get('observation_step')==old['observation_step']
                        and cap.get('interval_steps')==periodic and origin<=request_step) if periodic is not None else (
                        cfg.get('m3_first_refresh_step')==30 and old['prediction_id']==0 and d.get('critical_step') is None)
                    if cap.get('scope')=='post_close_feedback':
                        from perceptbudget.post_close_feedback import confirmed_closure,saturated_lift_warning
                        proof=cap['feedback']
                        prior_controls=[by_step[i] for i in sorted(by_step)
                            if i<proof['activation_step']]
                        closure=confirmed_closure(prior_controls,EventConfig(**cfg.get('m3_events',{})))
                        previous_observation=max((r['observation_step'] for r in requests
                            if r['prediction_id']<pid),default=-1)
                        request_step=proof['activation_step']+1
                        signal_settings=cfg.get('m3_action_signal_settings',{})
                        gated=bool(signal_settings.get('gripper_lift_feedback'))
                        activation=next((e for e in events if e.get('event')=='m3_decision'
                            and e.get('decision_monotonic')==proof['activation_monotonic']),None) if gated else None
                        wait_limit=(2*activation['estimated_control_period_seconds'] if gated and activation
                            else proof['full_seconds']+proof['margin_seconds'])
                        gate_valid=True
                        if gated:
                            source=old['observation_step']
                            next_actions=old['actions'][proof['activation_step']-source:
                                proof['activation_step']-source+ec.confirm_steps]
                            warning=saturated_lift_warning(prior_controls,next_actions,ec)
                            gate_valid=(activation is not None and warning is not None and warning==proof.get('gripper_lift_warning')
                                and abs(proof.get('existing_wait_threshold_seconds',-1.)-wait_limit)<1e-8)
                        valid_cap=(bool(signal_settings.get('post_close_feedback') or gated) and gate_valid
                            and proof.get('policy')=='post_close_observation_v1'
                            and closure=={k:proof[k] for k in ('close_step','confirmed_step')}
                            and proof['observation_step']==previous_observation
                            and previous_observation<proof['confirmed_step']<=proof['activation_step']<=origin
                            and proof['request_step']==request_step
                            and proof['full_seconds']>0
                            and proof['ordinary_wait_seconds']>wait_limit
                            and abs(proof['ordinary_wait_seconds']-(proof['ordinary_latest_start']-
                                proof['activation_monotonic']))<1e-8)
                    if cap.get('scope')=='safe_failure_probe':
                        proof=cap['alarm'];request_step=proof['request_step']
                        alarm=by_id[proof['prediction_id']].get('safe_failure_probe',{})
                        source_commit=next((c for c in commits if c['prediction_id']==proof['prediction_id']),None)
                        valid_cap=(bool(cfg.get('safe_failure_probe')) and cfg.get('safe_probe_mode')=='refresh'
                            and alarm.get('new_alarm') is True and source_commit is not None
                            and proof['commit_step']==source_commit['commit_step']
                            and request_step==proof['commit_step']+1
                            and d.get('safe_refresh')==proof and request.get('safe_failure_response')==proof
                            and sum(bool(r.get('safe_failure_response')) for r in requests)==1)
                    if cap.get('scope')=='visual_monitor':
                        proof=cap['alarm'];request_step=proof['request_step']
                        observed=next((e for e in events if e.get('event')=='visual_monitor'
                            and e.get('sample_id')==proof['sample_id']),{})
                        valid_cap=(cfg.get('visual_monitor',{}).get('mode')=='refresh'
                            and observed.get('alarm') is True and request_step==observed.get('step')
                            and proof['sample_step']==observed.get('step')
                            and proof['observed_monotonic']==observed.get('acquisition',{}).get('stop_monotonic')
                            and proof['observed_monotonic']<request['observation_monotonic']
                            and d.get('visual_refresh')==proof and request.get('visual_monitor_response')==proof)
                    first_start=d['decision_monotonic']+(request_step-decision_step-recovery_ticks)*d['estimated_control_period_seconds']
                    check('m3_first_refresh_cap_inputs',valid_cap
                          and cap['request_step']==request_step and cap['recovery_ticks']==recovery_ticks
                          and abs(cap['latest_start']-first_start)<1e-8,pid)
                    if first_start<expected_start:expected_constraint='first_refresh'
                    expected_start=min(expected_start,first_start)
                if d.get('rtc_latest_submit_step') is not None or d.get('first_refresh_cap'):
                    anchor=d.get('energy_prepare_anchor')
                    if cfg.get('m3_full_energy') and anchor and anchor['latest_start']<expected_start-1e-9:
                        expected_start=anchor['latest_start'];expected_constraint='full_energy_fixed_prepare'
                    anchor=d.get('critical_high_prepare_anchor')
                    if cfg.get('m3_critical_high') and anchor and anchor['latest_start']<expected_start-1e-9:
                        expected_start=anchor['latest_start'];expected_constraint='critical_event_fixed_prepare'
                    check('m3_start_constraint',d['start_constraint']==expected_constraint,pid)
                if d.get('fast_reserve_full') and request.get('m3_fast_choice',{}).get('mode')=='Fast':
                    expected_start-=math.ceil((d['fast_full_reserve_service_seconds']+d['margin_seconds'])/d['estimated_control_period_seconds']-1e-9)*d['estimated_control_period_seconds']
                check('m3_service_and_margin_subtracted_once',bool(d)
                      and abs(d['latest_start']-expected_start)<1e-8,pid)
                check('m3_nonnegative_wait_and_due',(d.get('wait_seconds',-1)>=0 if cfg.get('m4_enabled') else d.get('wait_seconds')==0)
                      and d.get('service_seconds',0)>0 and d.get('margin_seconds',-1)>=0,pid)
                if cfg.get('m3_start_floor_ms'):
                    timing=d.get('start_point_rounding',{})
                    originals=[e for e in events if e['event']=='m3_decision'
                               and e.get('decision_monotonic')==d.get('decision_monotonic')]
                    at=min(e['step'] for e in originals)
                    period=d['estimated_control_period_seconds']
                    remaining=d['latest_start']-d['decision_monotonic']
                    check('startup_point_not_duration_rounding',cfg['m3_start_floor_ms']==25
                        and timing.get('decision_step')==at
                        and abs(timing.get('continuous_start_step',float('inf'))-(at+remaining/period))<1e-7
                        and timing.get('limit_seconds')==.025
                        and timing.get('floor_allowed')==(remaining<=.025+1e-9),pid)
                    ack=d.get('startup_ack_wait')
                    if ack:
                        a,b=ack['start_monotonic'],ack['stop_monotonic']
                        check('startup_ack_wait_no_actions',timing['floor_allowed']
                            and ack['limit_seconds']==.025 and ack['simulation_frozen']
                            and not ack['queue_consumed'] and b>=a
                            and not any(a<=r['action_start_monotonic']<b for r in controls),pid)
                        check('startup_after_frequency_wait',request['observation_monotonic']>=b,pid)
                # The source step and normalized state checks above independently
                # bind this request to the actual observation at the trigger.
            else:
                check('next_request_counter_rule',origin==max(old['observation_step']+q,previous['commit_step']),pid)
            check('no_overlapping_inference',request['inference_started_monotonic']>=old['inference_ended_monotonic'],pid)
            check('guidance_delay_matches_solver',solver['delay']==g['delay_estimate_steps'],pid)
            if synchronous:
                check('sync_request_interval',origin==old['observation_step']+q,pid)
            elif cfg.get('rtc_paper_schedule'):
                estimate=cfg.get('rtc_fixed_delay_estimate')
                expected_history=history
                if frequency_histories is not None:
                    profile=request.get('m4_compute_profile')
                    check('full_frequency_delay_profile',profile in frequency_histories
                          and request.get('full_delay_profile')==profile
                          and g.get('delay_estimate_mode')=='frequency_history_max',pid)
                    expected_history=frequency_histories.get(profile,[])
                if mode_histories is not None:
                    mode=request.get('context_refresh' if cfg.get('m2_visual_alternation') else 'kv_recompute',{}).get('mode')
                    check('planned_delay_mode_matches_execution',request.get('planned_context_mode' if cfg.get('m2_visual_alternation') else 'planned_kv_mode')==mode,pid)
                    check('delay_history_mode',g.get('delay_history_mode')==mode
                        and g.get('delay_estimate_mode')=='mode_history_max',pid)
                    key=mode+':'+request['m4_compute_profile'] if cfg.get('m4_enabled') or cfg.get('m3_ordinary_fast_context') else mode
                    if cfg.get('m4_enabled') or cfg.get('m3_ordinary_fast_context'):
                        check('m4_compute_delay_key',request.get('planned_delay_key')==key and g.get('delay_history_key')==key,pid)
                    expected_history=mode_histories.get(key,[])
                expected_delay=max(expected_history) if estimate is None else estimate
                if request.get('visual_stop_interrupted'):
                    interrupted=request['visual_stop_interrupted']
                    observed=next((s for s in visual_samples if s.get('sample_id')==interrupted.get('sample_id')), {})
                    check('cnn_inflight_stop_not_async_history',cnn_settings
                        and observed.get('alarm') is True
                        and interrupted.get('step')==observed.get('step')
                        and request.get('delay_queue_observed') is False
                        and bool(g.get('delay_history_excluded_reason')),pid)
                if g.get('visual_stop_replan'):
                    check('visual_stop_zero_delay_not_async_history',
                        g.get('uncapped_delay_estimate_steps')==expected_delay
                        and request.get('delay_queue_observed') is False
                        and bool(g.get('delay_history_excluded_reason')),pid)
                    expected_delay=0
                elif cfg.get('rtc_critical_wait_prefix_policy')=='committed_stop_v1':
                    review=dict(wait=wait_target(request) is not None,reason='point_event')
                    if review['wait'] and cfg.get('m3_critical_wait_phases'):
                        review=wait_guard.review(request,old['actions'],old['observation_step'],EventConfig(**cfg['m3_events']))
                    prefix=rtc_prefix_plan(request,origin,expected_delay,review)
                    check('known_stop_prefix_record',cfg.get('m3_critical_wait') is True
                          and request.get('rtc_wait_prefix')==prefix
                          and g.get('uncapped_delay_estimate_steps')==expected_delay,pid)
                    expected_delay=prefix['selected_delay_steps']
                    if pid in critical_waits:
                        check('known_stop_wait_trace',review['wait'] and
                              critical_waits[pid]['step']==max(origin,prefix['stop_step']),pid)
                        wait_guard.complete(review)
                check('independent_history_max' if estimate is None else 'fixed_guidance_estimate',
                      g['delay_estimate_steps']==expected_delay
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
        if origin and cfg.get('rtc_critical_wait_prefix_policy')=='committed_stop_v1':
            cap=request.get('rtc_wait_prefix',{}).get('maximum_consumed_steps')
            if cap is not None:
                check('known_stop_no_uncommitted_execution',actual<=cap,pid)
        if origin and not synchronous and not g.get('visual_stop_replan'):
            for step in range(origin,at):
                source=by_step[step]['source']
                old=by_id[previous['prediction_id']]
                limit=cfg['m3_timing']['maximum_action_offset'] if m3 else cfg['H']
                exhausted=(step>=old['observation_step']+limit or (age_limit is not None and
                    by_step[step]['action_start_monotonic']>=old['observation_monotonic']+age_limit))
                check('old_actions_during_inference',source is None if exhausted else
                      source is not None and source['prediction_id']==previous['prediction_id'],(pid,step))
            slacks.append(g['delay_estimate_steps']-actual)
            if not request.get('visual_stop_interrupted'):
                history=(history+[actual])[-cfg.get('rtc_delay_buffer_size',10):]
            if frequency_histories is not None and not request.get('visual_stop_interrupted'):
                profile=request['m4_compute_profile']
                frequency_histories[profile]=(frequency_histories[profile]+[actual])[-cfg.get('rtc_delay_buffer_size',10):]
            if mode_histories is not None and not request.get('visual_stop_interrupted'):
                mode=request['context_refresh' if cfg.get('m2_visual_alternation') else 'kv_recompute']['mode']
                key=mode+':'+request['m4_compute_profile'] if cfg.get('m4_enabled') or cfg.get('m3_ordinary_fast_context') else mode
                mode_histories[key]=(mode_histories[key]+[actual])[-cfg.get('rtc_delay_buffer_size',10):]
        if service_history is not None and origin:
            timing=request['m3_timing'];mode=request['m3_fast_choice']['mode']
            service_bucket=timing.get('history','hot')
            service_seconds=c['commit_monotonic']-timing.get('service_start_monotonic',request['observation_monotonic'])
            service_history[mode][service_bucket]=(service_history[mode][service_bucket]+[service_seconds])[-128:]
        previous=c
    underflows=[]
    for step,row in by_step.items():
        active=latest_commit_at_step(commits,step)
        request=by_id[active['prediction_id']]
        offset=step-request['observation_step']
        check('control_timestamp_order',row['tick_monotonic']<=row['action_start_monotonic']<=row['action_end_monotonic'],step)
        if step:
            check('sequential_action_timestamps',by_step[step-1]['action_end_monotonic']<=row['tick_monotonic'],step)
        check('commit_before_action',active['commit_monotonic']<=row['action_start_monotonic'],step)
        limit=cfg['m3_timing']['maximum_action_offset'] if m3 else cfg['H']
        if offset>=limit or (age_limit is not None and row['action_start_monotonic']>=
                request['observation_monotonic']+age_limit):
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
        audit_mode='m3_dynamic_rtc' if m3 else 'sync_no_rtc' if synchronous else 'async_rtc',
        scope_id='m3_dynamic_rtc_records_v1' if m3 else 'm1_sync_no_rtc_records_v1' if synchronous else 'm1_async_rtc_records_v1',
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
