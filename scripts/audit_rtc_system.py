"""Independent saved-record system audit; does not call production queue/history helpers."""
import argparse
import json
import math
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
    if cfg.get('m4_enabled') and mode_histories is not None:
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
            if cfg.get('m2_fast_no_state',False):
                check('fresh_visual_state_prompt',g.get('state_in_prompt') == (expected_mode=='full'),pid)
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
            if m3:
                d=decisions.get(origin,{})
                t=request.get('m3_timing',{})
                check('m3_request_minimum_interval',origin>=max(old['observation_step']+q,previous['commit_step']),pid)
                check('m3_request_triggered_by_recorded_deadline',
                      d.get('start_now') is True and d.get('inflight') is False
                      and t.get('deadline')==d.get('deadline') and t.get('reason')==d.get('reason'),pid)
                policy=cfg.get('m3_signal_policy','earliest')
                check('m3_signal_policy_matches_config',policy in ('earliest','latest')
                      and d.get('signal_policy','earliest')==policy,pid)
                if group is not None:
                    check('formal_runtime_signal_switches',d.get('action_events_enabled')==expected_events
                          and d.get('dvac_enabled')==expected_variance,pid)
                    expected_policy='independent' if group=='C_independent' else 'joint'
                    expected_budget='recovery' if group=='D_joint' and d.get('path')=='low_continue_recovery' else 'hot'
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
                    delay=max(max(v) for v in mode_histories.values()) if cfg.get('m4_enabled') or cfg.get('m3_fast_enabled') else g['delay_estimate_steps']
                    latest_submit=old['observation_step']+cfg['H']-delay
                    guard=(1 if d['path']=='low_continue_recovery' else 0)+math.ceil(d['margin_seconds']/period-1e-9)
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
                    recovery_ticks=1 if d['path']=='low_continue_recovery' else 0
                    first_start=d['decision_monotonic']+(30-decision_step-recovery_ticks)*d['estimated_control_period_seconds']
                    check('m3_first_refresh_cap_inputs',cfg.get('m3_first_refresh_step')==30
                          and old['prediction_id']==0 and d.get('critical_step') is None
                          and cap['request_step']==30 and cap['recovery_ticks']==recovery_ticks
                          and abs(cap['latest_start']-first_start)<1e-8,pid)
                    if first_start<expected_start:expected_constraint='first_refresh'
                    expected_start=min(expected_start,first_start)
                if d.get('rtc_latest_submit_step') is not None or d.get('first_refresh_cap'):
                    check('m3_start_constraint',d['start_constraint']==expected_constraint,pid)
                if d.get('fast_reserve_full') and request.get('m3_fast_choice',{}).get('mode')=='Fast':
                    expected_start-=math.ceil((d['fast_full_reserve_service_seconds']+d['margin_seconds'])/d['estimated_control_period_seconds']-1e-9)*d['estimated_control_period_seconds']
                check('m3_service_and_margin_subtracted_once',bool(d)
                      and abs(d['latest_start']-expected_start)<1e-8,pid)
                check('m3_nonnegative_wait_and_due',(d.get('wait_seconds',-1)>=0 if cfg.get('m4_enabled') else d.get('wait_seconds')==0)
                      and d.get('service_seconds',0)>0 and d.get('margin_seconds',-1)>=0,pid)
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
                if mode_histories is not None:
                    mode=request.get('context_refresh' if cfg.get('m2_visual_alternation') else 'kv_recompute',{}).get('mode')
                    check('planned_delay_mode_matches_execution',request.get('planned_context_mode' if cfg.get('m2_visual_alternation') else 'planned_kv_mode')==mode,pid)
                    check('delay_history_mode',g.get('delay_history_mode')==mode
                        and g.get('delay_estimate_mode')=='mode_history_max',pid)
                    key=mode+':'+request['m4_compute_profile'] if cfg.get('m4_enabled') else mode
                    if cfg.get('m4_enabled'):
                        check('m4_compute_delay_key',request.get('planned_delay_key')==key and g.get('delay_history_key')==key,pid)
                    expected_history=mode_histories.get(key,[])
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
                limit=cfg['m3_timing']['maximum_action_offset'] if m3 else cfg['H']
                exhausted=(step>=old['observation_step']+limit or (age_limit is not None and
                    by_step[step]['action_start_monotonic']>=old['observation_monotonic']+age_limit))
                check('old_actions_during_inference',source is None if exhausted else
                      source is not None and source['prediction_id']==previous['prediction_id'],(pid,step))
            slacks.append(g['delay_estimate_steps']-actual)
            history=(history+[actual])[-cfg.get('rtc_delay_buffer_size',10):]
            if mode_histories is not None:
                mode=request['context_refresh' if cfg.get('m2_visual_alternation') else 'kv_recompute']['mode']
                key=mode+':'+request['m4_compute_profile'] if cfg.get('m4_enabled') else mode
                mode_histories[key]=(mode_histories[key]+[actual])[-cfg.get('rtc_delay_buffer_size',10):]
        if service_history is not None and origin:
            timing=request['m3_timing'];mode=request['m3_fast_choice']['mode']
            service_bucket=timing.get('history','hot')
            service_seconds=c['commit_monotonic']-timing.get('service_start_monotonic',request['observation_monotonic'])
            service_history[mode][service_bucket]=(service_history[mode][service_bucket]+[service_seconds])[-128:]
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
