import json
import numpy as np
import pytest
from scripts.audit_rtc_system import audit,decode_raw


def test_phase_provenance_uses_decision_before_clock_acknowledgement():
    from scripts.audit_rtc_system import phase_active_at_decision
    phase=dict(completed_step=86,stop_step=104)
    timing=dict(decision_monotonic=10.)
    events=[dict(event='m3_decision',step=103,decision_monotonic=10.,clock_wait=True),
            dict(event='m3_decision',step=104,decision_monotonic=10.,start_now=True)]
    assert phase_active_at_decision(phase,timing,events)
    assert not phase_active_at_decision(phase,timing,events[1:])
    assert not phase_active_at_decision(phase,timing,[])


def fixture(tmp_path):
    (tmp_path/'rtc').mkdir()
    stats={'state':{'q01':[-1.]*8,'q99':[1.]*8},'actions':{'q01':[-1.]*7,'q99':[1.]*7}}
    norm=tmp_path/'norm.json';norm.write_text(json.dumps(stats))
    cfg=dict(H=25,rtc_initial_delay_steps=5,controlled_delay_steps=5,rtc_paper_schedule=False,norm_stats_path=str(norm))
    (tmp_path/'run.json').write_text(json.dumps({'config_snapshot':cfg}))
    normalized=((np.zeros(8,dtype=np.float32)+1)/(2+1e-6)*2-1).tolist()
    requests=[]
    for pid,origin in ((0,0),(1,5)):
        raw=np.full((25,32),pid*.1,dtype=np.float32)
        requests.append(dict(prediction_id=pid,observation_step=origin,actions_raw=raw.tolist(),
            actions=decode_raw(raw,stats).tolist(),observation_raw_state=[0]*8,model_state=normalized,
            observation_monotonic=origin*10.,input_ready_monotonic=origin*10.+.1,
            inference_started_monotonic=origin*10.+.2,inference_ended_monotonic=origin*10.+.3,
            ready_monotonic=origin*10.+.4,rtc_solver_inputs=dict(noise=np.zeros((25,32)).tolist(),
                previous_raw=np.zeros((20,32)).tolist() if pid else None,executed=5 if pid else 0,delay=5 if pid else 0),
            rtc_guidance=dict(guidance_enabled=bool(pid),delay_estimate_steps=5 if pid else 0,
                executed_steps=5 if pid else 0,actual_delay_steps=5 if pid else 0)))
    events=[]
    for pid,origin,at in ((0,0,0),(1,5,10)):
        events.append(dict(event='prediction_committed',prediction_id=pid,observation_step=origin,
            commit_step=at,commit_monotonic=at*10.+.5,expired_new_actions=at-origin,
            new_prediction_remaining_actions=25-(at-origin)))
    for step in range(11):
        pid=int(step>=10);offset=step-(5 if pid else 0)
        events.append(dict(event='control_step',step=step,before={'raw_state':[0]*8},
            tick_monotonic=step*10.,action_start_monotonic=step*10.+.6,
            action_end_monotonic=step*10.+1,applied_arm_gain=1,action=requests[pid]['actions'][offset],
            source=dict(prediction_id=pid,offset=offset,origin_step=5 if pid else 0)))
    def save():
        (tmp_path/'rtc/q5_requests.json').write_text(json.dumps(requests))
        (tmp_path/'rtc/trace.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    save()
    return requests,events,save


def test_synthetic_valid_system_trace(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)['passed']


def test_formal_event_only_audit_checks_config_and_runtime_switches(tmp_path):
    requests,events,save=fixture(tmp_path)
    meta=json.loads((tmp_path/'run.json').read_text())
    cfg=meta['config_snapshot']
    cfg.update(H=50,N=5,precision='bfloat16',state_enabled=True,m3_scheduler_mode='deadline_high',
        m3_formal_group='B_high',m3_formal_signal='events_only',m3_dvac_enabled=False,
        m3_action_events_enabled=True,m3_merge_slowdown_steps=None,m3_request_timeout_seconds=3.,
        m3_watchdog_path=str(tmp_path/'watchdog.json'),
        m3_timing=dict(maximum_action_offset=50,maximum_observation_age_seconds=None))
    (tmp_path/'run.json').write_text(json.dumps(meta))
    for request in requests:
        request['actions_raw']+=request['actions_raw'][:25]
        request['actions']+=request['actions'][:25]
        request['rtc_solver_inputs']['noise']=[[0.]*32 for _ in range(50)]
    requests[1]['rtc_solver_inputs']['previous_raw']=[[0.]*32 for _ in range(45)]
    for event in events:
        if event['event']=='prediction_committed':event['new_prediction_remaining_actions']+=25
    decision=dict(event='m3_decision',step=5,start_now=True,inflight=False,deadline=50.5,latest_start=49.5,
        service_seconds=.9,margin_seconds=.1,wait_seconds=0.,reason='action_event',signal_policy='earliest',
        decision_monotonic=50.,estimated_control_period_seconds=.05,admitted_remaining_actions=45,
        critical_step=15,dvac_step=None,action_events_enabled=True,dvac_enabled=False,
        dvfs_policy='joint',service_budget_history='hot',path='high_frequency')
    events.append(decision);requests[1]['m3_timing']=dict(decision);save()
    assert audit(tmp_path)['passed']
    cfg['m3_dvac_enabled']=True;(tmp_path/'run.json').write_text(json.dumps(meta))
    assert 'formal_signal_switches' in [f['check'] for f in audit(tmp_path)['failures']]
    cfg['m3_dvac_enabled']=False;(tmp_path/'run.json').write_text(json.dumps(meta))
    decision['dvac_enabled']=True;save()
    assert 'formal_runtime_signal_switches' in [f['check'] for f in audit(tmp_path)['failures']]


def test_latest_signal_audit_recomputes_boundary_and_rejects_tampered_deadline(tmp_path):
    requests,events,save=fixture(tmp_path)
    meta=json.loads((tmp_path/'run.json').read_text())
    meta['config_snapshot'].update(m3_scheduler_mode='deadline_high',m3_signal_policy='latest',
        m3_timing={'maximum_action_offset':25,'maximum_observation_age_seconds':None})
    (tmp_path/'run.json').write_text(json.dumps(meta))
    decision=dict(step=5,start_now=True,inflight=False,deadline=50.5,latest_start=49.5,
        service_seconds=.9,margin_seconds=.1,wait_seconds=0.,reason='action_event',signal_policy='latest',
        decision_monotonic=50.,estimated_control_period_seconds=.05,admitted_remaining_actions=20,
        critical_step=15,dvac_step=10)
    events.append(dict(event='m3_decision',**decision));requests[1]['m3_timing']=dict(decision);save()
    assert audit(tmp_path)['passed']
    for record in (events[-1],requests[1]['m3_timing']):record.update(deadline=51.,latest_start=50.,reason='buffer')
    save()
    assert 'm3_signal_deadline_and_hard_bounds' in [f['check'] for f in audit(tmp_path)['failures']]


@pytest.mark.parametrize('recovery',[False,True])
def test_dynamic_rtc_prefix_bound_and_copied_recovery_plan(tmp_path,recovery):
    requests,events,save=fixture(tmp_path)
    meta=json.loads((tmp_path/'run.json').read_text())
    meta['config_snapshot'].update(m3_scheduler_mode='deadline_dvfs' if recovery else 'deadline_high',
        controlled_delay_steps=20,m3_timing={'maximum_action_offset':25,'maximum_observation_age_seconds':None})
    (tmp_path/'run.json').write_text(json.dumps(meta))
    requests[1]['rtc_solver_inputs']['delay']=20
    requests[1]['rtc_guidance']['delay_estimate_steps']=20
    step=4 if recovery else 5
    guard=3 if recovery else 2
    decision=dict(step=step,start_now=True,inflight=False,deadline=100.,
        latest_start=50.+(5-step-guard)*.05,service_seconds=.9,margin_seconds=.1,
        wait_seconds=0.,reason='buffer',decision_monotonic=50.,
        estimated_control_period_seconds=.05,rtc_delay_steps=20,rtc_latest_submit_step=5,
        rtc_submit_guard_steps=guard,path='low_continue_recovery' if recovery else 'high_frequency',
        start_constraint='rtc_prefix')
    events.append(dict(event='m3_decision',**decision))
    if recovery:events.append(dict(event='m3_decision',**dict(decision,step=5)))
    requests[1]['m3_timing']=dict(decision)
    save()
    assert audit(tmp_path)['passed']
    events[-1]['latest_start']=99.
    save()
    assert not audit(tmp_path)['passed']
    events[-1]['latest_start']=decision['latest_start']
    events[-1]['rtc_submit_guard_steps']=0
    save()
    assert 'm3_rtc_prefix_bound_inputs' in [f['check'] for f in audit(tmp_path)['failures']]


def test_dynamic_refresh_requires_recorded_deadline_instead_of_fixed_counter(tmp_path):
    requests,events,save=fixture(tmp_path)
    meta=json.loads((tmp_path/'run.json').read_text())
    meta['config_snapshot'].update(m3_scheduler_mode='deadline_high',
        m3_timing={'maximum_action_offset':20,'maximum_observation_age_seconds':1000})
    (tmp_path/'run.json').write_text(json.dumps(meta))
    decision=dict(step=5,start_now=True,inflight=False,deadline=51.,latest_start=50.,
                  service_seconds=.9,margin_seconds=.1,wait_seconds=0.,reason='action_event')
    events.append(dict(event='m3_decision',**decision))
    requests[1]['m3_timing']=dict(decision)
    save()
    assert audit(tmp_path)['passed']
    decision['start_now']=False
    events[-1].update(decision)
    save()
    result=audit(tmp_path)
    assert not result['passed']
    assert 'm3_request_triggered_by_recorded_deadline' in [f['check'] for f in result['failures']]


@pytest.mark.parametrize('fast_mode',['wrist','pool2'])
@pytest.mark.parametrize('fast_no_state',[False,True])
def test_fresh_visual_audit_checks_current_source_geometry_and_mode_history(tmp_path,fast_mode,fast_no_state):
    requests,events,save=fixture(tmp_path)
    metadata=json.loads((tmp_path/'run.json').read_text())
    metadata['config_snapshot'].update(rtc_paper_schedule=True,m2_visual_alternation=True,
        m2_visual_fast_enabled=True,m2_visual_fast_mode=fast_mode,m2_split_delay=True,m2_delay_modes=['full',fast_mode],
        rtc_initial_delays_by_mode={'full':9,fast_mode:5},m2_fast_no_state=fast_no_state)
    (tmp_path/'run.json').write_text(json.dumps(metadata))
    for i,r in enumerate(requests):
        mode=fast_mode if i else 'full'
        r['planned_context_mode']=mode
        r['rtc_guidance']['state_in_prompt']=not (i and fast_no_state)
        r['context_refresh']=dict(mode=mode,current_images_encoded=True,current_state_encoded=not (i and fast_no_state),
            vision_pool_factor=2 if mode=='pool2' else 1,
            source_prediction_id=i,source_observation_step=r['observation_step'],
            num_views=1 if i and fast_mode=='wrist' else 2,
            visual_tokens=(128 if fast_mode=='pool2' else 256) if i else 512,
            camera_indices=[1] if i and fast_mode=='wrist' else [0,1])
    requests[1]['rtc_guidance'].update(delay_history_steps=[5],delay_history_mode=fast_mode,
                                      delay_estimate_mode='mode_history_max')
    save()
    assert audit(tmp_path)['passed']
    requests[1]['context_refresh']['source_observation_step']=0
    save()
    result=audit(tmp_path)
    assert not result['passed']
    assert 'fresh_visual_source' in [r['check'] for r in result['failures']]


def test_split_history_audit_rejects_full_delay_used_for_expert(tmp_path):
    requests,events,save=fixture(tmp_path)
    metadata=json.loads((tmp_path/'run.json').read_text())
    cfg=metadata['config_snapshot']
    cfg.update(rtc_paper_schedule=True,m2_expert_recompute=True,m2_reuse_enabled=True,
        m2_split_delay=True,rtc_initial_delays_by_mode={'full':9,'expert_only':5})
    (tmp_path/'run.json').write_text(json.dumps(metadata))
    for i,r in enumerate(requests):
        mode='expert_only' if i else 'full'
        r['planned_kv_mode']=mode
        r['kv_recompute']=dict(mode=mode,source_prediction_id=0,source_raw_state=[0]*8,
            source_observation_step=0,nominal_expiry_step=25,
            current_images_encoded=not bool(i),current_state_encoded=not bool(i))
        r['rtc_guidance']['valid_prefix_length']=657
    guidance=requests[1]['rtc_guidance']
    guidance.update(delay_history_steps=[5],delay_history_mode='expert_only',
        delay_estimate_mode='mode_history_max')
    save()
    assert audit(tmp_path)['passed']
    guidance.update(delay_history_steps=[9],delay_estimate_steps=9)
    requests[1]['rtc_solver_inputs']['delay']=9
    save()
    result=audit(tmp_path)
    assert not result['passed']
    assert 'independent_history_max' in [x['check'] for x in result['failures']]


@pytest.mark.parametrize('change,expected',[
    ('state','state_normalization'),('target','old_target_alignment'),
    ('decode','action_postprocessing'),('source','action_is_latest_committed_chunk'),
    ('crop','exact_absolute_crop')])
def test_system_audit_rejects_injected_mistakes(tmp_path,change,expected):
    requests,events,save=fixture(tmp_path)
    if change=='state':requests[1]['model_state'][0]=10
    if change=='target':requests[1]['rtc_solver_inputs']['previous_raw'][0][0]=10
    if change=='decode':requests[1]['actions'][0][0]=10
    if change=='source':events[-1]['source']['offset']=0
    if change=='crop':events[1]['expired_new_actions']=4
    save()
    report=audit(tmp_path)
    assert not report['passed']
    assert expected in [x['check'] for x in report['failures']]


def test_dynamic_audit_accepts_disabled_age_cutoff_with_finite_chunk_limit(tmp_path):
    requests,events,save=fixture(tmp_path)
    meta=json.loads((tmp_path/'run.json').read_text())
    meta['config_snapshot'].update(m3_scheduler_mode='deadline_high',
        m3_timing={'maximum_action_offset':25,'maximum_observation_age_seconds':None})
    (tmp_path/'run.json').write_text(json.dumps(meta))
    decision=dict(step=5,start_now=True,inflight=False,deadline=51.,latest_start=50.,
                  service_seconds=.9,margin_seconds=.1,wait_seconds=0.,reason='buffer')
    events.append(dict(event='m3_decision',**decision));requests[1]['m3_timing']=dict(decision);save()
    assert audit(tmp_path)['passed']


def test_m3_fast_guard_uses_both_modes_and_choice_is_bound_to_trace(tmp_path):
    requests,events,save=fixture(tmp_path)
    meta=json.loads((tmp_path/'run.json').read_text())
    meta['config_snapshot'].update(m3_fast_enabled=True,m4_enabled=False,
        m3_action_events_enabled=True,m3_dvac_enabled=False,
        m2_visual_alternation=True,m2_fast_no_state=True,m2_visual_fast_mode='pool2_agent',
        m2_visual_fast_enabled=False,m2_split_delay=True,m2_delay_modes=['full','pool2_agent'],
        rtc_initial_delays_by_mode={'full':5,'pool2_agent':20},
        m3_scheduler_mode='deadline_high',
        m3_timing={'maximum_action_offset':25,'maximum_observation_age_seconds':None})
    (tmp_path/'run.json').write_text(json.dumps(meta))
    decision=dict(step=5,start_now=True,inflight=False,deadline=100.,latest_start=49.9,
        service_seconds=.9,margin_seconds=.1,wait_seconds=0.,reason='action_event',
        decision_monotonic=50.,estimated_control_period_seconds=.05,
        rtc_delay_steps=20,rtc_latest_submit_step=5,rtc_submit_guard_steps=2,
        path='high_frequency',start_constraint='rtc_prefix')
    events.append(dict(event='m3_decision',**decision))
    requests[1]['m3_timing']=dict(decision)
    for r in requests:
        r['m3_fast_choice']={'mode':'Full'}
        r['m3_fast_decision']=({k:v for k,v in decision.items() if k!='step'} if r['prediction_id'] else {})
        r['rtc_guidance']['state_in_prompt']=True
        r['context_refresh']=dict(mode='full',current_images_encoded=True,current_state_encoded=True,
            source_observation_step=r['observation_step'],source_prediction_id=r['prediction_id'],
            num_views=2,visual_tokens=512,camera_indices=[0,1],vision_pool_factor=1)
    save();assert audit(tmp_path)['passed']
    requests[1]['m3_fast_decision']['service_seconds']=.2;save()
    assert 'm3_fast_decision_bound_to_trace' in [f['check'] for f in audit(tmp_path)['failures']]


def test_same_control_step_uses_later_commit_after_critical_wait():
    from scripts.audit_rtc_system import latest_commit_at_step
    early=dict(prediction_id=1,commit_step=43,commit_monotonic=10.)
    late=dict(prediction_id=2,commit_step=43,commit_monotonic=10.5)
    future=dict(prediction_id=3,commit_step=90,commit_monotonic=20.)
    for records in ([early,late,future],[late,future,early]):
        assert latest_commit_at_step(records,43)==late
        assert latest_commit_at_step(records,86)==late
        assert latest_commit_at_step(records,90)==future


def test_unguided_ablation_audit_requires_no_target_and_no_vjp(tmp_path):
    requests,events,save=fixture(tmp_path)
    meta=json.loads((tmp_path/'run.json').read_text());meta['config_snapshot']['rtc_disable_guidance']=True
    (tmp_path/'run.json').write_text(json.dumps(meta))
    requests[1]['rtc_solver_inputs']['previous_raw']=None
    requests[1]['rtc_guidance'].update(guidance_enabled=False,vjp_norms=[],solver_backend='original_openpi_eager_unguided')
    save();assert audit(tmp_path)['passed']
    requests[1]['rtc_guidance']['vjp_norms']=[1.]
    save();assert not audit(tmp_path)['passed']
