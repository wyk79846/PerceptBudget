import pytest
from perceptbudget.closed_loop import DeadlineRuntime
from perceptbudget.deadline_controller import timing_plan
from test_m3_service_choice import fixture


def runtime(*,response=True,start=105,stop=144,mode='Fast'):
    costs,_=fixture()
    for kind in ('Full','Fast'):
        for cell in costs['service'][kind].values():cell['samples_seconds']=[.55 if kind=='Full' else .35]
    cfg=dict(m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,maximum_action_offset=50,maximum_observation_age_seconds=None),
        m3_events={},m3_service_samples=dict(hot=[.55],short=[.65],long=[.65]),
        rtc_initial_control_period_seconds=.05,H=50,m3_scheduler_mode='deadline_high',
        m3_fast_enabled=True,m3_fast_advance=True,m3_fast_key_phases=True,m3_fast_cost_model=costs)
    if response:cfg['m3_key_phase_response_policy']='full_once_v1'
    r=DeadlineRuntime(cfg);r.fast_last_mode=mode
    r.chunks={3:dict(origin=102,observation=5.1,dvac_step=None,context_mode=mode,
        key_phases=[dict(start_step=start,stop_step=stop)])}
    r.events=[dict(kind='gripper_open',target_step=120)]
    return r


def decide(r):
    queue={i:([0.]*7,3,0,i-102) for i in range(110,152)}
    return r.decide(now=5.5,step=110,queue=queue,rows=[],inflight=False,rtc_delay_steps=11)


def test_recorded110_gap_is_fixed_even_when_point_event_is_excluded():
    old=decide(runtime(response=False));new=decide(runtime())
    assert old['reason']=='buffer' and not old['start_now']
    assert old['latest_start']==pytest.approx(7.)  #140th tick
    assert new['reason']=='key_phase' and new['start_now']
    assert new['critical_step'] is None
    assert new['fast_ignored_events'][0]['target_step']==120
    assert new['deadline']==pytest.approx(5.5)
    assert new['key_phase_response']==dict(prediction_id=3,start_step=105,stop_step=144,target_step=110,state='active')
    assert new['m3_fast_formula']['mode']=='Full'


def test_upcoming_phase_subtracts_complete_full_budget_once():
    p=decide(runtime(start=130))
    assert p['reason']=='key_phase' and not p['start_now']
    assert p['deadline']==pytest.approx(6.5)
    assert p['latest_start']==pytest.approx(6.5-.55-.025)
    assert p['key_phase_response']['state']=='upcoming'


def test_full_and_expired_phase_do_not_trigger_response():
    for r in [runtime(mode='Full'),runtime(stop=110)]:
        p=decide(r)
        assert p['key_phase_response'] is None
        assert p['reason']=='buffer' and not p['start_now']


def test_completed_full_records_response_and_does_not_retrigger():
    r=runtime();plan=decide(r)
    request=dict(prediction_id=4,observation_step=110,observation_monotonic=5.5,
                 m3_fast_choice=dict(mode='Full'),m3_timing=plan,actions=[[0.,0.,0.,0.,0.,0.,-1.]]*50)
    commit=dict(commit_step=121,commit_monotonic=6.05)
    r.committed(request,commit)
    assert request['m3_timing']['key_phase_response_completed']==dict(prediction_id=4,**commit)
    assert r.chunks[4]['context_mode']=='Full'
    # Even if the new Full independently predicts the same phase, no repeated
    # urgent response is generated from a Full chunk.
    r.chunks[4]['key_phases']=[dict(start_step=105,stop_step=144)]
    queue={i:([0.]*7,4,0,i-110) for i in range(121,160)}
    p=r.decide(now=6.05,step=121,queue=queue,rows=[],inflight=False,rtc_delay_steps=11)
    assert p['key_phase_response'] is None and not p['start_now']


def test_phase_cap_is_hard_even_under_latest_signal_policy():
    p=timing_plan(now=0.,step=0,buffer_end_step=50,period=.05,service_seconds=.5,
        critical_step=30,dvac_step=40,key_phase_step=20,signal_policy='latest')
    assert p.reason=='key_phase' and p.deadline==1.


def test_completion_audit_allows_terminal_drain_without_faking_commit():
    from scripts.audit_rtc_system import phase_completion_matches
    request=dict(prediction_id=4,m3_fast_choice=dict(mode='Full'),m3_timing={})
    assert phase_completion_matches(request,None)
    commit=dict(commit_step=121,commit_monotonic=6.05)
    assert not phase_completion_matches(request,commit)
    request['m3_timing']['key_phase_response_completed']=dict(prediction_id=4,**commit)
    assert phase_completion_matches(request,commit)
    assert not phase_completion_matches(request,None)
