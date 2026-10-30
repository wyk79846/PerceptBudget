from copy import deepcopy
import pytest

from perceptbudget.m4_acceptance import configure_m3, configure_m3_fast
from perceptbudget.closed_loop import DeadlineRuntime
from perceptbudget.deadline_controller import FullServiceTimes


def test_fast_setup_preserves_m3_deadline_and_device_rules():
    a,b={},{}
    sa,sb=dict(timing={}),dict(timing={})
    configure_m3(a,sa);configure_m3_fast(b,sb)
    assert sa==sb
    for field in ('m3_action_events_enabled','m3_dvac_enabled','m3_dvfs_policy',
                  'm3_scheduler_mode','m3_event_coverage_mode','m3_hold_high_clocks'):
        assert a[field]==b[field]
    assert b['m4_enabled'] is False and b['m3_fast_enabled'] is True
    assert b['m2_fast_no_state'] and b['m4_fast_prompt_capacity']==20
    assert 'm4_settings' not in b
    configure_m3(b,sb)
    assert b['m3_fast_enabled'] is False


def test_fast_completion_cannot_shrink_full_service_history():
    runtime=object.__new__(DeadlineRuntime)
    runtime.chunks={};runtime.dvac_enabled=False;runtime.action_events_enabled=False
    runtime.merge_gap=None;runtime.open_probe=None;runtime.event_coverage_mode='event_only'
    runtime.settings={'margin_updates':False};runtime.fast_selection=True
    runtime.times=FullServiceTimes({k:[.5] for k in ('hot','short','long')})
    runtime.fast_times=FullServiceTimes({k:[.3] for k in ('hot','short','long')})
    request=dict(prediction_id=1,observation_step=10,observation_monotonic=1.,
        actions=[],m3_fast_choice={'mode':'Fast'},
        m3_timing=dict(deadline=3.,critical_step=None,reason='buffer',history='hot'))
    commit=dict(commit_monotonic=1.2,commit_step=14)
    before=deepcopy(runtime.times.samples)
    runtime.committed(request,commit)
    assert runtime.times.samples==before
    assert list(runtime.fast_times.samples['hot'])==pytest.approx([.3,.2])
    request['m3_fast_choice']['mode']='Full'
    runtime.committed(request,dict(commit,commit_monotonic=1.6))
    assert list(runtime.times.samples['hot'])==pytest.approx([.5,.6])


def test_runtime_applies_selected_mode_latency_before_request_is_due():
    costs=dict(waits={'baseline':[5.]*4,'gpu_emc':[2.]*4},service={
        mode:{h:dict(energy_j=energy,mean_seconds=seconds,samples_seconds=[seconds],
                     from_profile='baseline' if h=='hot' else 'gpu_emc')
              for h in ('hot','short','long')}
        for mode,energy,seconds in [('Full',10.,.5),('Fast',5.,.3)]})
    timing=dict(service_quantile=.95,safety_margin_seconds=.025,
        maximum_action_offset=50,maximum_observation_age_seconds=None)
    cfg=dict(m3_timing=timing,m3_events={},m3_service_samples={h:[.5] for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05,H=50,m3_scheduler_mode='deadline_high',
        m3_fast_enabled=True,m3_fast_cost_model=costs,m3_action_events_enabled=True)
    runtime=DeadlineRuntime(cfg)
    runtime.chunks={0:dict(origin=0,observation=0.,dvac_step=None)}
    queue={i:([0.]*7,0,0,i) for i in range(5,50)}
    plan=runtime.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,rtc_delay_steps=8)
    assert plan['m3_fast_formula']['mode']=='Fast'
    assert plan['service_seconds']==pytest.approx(.3)
    assert plan['latest_start']==pytest.approx(2.05)  # RTC step cap, earlier than Fast's2.175.
    assert plan['start_constraint']=='rtc_prefix'
    runtime.fast_last_mode='Fast'
    full=runtime.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,rtc_delay_steps=8)
    assert full['m3_fast_formula']['mode']=='Full'
    assert full['service_seconds']==pytest.approx(.5)
    assert full['latest_start']==pytest.approx(1.975)
    runtime.fast_last_mode='Full'
    runtime.events=[dict(kind='gripper_close',target_step=25)]
    event=runtime.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,rtc_delay_steps=8)
    assert event['reason']=='action_event' and event['m3_fast_formula']['mode']=='Full'


def test_execution_window_is_frozen_and_does_not_swallow_future_events():
    costs=dict(waits={'baseline':[5.]*4,'gpu_emc':[2.]*4},service={
        mode:{h:dict(energy_j=energy,mean_seconds=seconds,samples_seconds=[seconds],
                     from_profile='baseline' if h=='hot' else 'gpu_emc')
              for h in ('hot','short','long')}
        for mode,energy,seconds in [('Full',10.,.5),('Fast',5.,.3)]})
    cfg=dict(m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
        maximum_action_offset=50,maximum_observation_age_seconds=None),
        m3_events={},m3_service_samples={h:[.5] for h in ('hot','short','long')},
        m3_recovery_samples=[.65],rtc_initial_control_period_seconds=.05,H=50,
        m3_scheduler_mode='deadline_high',m3_fast_enabled=True,m3_fast_cost_model=costs,
        m3_action_events_enabled=True)
    runtime=DeadlineRuntime(cfg)
    runtime.chunks={0:dict(origin=0,observation=0.,dvac_step=None)}
    runtime.events=[dict(kind='gripper_close',target_step=8),dict(kind='turn',target_step=25)]
    queue={i:([0.]*7,0,0,i) for i in range(5,50)}
    first=runtime.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,rtc_delay_steps=8)
    assert first['fast_execution_window']['end_step']==19
    assert first['critical_step']==25
    assert [e['target_step'] for e in first['fast_ignored_events']]==[8]
    later=runtime.decide(now=1.,step=20,queue=queue,rows=[],inflight=False,rtc_delay_steps=8)
    assert later['fast_execution_window']==first['fast_execution_window']
    assert later['critical_step']==25  # A sliding window would wrongly exclude25.
    assert later['m3_fast_formula']['mode']=='Full'
    # When only committed-prefix events exist, ordinary refresh may use Fast.
    runtime.events=[dict(kind='gripper_close',target_step=8)]
    ordinary=runtime.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,rtc_delay_steps=8)
    assert ordinary['critical_step'] is None and ordinary['m3_fast_formula']['mode']=='Fast'
    runtime.fast_last_mode='Fast'
    assert runtime.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,
        rtc_delay_steps=8)['m3_fast_formula']['mode']=='Full'
