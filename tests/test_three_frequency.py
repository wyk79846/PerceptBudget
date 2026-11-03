import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.three_frequency import select_full,ThreeFullRuntime
from perceptbudget.device_clocks import runtime_trace


def decision(end=12.):
    return dict(decision_monotonic=10.,deadline=end,margin_seconds=.025,
        estimated_control_period_seconds=.05,service_seconds=.55,latest_start=end-.575)


@pytest.mark.parametrize('profile',['baseline','gpu_emc','deep_idle'])
def test_medium_is_preferred_even_when_high_is_faster(profile):
    f=select_full(decision(),medium_seconds=.75,medium_cap=20.,current_profile=profile,
                  transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018))
    assert f['selected']['profile']=='gpu_emc'
    assert f['candidates'][1]['feasible']
    assert f['selection_order']==['gpu_emc','baseline']


def test_high_only_when_medium_cannot_meet_deadline():
    args=dict(medium_seconds=.75,medium_cap=20.,current_profile='deep_idle',
              transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018))
    f=select_full(decision(10.7),**args)
    assert not f['candidates'][0]['feasible'] and f['selected']['profile']=='baseline'
    assert f['reason']=='high_medium_too_late'
    assert select_full(decision(10.3),**args)['reason']=='high_emergency'


def test_medium_rtc_prefix_and_first_refresh_caps_are_respected():
    args=dict(medium_seconds=.75,medium_cap=9.,current_profile='deep_idle',
              transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018))
    assert select_full(decision(),**args)['selected']['profile']=='baseline'
    d=decision();d['first_refresh_cap']={'latest_start':9.}
    f=select_full(d,**dict(args,medium_cap=20.))
    assert not any(r['feasible'] for r in f['candidates'])


def test_first_refresh_reserves_ack_tick_even_when_starting_from_high():
    d=decision();d['first_refresh_cap']=dict(latest_start=10.5,recovery_ticks=0)
    f=select_full(d,medium_seconds=.75,medium_cap=20.,current_profile='baseline',
                  transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018))
    assert f['candidates'][0]['first_cap']==pytest.approx(10.45)
    assert f['candidates'][1]['first_cap']==pytest.approx(10.5)


def config():
    costs=dict(cells=[dict(profile=p,from_profile='gpu_emc',history=h,
        service_samples_seconds=[.5 if p=='baseline' else .7]*20)
        for p in ('baseline','gpu_emc') for h in ('hot','short','long')])
    return dict(H=50,m3_low_full_costs=costs,m3_scheduler_mode='deadline_dvfs',
        m3_three_frequency_transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018),
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,maximum_action_offset=50,
            maximum_observation_age_seconds=None,minimum_low_idle_seconds=.2),
        m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)


def test_no_inference_in_deep_idle_and_no_switch_during_compute():
    r=ThreeFullRuntime(config())
    changes=[]
    r.clocks=SimpleNamespace(profile='deep_idle',pending=None,poll=lambda:None,
                            switch=changes.append,last_low=9.,last_change=None)
    with pytest.raises(RuntimeError,match='Cannot infer'):r.requested({'observation_step':30})
    d=dict(start_now=True,compute_profile='gpu_emc')
    assert not r.control_clocks(d,inflight=False)['start_now']
    assert changes==['gpu_emc'] and r.recovery_target=='gpu_emc'
    assert not r.control_clocks(d,inflight=False)['start_now']
    r.clocks.profile='gpu_emc'
    assert r.control_clocks(d,inflight=False)['start_now']
    r.active_profile='gpu_emc'
    assert r.control_clocks(d,inflight=True)==d
    r.clocks.profile='deep_idle'
    with pytest.raises(RuntimeError):r.control_clocks(d,inflight=True)


def test_wait_can_enter_low_but_short_gap_does_not_switch():
    r=ThreeFullRuntime(config());changes=[]
    r.clocks=SimpleNamespace(profile='gpu_emc',pending=None,poll=lambda:None,switch=changes.append)
    d=dict(start_now=False,latest_start=10.1,decision_monotonic=10.)
    r.control_clocks(d,inflight=False);assert changes==[]
    d['latest_start']=11.
    r.control_clocks(d,inflight=False);assert changes==['deep_idle']


def test_three_state_runtime_protocol_and_acknowledgement():
    read,write=os.pipe();events=[]
    tokens={b'HIGH':'baseline',b'MID':'gpu_emc',b'IDLE':'deep_idle'}
    payloads=iter([b'MID\n',b'HIGH\n',b'STOP\n'])
    def emit(e):
        events.append(e)
        if e['status']=='APPLIED':os.write(write,next(payloads))
    try:
        os.write(write,b'IDLE\n')
        with patch('perceptbudget.device_clocks.observed_switch',return_value={}):
            runtime_trace(SimpleNamespace(actual=lambda:{}),emit,read,seconds=2,tokens=tokens)
    finally:os.close(write);os.close(read)
    assert [e['profile'] for e in events if e['status']=='APPLIED']==['deep_idle','gpu_emc','baseline']


def test_runtime_decision_does_not_modify_buffer_or_event_deadline():
    r=ThreeFullRuntime(config())
    r.clocks=SimpleNamespace(profile='deep_idle',pending=None,last_low=9.,poll=lambda:None)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=9.,actions=actions),
                dict(commit_step=0,commit_monotonic=9.5))
    q={i:(a,0,0,i) for i,a in enumerate(actions)}
    d=r.decide(now=10.,step=10,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['compute_profile']=='gpu_emc'
    assert d['deadline']==12.  #40 remaining actions *50ms
    assert len(q)==50 and d['three_frequency_formula']['policy']=='medium_first_v1'
