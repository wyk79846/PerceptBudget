"""Every observation gets a refresh bound; only gripper hints advance it."""
from types import SimpleNamespace

import pytest

from perceptbudget.closed_loop import DeadlineRuntime
from perceptbudget.three_frequency import ThreeFullRuntime
from test_three_frequency import config


def runtime(*, enabled=True, three=False, origin=30, pid=1):
    cfg=dict(config(),m3_gripper_q30=enabled,m3_phase_refresh=True,m3_first_refresh_step=30)
    r=(ThreeFullRuntime if three else DeadlineRuntime)(cfg)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    r.committed(dict(prediction_id=pid,observation_step=origin,observation_monotonic=origin*.05,
        actions=actions),dict(commit_step=origin+12,commit_monotonic=(origin+12)*.05))
    queue={origin+i:(a,pid,origin,i) for i,a in enumerate(actions)}
    return r,queue


@pytest.mark.parametrize('origin,pid',[(0,0),(30,1),(83,4)])
def test_every_observation_has_its_own30_step_submission_cap(origin,pid):
    r,q=runtime(origin=origin,pid=pid)
    at=origin+30
    d=r.decide(now=at*.05,step=at,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['start_now'] and d['start_constraint']=='refresh_interval'
    assert d['refresh_request_cap']['request_step']==origin+30
    assert d['refresh_request_cap']['source_prediction_id']==pid
    assert d['latest_start']==pytest.approx(at*.05)
    # The commit was12 steps later; the bound must not become origin+42.
    assert d['deadline']>d['latest_start']+d['service_seconds']


def test_gripper_advances_refresh_but_slowdown_and_turn_do_not():
    r,q=runtime(origin=0,pid=0)
    r.events=[dict(kind=k,target_step=s,prediction_id=0) for k,s in
              [('slowdown',14),('turn',16),('gripper_close',20)]]
    d=r.decide(now=.6,step=12,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['critical_step']==20 and d['candidate_event_kinds']==['gripper_close']
    assert d['reason']=='action_event'
    assert d['latest_start']<d['refresh_request_cap']['latest_start']


def test_phase_dedup_preserves_one_close_but_allows_opposite_open():
    r,q=runtime(origin=0,pid=0)
    r.refresh_phase=dict(start_step=10,stop_step=40,kinds=['gripper_close'])
    r.events=[dict(kind=k,target_step=s,prediction_id=0) for k,s in
              [('gripper_close',24),('gripper_open',28)]]
    d=r.decide(now=.6,step=12,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['critical_step']==28 and d['candidate_event_kinds']==['gripper_open']
    assert [e['kind'] for e in d['refresh_phase_suppressed']]==['gripper_close']


def test_periodic_bound_applies_even_when_gripper_is_later():
    r,q=runtime(origin=30,pid=1)
    r.events=[dict(kind='gripper_open',target_step=77,prediction_id=1)]
    d=r.decide(now=3.,step=60,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['critical_step']==77 and d['reason']=='action_event'
    assert d['start_now'] and d['start_constraint']=='refresh_interval'


def test_medium_selection_keeps_periodic_cap_and_reserves_frequency_ack():
    r,q=runtime(three=True)
    r.clocks=SimpleNamespace(profile='baseline',pending=None,last_low=None)
    d=r.decide(now=2.9,step=58,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['compute_profile']=='gpu_emc'
    assert d['refresh_request_cap']['request_step']==60
    assert d['refresh_request_cap']['recovery_ticks']==1
    assert d['latest_start']==pytest.approx(2.95)
    assert d['three_frequency_formula']['selected']['first_cap']==pytest.approx(2.95)


def test_disabled_candidate_keeps_original_events_and_no_periodic_cap():
    r,q=runtime(enabled=False)
    r.events=[dict(kind='slowdown',target_step=75,prediction_id=1)]
    d=r.decide(now=3.,step=60,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['critical_step']==75 and d['candidate_event_kinds']==['slowdown']
    assert 'refresh_request_cap' not in d
