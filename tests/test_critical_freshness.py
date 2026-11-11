from types import SimpleNamespace

import pytest

from perceptbudget.critical_freshness import propose
from perceptbudget.three_frequency import ThreeFullRuntime
from tests.test_three_frequency import config


def proposal(**changes):
    args=dict(events=[dict(kind='gripper_close',target_step=35)],candidates=[],
        phases=[dict(start_step=10,stop_step=50,kinds=['gripper_close'],event_source='remaining_v2')],
        covered=[],used=[],ordinary_turns=set(),observation_step=10,
        step=20,end=60,prediction_id=2,maximum_age=16,tolerance=5)
    args.update(changes)
    return propose(**args)


def test_old_covered_event_can_be_renewed_but_new_events_keep_original_path():
    p=proposal()
    assert p['source_age_steps']==25 and p['target_step']==35
    assert proposal(phases=[]) is None
    assert proposal(observation_step=19) is None
    assert proposal(candidates=[dict(kind='slowdown',target_step=30)]) is None
    assert proposal(step=35) is None


def test_one_observation_per_original_typed_phase_and_no_ordinary_turn_revival():
    p=proposal()
    assert proposal(used=[p]) is None
    assert proposal(used=[dict(p,kind='gripper_open')]) is not None
    assert proposal(events=[dict(kind='turn',target_step=35)],
        phases=[dict(start_step=10,stop_step=50,kinds=['turn'])],ordinary_turns={35}) is None


def runtime(event_step=35, *, enabled=True, wait_steps=0, correction_steps=0):
    c=config()
    c.update(m3_critical_high=True,m3_phase_refresh=True,
        m3_critical_forecast='remaining_v1',m3_phase_event_source='remaining_v2')
    r=ThreeFullRuntime(c)
    r.clocks=SimpleNamespace(profile='baseline',pending=None,last_low=None,poll=lambda:None)
    r.critical_freshness_age=16 if enabled else None
    r.critical_freshness_wait_steps=wait_steps
    r.critical_freshness_correction_steps=correction_steps
    actions=[[.4,0.,0.,0.,0.,0.,-1.]]*50
    actions=[a[:6]+[1. if i+10>=event_step else -1.] for i,a in enumerate(actions)]
    r.committed(dict(prediction_id=0,observation_step=10,observation_monotonic=.5,actions=actions),
        dict(commit_step=20,commit_monotonic=1.))
    r.refresh_phase=dict(start_step=10,stop_step=60,kinds=['gripper_close'],event_source='remaining_v2')
    queue={10+i:(a,0,10,i) for i,a in enumerate(actions)}
    return r,queue


def test_request_starts_near_event_using_selected_high_service_and_only_once():
    original,q=runtime(enabled=False)
    assert original.decide(now=1.,step=20,queue=q,rows=[],inflight=False,rtc_delay_steps=10)['reason']=='buffer'
    r,q=runtime()
    d=r.decide(now=1.,step=20,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['critical_step']==35 and d['compute_profile']=='baseline'
    assert d['latest_start']==pytest.approx(1.225)  # step24.5, not the medium anchor
    assert not d['start_now'] and 'critical_high_prepare_anchor' not in d
    r.requested(dict(observation_step=24,observation_monotonic=1.2))
    assert len(r.critical_freshness_used)==1
    d=r.decide(now=1.25,step=25,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert 'critical_freshness' not in d


def test_late_optional_refresh_is_dropped_instead_of_creating_a_wait():
    r,q=runtime(event_step=28)
    d=r.decide(now=1.,step=20,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert 'critical_freshness' not in d and d['reason']=='buffer'
    assert d['critical_freshness_skipped']['reason']=='selected_service_too_late'
    assert not r.critical_freshness_used


def test_inflight_request_is_not_overlapped():
    r,q=runtime()
    d=r.decide(now=1.,step=20,queue=q,rows=[],inflight=True,rtc_delay_steps=10)
    assert 'critical_freshness' not in d and not r.critical_freshness_used


def test_small_expected_lateness_can_wait_at_the_event_without_changing_v4():
    r,q=runtime(event_step=29,wait_steps=2)
    d=r.decide(now=1.,step=20,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    p=d['critical_freshness']
    assert d['start_now'] and d['compute_profile']=='baseline'
    assert 0<p['estimated_wait_seconds']<=.1
    assert p['maximum_wait_steps']==2 and d['critical_step']==29
    assert d['deadline_already_unreachable']  # Report lateness honestly.
    r.requested(dict(observation_step=20,observation_monotonic=1.))
    assert len(r.critical_freshness_used)==1
    v4,q=runtime(event_step=29)
    assert 'critical_freshness' not in v4.decide(now=1.,step=20,queue=q,rows=[],inflight=False,rtc_delay_steps=10)


def test_more_than_two_estimated_late_steps_do_not_create_extra_refresh():
    r,q=runtime(event_step=27,wait_steps=2)
    d=r.decide(now=1.,step=20,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert 'critical_freshness' not in d
    assert d['critical_freshness_skipped']['reason']=='selected_service_too_late'


def test_supplement_wait_does_not_recommit_actions_past_a_previously_waited_event():
    from perceptbudget.critical_wait import PhaseWait,rtc_prefix_plan
    from perceptbudget.action_events import EventConfig
    guard=PhaseWait()
    guard.complete(dict(phase=dict(start_step=10,stop_step=50),kinds=['gripper_close'],event_source='remaining_v2'))
    request=dict(m3_timing=dict(reason='action_event',critical_step=29,
        candidate_event_kinds=['gripper_close'],
        critical_freshness=dict(wait_policy='critical_freshness_short_wait_v1',estimated_wait_seconds=.05)))
    review=guard.review(request,[],10,EventConfig())
    assert review['wait'] and review['reason']=='critical_freshness_short_wait'
    assert rtc_prefix_plan(request,20,11,review)['selected_delay_steps']==9
    assert guard.completed['stop_step']==50
    del request['m3_timing']['critical_freshness']
    assert not guard.review(request,[],10,EventConfig())['wait']


@pytest.mark.parametrize('wait',[True,-1,3,1.5])
def test_wait_budget_must_be_an_explicit_small_integer(wait):
    from perceptbudget.action_signal_experiment import load_settings,validate
    from pathlib import Path
    settings=load_settings(Path(__file__).resolve().parents[1]/'action_signals_freshness.json')
    with pytest.raises(ValueError):validate(dict(settings,critical_freshness_max_wait_steps=wait))


def test_only_stale_grasp_supplement_leaves_two_uncommitted_correction_actions():
    from perceptbudget.critical_wait import PhaseWait,rtc_prefix_plan,wait_target
    from perceptbudget.action_events import EventConfig
    r,q=runtime(event_step=29,wait_steps=2,correction_steps=2)
    d=r.decide(now=1.,step=20,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    p=d['critical_freshness']
    assert d['critical_step']==29 and p['pause_step']==27
    assert p['correction_steps']==2 and p['estimated_pause_seconds']>.1
    request=dict(m3_timing=d)
    assert wait_target(request)==27
    review=PhaseWait().review(request,[[.4,0,0,0,0,0,-1.]]*50,10,EventConfig())
    assert review['wait'] and review['reason']=='critical_freshness_approach_wait'
    assert rtc_prefix_plan(request,20,10,review)['selected_delay_steps']==7


def test_zero_correction_keeps_v4_physical_event_as_the_pause_boundary():
    from perceptbudget.critical_wait import wait_target
    r,q=runtime()
    d=r.decide(now=1.,step=20,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert 'pause_policy' not in d['critical_freshness']
    assert wait_target(dict(m3_timing=d))==35
