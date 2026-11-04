"""New event evidence must drive both refresh coverage and actual waiting."""
import math

import pytest

from perceptbudget.action_events import EventConfig
from perceptbudget.critical_forecast import forecast
from perceptbudget.key_phases import absolute_intervals,intervals_for_timing


def turning_actions():
    speeds=(1.,1.,.8,.5,.2,.2,.2,.5,.8,1.,1.)
    angles=(0.,15.,30.,45.,60.,75.,90.,90.,90.,90.,90.)
    return [[v*math.cos(math.radians(a)),v*math.sin(math.radians(a)),0.,0.,0.,0.,-1.]
            for v,a in zip(speeds,angles)]


def timing(actions,origin=100,consumed=2):
    at=origin+consumed
    result=forecast(actions[consumed:],previous_actions=actions[max(0,consumed-2):consumed],
        current_step=at,now=1.,control_period=.05)
    return dict(phase_event_source='remaining_v1',
        critical_forecast=dict(version='remaining_v1',source_step=at,result=result),
        reason='action_event',critical_step=result['critical_step'],
        candidate_event_kinds=result['event_kinds'])


def test_gradual_turn_anchors_the_slow_region_with_absolute_step_alignment():
    actions=turning_actions();t=timing(actions)
    assert absolute_intervals(actions,100)==[]
    phases=intervals_for_timing(actions,100,EventConfig(),t)
    assert len(phases)==1
    assert phases[0]['start_step']==100 and phases[0]['stop_step']==110
    assert phases[0]['end_observed']
    assert phases[0]['anchors']==[dict(kind='turn',offset=5)]
    assert t['critical_step']==103  # slowdown before the gradual turn


def test_legacy_phase_behavior_is_unchanged_without_the_explicit_source():
    actions=turning_actions()
    assert intervals_for_timing(actions,100,EventConfig(),{})==absolute_intervals(actions,100)


def test_consumed_grip_anchor_is_not_reintroduced_when_new_forecast_is_empty():
    actions=[[v,0.,0.,0.,0.,0.,-1. if i<4 else 1.]
             for i,v in enumerate((1.,1.,.8,.5,.2,.2,.2,.5,.8,1.,1.))]
    assert absolute_intervals(actions,100)
    t=timing(actions,consumed=6)
    assert t['critical_forecast']['result']['events']==[]
    assert intervals_for_timing(actions,100,EventConfig(),t)==[]


def test_waiting_uses_the_same_phase_as_the_new_refresh_decision():
    from perceptbudget.critical_wait import PhaseWait
    actions=turning_actions();t=timing(actions)
    review=PhaseWait().review(dict(m3_timing=t),actions,100,EventConfig())
    assert review['reason']=='new_phase'
    assert review['phase']==intervals_for_timing(actions,100,EventConfig(),t)[0]
    guard=PhaseWait();guard.complete(review)
    next_t=dict(t,critical_step=107,candidate_event_kinds=['turn'])
    assert not guard.review(dict(m3_timing=next_t),actions,100,EventConfig())['wait']
    next_t['candidate_event_kinds']=['gripper_open']
    # Turn-only coverage retains the existing opposite-grip behavior; a close
    # phase must never suppress a later release.
    guard.complete(dict(phase=review['phase'],kinds=['gripper_close']))
    assert guard.review(dict(m3_timing=next_t),actions,100,EventConfig())['wait']


def test_runtime_reuses_one_forecast_for_event_target_phase_and_later_decisions():
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_critical_forecast='remaining_v1',m3_phase_event_source='remaining_v1',
        m3_phase_refresh=True,m3_scheduler_mode='deadline_high',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None),m3_events={},
        m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(cfg);actions=turning_actions()
    runtime.committed(dict(prediction_id=1,observation_step=100,observation_monotonic=0.,actions=actions),
        dict(commit_step=102,commit_monotonic=.1))
    queue={100+i:(a,1,100,i) for i,a in enumerate(actions)}
    rows=[dict(step=100+i,action=actions[i],action_start_monotonic=i*.05) for i in (0,1)]
    d=runtime.decide(now=.1,step=102,queue=queue,rows=rows,inflight=False,rtc_delay_steps=2)
    assert d['critical_step']==103 and d['phase_event_source']=='remaining_v1'
    assert d['refresh_phase_response']['stop_step']==110
    assert runtime.chunks[1]['refresh_phases']==intervals_for_timing(actions,100,EventConfig(),d)
    next_d=runtime.decide(now=.15,step=103,queue=queue,
        rows=rows+[dict(step=102,action=actions[2],action_start_monotonic=.1)],
        inflight=True,rtc_delay_steps=2)
    assert next_d['critical_forecast'] is d['critical_forecast']
    assert next_d['refresh_phase_response']['stop_step']==110


def test_explicit_source_does_not_accept_incompatible_runtime():
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_phase_event_source='remaining_v1',m3_scheduler_mode='deadline_high',
        rtc_initial_control_period_seconds=.05,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None),m3_events={},
        m3_service_samples={h:[.5]*20 for h in ('hot','short','long')})
    with pytest.raises(ValueError,match='Shared phase events'):
        DeadlineRuntime(cfg)


def test_typed_coverage_does_not_hide_a_later_slowdown_or_grip_after_a_turn():
    from perceptbudget.critical_wait import covered_by_phase,PhaseWait
    phase=dict(start_step=38,stop_step=60,kinds=['turn'],event_source='remaining_v2')
    assert covered_by_phase(dict(target_step=49,kind='turn'),phase)
    assert not covered_by_phase(dict(target_step=47,kind='slowdown'),phase)
    assert not covered_by_phase(dict(target_step=52,kind='gripper_close'),phase)
    guard=PhaseWait();guard.complete(dict(phase=phase,kinds=['turn'],event_source='remaining_v2'))
    request=dict(m3_timing=dict(reason='action_event',critical_step=47,
        candidate_event_kinds=['slowdown']))
    assert guard.review(request,[],35,EventConfig())['wait']
    request['m3_timing']['candidate_event_kinds']=['turn']
    assert not guard.review(request,[],35,EventConfig())['wait']
    request['m3_timing']['candidate_event_kinds']=['turn','slowdown']
    assert guard.review(request,[],35,EventConfig())['wait']


def test_v2_uses_the_same_forecast_geometry_but_retains_new_event_kinds():
    actions=turning_actions();t=timing(actions)
    old=intervals_for_timing(actions,100,EventConfig(),t)
    t['phase_event_source']='remaining_v2'
    assert intervals_for_timing(actions,100,EventConfig(),t)==old
    from perceptbudget.critical_wait import PhaseWait
    guard=PhaseWait();review=guard.review(dict(m3_timing=t),actions,100,EventConfig())
    assert review['event_source']=='remaining_v2'
    guard.complete(review)
    assert guard.completed['event_source']=='remaining_v2'
    t['critical_step']=107;t['candidate_event_kinds']=['turn']
    assert guard.review(dict(m3_timing=t),actions,100,EventConfig())['wait']
