import math

import pytest

from perceptbudget.action_events import EventConfig
from perceptbudget.event_identity import review_ordinary_turn
from scripts.audit_rtc_system import ordinary_turn_from_record


def action(angle=0.,scale=1.,grip=-1.):
    a=math.radians(angle)
    return [scale*math.cos(a),scale*math.sin(a),0.,0.,0.,0.,grip]


def review(actions=None,*,source_step=0,history=None,phases=None,extra=()):
    actions=actions or [action(a) for a in (0,15,30,45,60,75,90,90)]
    event=dict(kind='turn',prediction_id=7,target_step=4)
    events=[event]+list(extra)
    result=review_ordinary_turn(event,actions=actions,origin=0,source_step=source_step,
        history=history or {},phases=phases or [],events=events,config=EventConfig())
    return result,event,actions,events


def test_smooth_isolated_bend_does_not_require_its_own_early_refresh():
    r,*_=review()
    assert r['ordinary'] and r['reason']=='ordinary_bend'
    assert r['local_angle_degrees']==pytest.approx(30.)


@pytest.mark.parametrize('kind,reason',[('gripper_close','near_gripper'),('slowdown','near_slowdown')])
def test_nearby_critical_cues_keep_the_turn(kind,reason):
    r,*_=review(extra=[dict(kind=kind,target_step=6)])
    assert not r['ordinary'] and r['reason']==reason


def test_protected_phase_is_retained_even_if_its_end_is_not_observed():
    r,*_=review(phases=[dict(start_step=2,stop_step=8,end_observed=False)])
    assert not r['ordinary'] and r['reason']=='protected_motion_phase'


def test_abrupt_turn_and_uncertain_direction_remain_actionable():
    r,*_=review([action()]*4+[action(180.)]*4)
    assert not r['ordinary'] and r['reason']=='abrupt_turn'
    r,*_=review(source_step=4)
    assert not r['ordinary'] and r['reason']=='direction_unavailable'


def test_expired_prediction_prefix_is_replaced_by_actual_control_history():
    actions=[action()]*4+[action(90.)]*4
    history={2:action(90.),3:action(90.)}
    r,event,values,events=review(actions,source_step=4,history=history)
    assert r['ordinary'] and r['local_angle_degrees']==0.
    proof=dict(source_step=4,result=dict(events=events))
    controls={k:dict(action=v) for k,v in history.items()}
    assert r==ordinary_turn_from_record(event,dict(observation_step=0,actions=values),proof,controls,{},[])


def runtime():
    from perceptbudget.closed_loop import DeadlineRuntime
    return DeadlineRuntime(dict(H=50,m3_critical_forecast='remaining_v1',m3_phase_event_source='remaining_v2',
        m3_phase_refresh=True,m3_phase_coverage_memory='all_active_v1',
        m3_event_identity='turn_direction_v1',m3_ordinary_turn_policy='context_v1',m3_max_refresh_interval=30,
        m3_scheduler_mode='deadline_high',m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None),m3_events={},
        m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05))


def test_ordinary_motion_targets30_without_a_hard20_step_minimum():
    r=runtime();actions=[action(a) for a in (0,15,30,45,60,75)]+[action(90.)]*44
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
        dict(commit_step=0,commit_monotonic=0.))
    q={i:(a,0,0,i) for i,a in enumerate(actions)}
    d=r.decide(now=0.,step=0,queue=q,rows=[],inflight=False,rtc_delay_steps=5)
    assert d['critical_step'] is None and not d['start_now']
    assert d['first_refresh_cap']['request_step']==30
    assert d['ordinary_turn_suppressed'][0]['target_step']==4
    assert r.decide(now=1.5,step=30,queue=q,rows=[],inflight=False,rtc_delay_steps=5)['start_now']
    r=runtime();actions=[list(a) for a in actions]
    for a in actions[10:]:a[6]=1.
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
        dict(commit_step=0,commit_monotonic=0.))
    q={i:(a,0,0,i) for i,a in enumerate(actions)}
    d=r.decide(now=0.,step=0,queue=q,rows=[],inflight=False,rtc_delay_steps=5)
    assert d['critical_step']==10 and d['candidate_event_kinds']==['gripper_close'] and d['start_now']


def test_context_filter_does_not_erase_a_new_direction_restored_by_identity():
    r=runtime();response=dict(start_step=10,stop_step=50,source_prediction_id=0,
        target_step=11,kinds=['turn'],event_source='remaining_v2')
    r.committed(dict(prediction_id=1,observation_step=0,observation_monotonic=0.,actions=[action()]*50,
        m3_timing=dict(reason='action_event',deadline=1.,critical_step=11,candidate_event_kinds=['turn'],
            refresh_phase_response=response)),dict(commit_step=10,commit_monotonic=.1))
    actions=[action()]*6+[action(a) for a in (0,15,30,45,60,75)]+[action(90.)]*38
    r.committed(dict(prediction_id=2,observation_step=10,observation_monotonic=.2,actions=actions),
        dict(commit_step=10,commit_monotonic=.2))
    q={10+i:(a,2,10,i) for i,a in enumerate(actions)}
    rows=[dict(step=i,action=action(),action_start_monotonic=(i-8)*.05) for i in (8,9)]
    d=r.decide(now=1.,step=10,queue=q,rows=rows,inflight=False,rtc_delay_steps=5)
    assert d['critical_step']==20 and not d['ordinary_turn_suppressed']
    assert d['ordinary_turn_review'][0]['ordinary']
    assert not d['refresh_phase_identity_review'][0]['same_event']
