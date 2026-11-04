import math

import pytest

from perceptbudget.action_events import detect_events
from perceptbudget.critical_forecast import forecast


def action(speed=1.,angle=0.,grip=-1.):
    rad=math.radians(angle)
    return [speed*math.cos(rad),speed*math.sin(rad),0.,0.,0.,0.,grip]


def estimate(future,history=(),period=.05):
    return forecast(future,previous_actions=history,current_step=125,now=10.,control_period=period)


def test_first_unexecuted_action_switches_from_actual_last_control():
    r=estimate([action(grip=1.)]*4,[action()]*2)
    assert r['critical_step']==125 and r['t_critical']==10.
    assert r['event_kinds']==['gripper_close']


def test_consumed_switch_is_not_reintroduced_as_a_future_event():
    assert estimate([action(grip=1.)]*4,[action(),action(grip=1.)])['events']==[]


def test_gradual_slowdown_is_found_without_an_adjacent_step_jump():
    future=[action(v) for v in (1.,.9,.8,.7,.6,.5,.4,.3)]
    assert not any(e['kind']=='slowdown' for e in detect_events(future))
    r=estimate(future,[action()]*2)
    assert r['critical_step']==130 and r['event_kinds']==['slowdown']
    assert r['t_critical']==pytest.approx(10.25)


def test_gradual_turn_uses_accumulated_heading_and_measured_period():
    future=[action(angle=v) for v in (15.,30.,45.,60.,75.,90.)]
    assert not any(e['kind']=='turn' for e in detect_events(future))
    r=estimate(future,[action()]*2,period=.074)
    assert r['critical_step']==128 and r['event_kinds']==['turn']
    assert r['t_critical']==pytest.approx(10.+3*.074)


def test_earliest_cue_wins_while_other_future_cues_are_retained():
    future=[action(v,grip=-1. if i<2 else 1.) for i,v in enumerate((1.,.9,.8,.7,.6,.5,.4))]
    r=estimate(future,[action()]*2)
    assert r['critical_step']==127 and r['event_kinds']==['gripper_close']
    assert {e['kind'] for e in r['events']}=={'gripper_close','slowdown'}


def test_no_history_does_not_invent_an_initial_gripper_transition():
    assert estimate([action(grip=1.)]*3)['critical_step'] is None
    assert estimate([])['critical_step'] is None


def test_runtime_selects_remaining_extractor_and_does_not_slide_targets():
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_critical_forecast='remaining_v1',m3_scheduler_mode='deadline_high',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None),m3_events={},
        m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    r=DeadlineRuntime(cfg)
    # The new model's expired prefix already says closed; actual controls are
    # still open. Only the executed history can expose the handoff transition.
    actions=[action(grip=1.)]*50
    r.committed(dict(prediction_id=1,observation_step=10,observation_monotonic=0.,actions=actions),
        dict(commit_step=20,commit_monotonic=.5))
    q={10+i:(a,1,10,i) for i,a in enumerate(actions)}
    rows=[dict(step=i,action=action(),action_start_monotonic=(i-18)*.05) for i in (18,19)]
    d=r.decide(now=.1,step=20,queue=q,rows=rows,inflight=False,rtc_delay_steps=10)
    assert d['critical_step']==20 and d['critical_time_monotonic']==.1
    assert d['candidate_event_kinds']==['gripper_close']
    proof=d['critical_forecast']
    d=r.decide(now=.15,step=21,queue=q,rows=rows+[dict(step=20,action=action(grip=1.),action_start_monotonic=.1)],inflight=True,rtc_delay_steps=10)
    assert d['critical_forecast'] is proof
    assert d['critical_step'] is None  # consumed target is discarded, not slid


def test_new_runtime_can_run_without_the_fixed30_step_cap():
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_critical_forecast='remaining_v1',m3_scheduler_mode='deadline_high',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None),m3_events={},
        m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    r=DeadlineRuntime(cfg);actions=[action()]*50
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
        dict(commit_step=0,commit_monotonic=0.))
    q={i:(a,0,0,i) for i,a in enumerate(actions)}
    d=r.decide(now=1.5,step=30,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert not d['start_now'] and 'first_refresh_cap' not in d
    assert d['rtc_latest_submit_step']==40  # async RTC bound still applies


@pytest.mark.parametrize('period',[0.,-1.,math.nan,math.inf])
def test_invalid_period_is_rejected(period):
    with pytest.raises(ValueError):estimate([action()]*3,period=period)
