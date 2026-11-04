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


@pytest.mark.parametrize('period',[0.,-1.,math.nan,math.inf])
def test_invalid_period_is_rejected(period):
    with pytest.raises(ValueError):estimate([action()]*3,period=period)
