import json
import sys

import pytest

from perceptbudget.action_events import EventConfig
from perceptbudget.feedback_refresh import review_cues
from tests.test_ordinary_turns import runtime,action


def review(events,reference=None,actions=None):
    return review_cues(events,prediction_id=2,origin=10,source_step=20,period=.05,
        response_seconds=.575,margin_seconds=.025,minimum_steps=20,
        actions=actions or [action()]*50,reference=reference,config=EventConfig())


def test_nearby_ordinary_cues_execute_from_the_fresh_plan_and_far_cues_request_feedback():
    r=review([dict(kind=k,target_step=s) for k,s in
        [('gripper_close',25),('slowdown',27),('turn',31),('gripper_open',42)]])
    assert r['fresh_plan_end_step']==42 and r['earliest_ordinary_request_step']==30
    assert [e['suppress'] for e in r['events']]==[True,True,True,False]


def test_consistent_transition_is_not_automatically_another_feedback_deadline():
    r=review([dict(kind='gripper_close',target_step=44)],
        dict(prediction_id=1,origin=0,actions=[action()]*50,
             events=[dict(kind='gripper_close',target_step=43)]))
    assert r['events'][0]['reason']=='consistent_forecast' and r['events'][0]['suppress']


def test_genuinely_new_gripper_transition_can_interrupt_the_ordinary_window():
    r=review([dict(kind='gripper_close',target_step=25)],
        dict(prediction_id=1,origin=0,actions=[action()]*50,events=[]))
    assert r['events'][0]['reason']=='new_gripper_transition' and not r['events'][0]['suppress']


def test_same_turn_label_with_a_reversed_direction_is_not_consistent():
    r=review([dict(kind='turn',target_step=44)],
        dict(prediction_id=1,origin=0,actions=[action()]*50,
             events=[dict(kind='turn',target_step=44)]),actions=[action(180.)]*50)
    e=r['events'][0]
    assert e['direction_difference_degrees']==180. and not e['suppress']


def test_window_cannot_slide_forward_when_the_control_tick_advances():
    r=runtime(50);r.feedback_policy='overlap_pacing_v1';r.feedback_minimum=20
    actions=[action()]*8+[action(scale=.3,grip=1.)]*32+[action(grip=-1.)]*10
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
        dict(commit_step=0,commit_monotonic=0.))
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    first=r.decide(now=0.,step=0,queue=queue,rows=[],inflight=False,rtc_delay_steps=5)
    assert first['critical_step']==40
    assert {e['kind'] for e in first['feedback_suppressed']}=={'gripper_close','slowdown'}
    later=r.decide(now=.5,step=10,queue=queue,rows=[],inflight=False,rtc_delay_steps=5)
    assert later['feedback_review']==first['feedback_review']
    assert first['feedback_review']['fresh_plan_end_step']==31


def test_buffer_limit_still_starts_a_request_even_when_every_cue_is_suppressed():
    r=runtime(50);r.feedback_policy='overlap_pacing_v1';r.feedback_minimum=20
    actions=[action()]*50
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
        dict(commit_step=0,commit_monotonic=0.))
    d=r.decide(now=2.,step=40,queue={i:(a,0,0,i) for i,a in enumerate(actions)},
        rows=[],inflight=False,rtc_delay_steps=5)
    assert d['reason']=='buffer' and d['start_now']


def test_cli_combines_energy_policy_without_changing_the_model_or_adding_a30_cap(monkeypatch,capsys):
    from scripts import run_m4
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','m3','--task-indices','0',
        '--energy-refresh','--dry-run'])
    assert run_m4.main()==0
    p=json.loads(capsys.readouterr().out)
    assert p['H']==50 and p['N']==5 and p['max_refresh_interval']==50
    assert p['feedback_policy']=='overlap_pacing_v1' and p['feedback_minimum_steps']==20
    assert p['Fast']=='disabled' and p['simplified_refresh']
