import pytest

from perceptbudget.action_events import EventConfig
from perceptbudget.event_identity import review_turn
from scripts.audit_rtc_system import turn_identity_from_record


def data(reference=(0.,0.,-1.),current=(0.,0.,1.)):
    phase=dict(start_step=20,stop_step=60,kinds=['turn'],target_step=25,
        completed_step=26,completed_prediction_id=1,event_source='remaining_v2')
    event=dict(kind='turn',prediction_id=2,target_step=40)
    chunks={1:dict(origin=10,phase_actions=[[*reference,0.,0.,0.,-1.]]*50),
        2:dict(origin=30,phase_actions=[[*current,0.,0.,0.,-1.]]*50)}
    return event,[phase],chunks


def test_downward_then_upward_is_a_new_turn_with_the_same_type_and_interval():
    args=data();result=review_turn(*args,EventConfig())
    assert not result['same_event'] and result['angle_degrees']==180.
    assert result['reference_action_step']==26  # Do not read expired response prefix.
    assert result['reason']=='new_turn_direction'


def test_same_direction_at_a_shifted_target_remains_covered():
    args=data(current=(.1,0.,-1.));result=review_turn(*args,EventConfig())
    assert result['same_event'] and result['angle_degrees']<6.
    assert args[1][0]['stop_step']==60


def test_newer_turn_supersedes_an_older_matching_direction():
    event,phases,chunks=data(reference=(0.,0.,1.))
    chunks[3]=dict(origin=20,phase_actions=[[0.,0.,-1.,0.,0.,0.,-1.]]*50)
    phases.append(dict(phases[0],completed_prediction_id=3))
    result=review_turn(event,phases,chunks,EventConfig())
    assert result['reference_prediction_id']==3 and not result['same_event']


@pytest.mark.parametrize('vector',[(0.,0.,0.),(.01,.01,.01)])
def test_unknown_direction_keeps_the_new_event(vector):
    result=review_turn(*data(reference=vector),EventConfig())
    assert not result['same_event'] and result['reason']=='direction_unavailable'


def test_auditor_reconstructs_from_raw_response_actions():
    event,phases,chunks=data()
    by_id={pid:dict(observation_step=c['origin'],actions=c['phase_actions']) for pid,c in chunks.items()}
    result=review_turn(event,phases,chunks,EventConfig())
    assert result==turn_identity_from_record(event,phases,by_id,{})
    by_id[1]['actions']=[[0.,0.,1.,0.,0.,0.,-1.]]*50
    assert result!=turn_identity_from_record(event,phases,by_id,{})


def test_runtime_retains_new_turn_instead_of_hiding_it_under_old_response():
    from tests.test_phase_coverage_history import runtime,decide
    r,q,rows=runtime();r.event_identity='turn_direction_v1'
    result=decide(r,q,rows)
    assert result['critical_step']==110
    assert not result['refresh_phase_suppressed']
    assert result['refresh_phase_identity_review'][0]['reason']=='new_turn_direction'
