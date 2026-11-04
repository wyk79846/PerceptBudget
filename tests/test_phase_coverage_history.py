"""A later event type must not erase an independently completed response."""
import pytest

from perceptbudget.closed_loop import DeadlineRuntime
from scripts.audit_rtc_system import phase_history_at_decision


def runtime(memory='all_active_v1',turn_stop=130):
    cfg=dict(H=50,m3_critical_forecast='remaining_v1',m3_phase_event_source='remaining_v2',
        m3_phase_refresh=True,m3_phase_coverage_memory=memory,m3_scheduler_mode='deadline_high',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None),
        m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    r=DeadlineRuntime(cfg)
    for pid,origin,at,kind,start,stop,target in (
            (1,80,90,'turn',100,turn_stop,101),
            (2,90,100,'slowdown',102,125,104)):
        response=dict(start_step=start,stop_step=stop,source_prediction_id=pid-1,
            target_step=target,kinds=[kind],event_source='remaining_v2')
        r.committed(dict(prediction_id=pid,observation_step=origin,observation_monotonic=0.,
            actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50,
            m3_timing=dict(reason='action_event',deadline=1.,critical_step=target,
                candidate_event_kinds=[kind],refresh_phase_response=response)),
            dict(commit_step=at,commit_monotonic=pid*.1))
    actions=[[1.,0.,0.,0.,0.,0.,-1.] if i<10 else
             [0.,1.,0.,0.,0.,0.,-1. if i<15 else 1.] for i in range(50)]
    r.committed(dict(prediction_id=3,observation_step=100,observation_monotonic=.3,actions=actions),
        dict(commit_step=100,commit_monotonic=.3))
    queue={100+i:(a,3,100,i) for i,a in enumerate(actions)}
    rows=[dict(step=i,action=[1.,0.,0.,0.,0.,0.,-1.],action_start_monotonic=(i-98)*.05)
        for i in (98,99)]
    return r,queue,rows


def decide(r,queue,rows,step=100):
    return r.decide(now=1.,step=step,queue=queue,rows=rows,inflight=False,rtc_delay_steps=5)


def test_turn_slowdown_turn_retains_first_response_but_new_gripper_still_triggers():
    r,q,rows=runtime()
    d=decide(r,q,rows)
    assert [p['kinds'] for p in d['refresh_phase_history']]==[['turn'],['slowdown']]
    assert [(e['kind'],e['target_step']) for e in d['refresh_phase_suppressed']]==[('turn',110)]
    assert d['critical_step']==115 and d['candidate_event_kinds']==['gripper_close']


def test_opt_out_preserves_last_response_only_behavior():
    r,q,rows=runtime(None);d=decide(r,q,rows)
    assert 'refresh_phase_history' not in d
    assert d['critical_step']==110 and d['candidate_event_kinds']==['turn']


def test_another_kind_cannot_extend_the_original_turn_interval():
    r,q,rows=runtime(turn_stop=108);d=decide(r,q,rows)
    assert d['critical_step']==110 and d['candidate_event_kinds']==['turn']
    assert not d['refresh_phase_suppressed']
    assert d['refresh_phase_history'][0]['stop_step']==108
    later=decide(r,q,rows,step=108)
    assert all('turn' not in p['kinds'] for p in later['refresh_phase_history'])


def test_history_expires_at_original_end_even_without_another_response():
    r,q,rows=runtime();decide(r,q,rows)
    assert decide(r,q,rows,step=130)['refresh_phase_history']==[]


def test_history_requires_same_kind_phase_semantics():
    cfg=dict(H=50,m3_phase_coverage_memory='all_active_v1',m3_scheduler_mode='deadline_high',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025),m3_events={},
        m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    with pytest.raises(ValueError,match='typed remaining_v2'):
        DeadlineRuntime(cfg)


def test_auditor_includes_only_responses_already_completed_at_the_decision():
    responses={1:dict(m3_timing=dict(refresh_phase_response=dict(start_step=100,stop_step=130,
        kinds=['turn'],event_source='remaining_v2'))),
        2:dict(m3_timing=dict(refresh_phase_response=dict(start_step=102,stop_step=125,
        kinds=['slowdown'],event_source='remaining_v2')))}
    commits=[dict(prediction_id=1,commit_step=100,commit_monotonic=1.),
        dict(prediction_id=2,commit_step=100,commit_monotonic=2.)]
    assert [p['completed_prediction_id'] for p in phase_history_at_decision(commits,responses,100,1.5)]==[1]
    assert len(phase_history_at_decision(commits,responses,100,2.))==2
    assert [p['completed_prediction_id'] for p in phase_history_at_decision(commits,responses,125,3.)]==[1]
    assert phase_history_at_decision(commits,responses,130,4.)==[]
