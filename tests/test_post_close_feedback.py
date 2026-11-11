import pytest

from perceptbudget.action_events import EventConfig
from perceptbudget.post_close_feedback import confirmed_closure, request_cap


def controls(signs):
    return [dict(step=i,action=[.1,0.,0.,0.,0.,0.,s]) for i,s in enumerate(signs)]


def plan(wait=1.):
    return dict(decision_monotonic=3.,latest_start=3.+wait,margin_seconds=.025)


def arguments():
    return dict(rows=controls([-1.]*10+[1.]*2),step=12,observation_step=0,
        inflight=False,config=EventConfig(),full_seconds=.5)


def test_only_a_confirmed_executed_open_to_close_transition_is_used():
    cfg=EventConfig()
    assert confirmed_closure(controls([-1.]*10+[1.]),cfg) is None
    assert confirmed_closure(controls([1.]*12),cfg) is None
    assert confirmed_closure(controls([-1.]*10+[1.]*2),cfg)==dict(close_step=10,confirmed_step=12)
    assert confirmed_closure(controls([-1.]*10+[1.]*2+[-1.]*2),cfg) is None


def test_a_distant_refresh_gets_one_fixed_cap_after_closure():
    kw=arguments();cap=request_cap(plan(),**kw)
    assert cap['request_step']==13 and cap['observation_step']==0
    # Waiting for a frequency acknowledgement does not slide the request.
    newer=dict(kw,rows=controls([-1.]*10+[1.]*3),step=13,previous=cap)
    assert request_cap(plan(.01),**newer)==cap
    # One ordinary observation also fulfills this feedback obligation.
    assert request_cap(plan(),**dict(newer,observation_step=12)) is None


@pytest.mark.parametrize('wait',[0.,.2,.525])
def test_an_imminent_existing_refresh_is_not_duplicated(wait):
    assert request_cap(plan(wait),**arguments()) is None


def test_inflight_preclose_image_is_not_postclose_feedback_but_never_overlapped():
    kw=arguments()
    assert request_cap(plan(),**dict(kw,inflight=True)) is None
    assert request_cap(plan(),**kw) is not None
    assert request_cap(plan(),**dict(kw,observation_step=12)) is None


def test_new_closing_phase_can_get_new_feedback_after_an_open():
    kw=arguments();old=request_cap(plan(),**kw)
    rows=controls([-1.]*10+[1.]*2+[-1.]*2+[1.]*2)
    new=request_cap(plan(),**dict(kw,rows=rows,step=16,observation_step=12,previous=old))
    assert new['close_step']==14 and new['request_step']==17


def test_runtime_keeps_v2_default_and_adds_only_the_opt_in_request_cap():
    from tests.test_ordinary_turns import runtime
    r=runtime(None)
    actions=[[.1,0.,0.,0.,0.,0.,1.]]*50
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
                dict(commit_step=0,commit_monotonic=0.))
    rows=controls([-1.]*10+[1.]*2)
    for row in rows:row['action_start_monotonic']=row['step']*.05
    kw=dict(now=.6,step=12,queue={i:(a,0,0,i) for i,a in enumerate(actions)},
            rows=rows,inflight=False,rtc_delay_steps=10)
    before=r.decide(**kw)
    assert 'first_refresh_cap' not in before
    r.post_close_enabled=True
    after=r.decide(**kw)
    assert after['first_refresh_cap']['scope']=='post_close_feedback'
    assert after['first_refresh_cap']['request_step']==13
    assert after['latest_start']<before['latest_start']
    assert after['deadline']==before['deadline'] and after['critical_step']==before['critical_step']
    r.requested(dict(observation_step=13,observation_monotonic=.65))
    assert 'first_refresh_cap' not in r.decide(**dict(kw,step=13,now=.65))


def joint_controls(width=.002):
    rows=controls([-1.]*10+[1.]*5)
    for r in rows:
        w=.08 if r['step']<10 else width
        r['after']=dict(raw_state=[0.,0.,0.,0.,0.,0.,w/2,-w/2])
    return rows


def test_gripper_warning_requires_near_stop_and_planned_sustained_lift():
    from perceptbudget.post_close_feedback import saturated_lift_warning
    cfg=EventConfig();up=[[0.,0.,.3,0.,0.,0.,1.]]*2
    warning=saturated_lift_warning(joint_controls(),up,cfg)
    assert warning['open_width_m']==.08 and warning['close_step']==10
    assert saturated_lift_warning(joint_controls(.03),up,cfg) is None
    assert saturated_lift_warning(joint_controls(),[[0.]*7]*2,cfg) is None
    rows=joint_controls();rows[-2]['after']['raw_state'][6]=.02
    assert saturated_lift_warning(rows,up,cfg) is None
    assert saturated_lift_warning(controls([-1.]*10+[1.]*5),up,cfg) is None


def test_gated_warning_uses_two_ticks_for_an_already_imminent_refresh():
    kw=arguments()
    assert request_cap(plan(.2),**kw) is None  # Original blanket rule unchanged.
    cap=request_cap(plan(.2),**kw,existing_wait_seconds=.1)
    assert cap['request_step']==13 and cap['existing_wait_threshold_seconds']==.1
    assert request_cap(plan(.05),**kw,existing_wait_seconds=.1) is None


def test_gated_runtime_preserves_stable_grasps_and_requests_one_conflict_image():
    from tests.test_ordinary_turns import runtime
    r=runtime(None);r.post_close_enabled=True;r.gripper_lift_feedback=True
    actions=[[0.,0.,.3,0.,0.,0.,1.]]*50
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
                dict(commit_step=0,commit_monotonic=0.))
    kw=dict(now=.75,step=15,queue={i:(a,0,0,i) for i,a in enumerate(actions)},
        rows=joint_controls(.03),inflight=False,rtc_delay_steps=10)
    for row in kw['rows']:
        row['action_start_monotonic']=row['step']*.05
        row['action'][:3]=[0.,0.,.3]
    assert 'first_refresh_cap' not in r.decide(**kw)
    rows=joint_controls()
    for row in rows:
        row['action_start_monotonic']=row['step']*.05
        row['action'][:3]=[0.,0.,.3]
    d=r.decide(**dict(kw,rows=rows))
    assert d['first_refresh_cap']['feedback']['gripper_lift_warning']['policy']=='saturated_gripper_lift_v1'
    r.requested(dict(observation_step=16,observation_monotonic=.8))
    assert 'first_refresh_cap' not in r.decide(**dict(kw,rows=rows,step=16,now=.8))
