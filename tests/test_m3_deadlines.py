import pytest

from perceptbudget.deadline_controller import FullServiceTimes, SafetyMargin, full_continue_plan, timing_plan


def test_key_event_deadline_and_restore_counted_once():
    p=timing_plan(now=10.,step=0,buffer_end_step=30,critical_step=16,period=.05,
                  service_seconds=.5,margin_seconds=.1)
    assert p.reason=='action_event'
    assert p.deadline==pytest.approx(10.8)
    assert p.wait_seconds==pytest.approx(.2)


def test_no_event_uses_admitted_buffer_or_observation_age():
    p=timing_plan(now=10.,step=10,buffer_end_step=30,period=.05,service_seconds=.5,
                  observation_deadline=10.7)
    assert p.reason=='observation_age' and p.wait_seconds==pytest.approx(.15)
    p=timing_plan(now=10.,step=10,buffer_end_step=20,period=.05,service_seconds=.5)
    assert p.reason=='buffer' and p.start_now


def test_recovery_path_must_have_its_own_idle_time():
    p=full_continue_plan(now=10.,step=0,buffer_end_step=16,period=.05,
                         hot_service=.55,recovery_service=.58,margin_seconds=.05)
    assert p['hot'].wait_seconds==pytest.approx(.2)
    assert p['recovery'].wait_seconds==pytest.approx(.17)
    assert p['path']=='hot_continue'


def test_unreachable_deadline_is_explicit_and_negative_wait_is_not_used():
    p=timing_plan(now=10.,step=5,buffer_end_step=7,period=.05,service_seconds=.5)
    assert p.deadline_already_unreachable and p.start_now and p.wait_seconds==0


def test_recovery_estimate_uses_larger_calibrated_history_and_updates():
    s=FullServiceTimes(dict(hot=[.48]*20,short=[.50]*20,long=[.51]*20))
    assert s.estimate('hot')==.48 and s.estimate('recovery')==.51
    for _ in range(3):s.observe('short',.6)
    assert s.estimate('recovery')==.6
    for _ in range(3):s.observe('hot',.7)
    assert s.estimate('recovery')==.7  # Live concurrent load also bounds recovery.


def test_margin_covers_start_lateness_and_changes_gradually():
    m=SafetyMargin()
    m.observe(predicted=.55,actual=.58,start_lateness=.01)
    assert m.seconds==.05
    m.observe(predicted=.55,actual=.62,start_lateness=.01)
    assert m.seconds==pytest.approx(.09)
    for _ in range(40):m.observe(predicted=.55,actual=.55)
    assert m.seconds==pytest.approx(.07)


def test_runtime_reads_future_hints_and_counts_only_admitted_actions():
    from perceptbudget.closed_loop import DeadlineRuntime
    config=dict(m3_scheduler_mode='deadline_high',m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,
        maximum_action_offset=20,maximum_observation_age_seconds=2.,margin_updates=False),
        m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(config)
    actions=[[1.,0.,0.,0.,0.,0.,-1. if i<16 else 1.] for i in range(50)]
    request=dict(prediction_id=0,observation_step=0,observation_monotonic=9.5,actions=actions)
    runtime.committed(request,dict(commit_step=0,commit_monotonic=10.))
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    decision=runtime.decide(now=10.,step=0,queue=queue,rows=[],inflight=False)
    assert decision['critical_step']==16 and decision['reason']=='action_event'
    assert decision['admitted_remaining_actions']==20
    assert decision['wait_seconds']==pytest.approx(.25)
    assert runtime.admitted(queue[19],19,10.)
    assert not runtime.admitted(queue[20],20,10.)
    assert not runtime.admitted(queue[0],0,11.6)
    # A new Full already arrived for this close event. Seeing the same hint
    # in the new chunk must not launch another Full immediately.
    runtime.mode='deadline_high'
    next_actions=[[1.,0.,0.,0.,0.,0.,-1. if i<6 else 1.] for i in range(50)]
    new_request=dict(prediction_id=1,observation_step=10,observation_monotonic=10.2,actions=next_actions,
        m3_timing=dict(reason='action_event',deadline=10.9,critical_step=16,
                      candidate_event_kinds=['gripper_close'],service_seconds=.5))
    runtime.committed(new_request,dict(commit_step=14,commit_monotonic=10.7))
    new_queue={10+i:(a,1,10,i) for i,a in enumerate(next_actions)}
    decision=runtime.decide(now=10.7,step=14,queue=new_queue,rows=[],inflight=False)
    assert decision['suppressed_event_count']==1
    assert decision['critical_step'] is None


@pytest.mark.parametrize('change',[dict(period=0),dict(service_seconds=float('nan')),dict(critical_step=1.5)])
def test_invalid_time_inputs_rejected(change):
    args=dict(now=10.,step=0,buffer_end_step=20,period=.05,service_seconds=.5)
    args.update(change)
    with pytest.raises(ValueError):timing_plan(**args)


def test_variance_only_tightens_existing_deadlines_and_overdue_boundary_starts_now():
    args=dict(now=10.,step=10,buffer_end_step=20,period=.05,service_seconds=.3,margin_seconds=.05)
    original=timing_plan(**args)
    assert timing_plan(**args,dvac_step=40).deadline==original.deadline
    early=timing_plan(**args,dvac_step=15)
    assert early.reason=='denoising_variance' and early.start_now
    overdue=timing_plan(**args,dvac_step=9)
    assert overdue.start_now and overdue.deadline_already_unreachable


def test_variance_refresh_does_not_reject_otherwise_admitted_actions():
    from perceptbudget.closed_loop import DeadlineRuntime
    config=dict(m3_scheduler_mode='deadline_high',m3_dvac_enabled=True,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=20,
            maximum_observation_age_seconds=2.,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(config)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    req=dict(prediction_id=1,observation_step=10,observation_monotonic=10.,actions=actions,
        dvac=dict(prefix_actions=5))
    runtime.committed(req,dict(commit_step=18,commit_monotonic=10.4))
    queue={10+i:(a,1,10,i) for i,a in enumerate(actions)}
    decision=runtime.decide(now=10.4,step=18,queue=queue,rows=[],inflight=False)
    assert decision['reason']=='denoising_variance' and decision['start_now']
    assert decision['dvac_step']==15
    assert runtime.admitted(queue[29],29,10.4)
    assert not runtime.admitted(queue[30],30,10.4)
    config['m3_dvac_enabled']=False
    baseline=DeadlineRuntime(config)
    baseline.committed(req,dict(commit_step=18,commit_monotonic=10.4))
    assert baseline.decide(now=10.4,step=18,queue=queue,rows=[],inflight=False)['dvac_step'] is None
