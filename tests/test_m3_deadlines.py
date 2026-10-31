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


def test_first_event_free_refresh_cap_only_applies_to_bootstrap_chunk():
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_first_refresh_step=30,m3_scheduler_mode='deadline_high',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None,margin_updates=False),
        m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(cfg)
    actions=[[1.,0.,0.,0.,0.,0.,-1.] for _ in range(50)]
    def commit_chunk(pid,origin):
        runtime.committed(dict(prediction_id=pid,observation_step=origin,
            observation_monotonic=10.,actions=actions),dict(commit_step=origin,commit_monotonic=10.))
        return {origin+i:(a,pid,origin,i) for i,a in enumerate(actions)}
    queue=commit_chunk(0,0)
    before=runtime.decide(now=10.,step=29,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert not before['start_now']
    at=runtime.decide(now=10.,step=30,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert at['start_now'] and at['start_constraint']=='first_refresh'
    queue=commit_chunk(1,30)
    later=runtime.decide(now=10.,step=31,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert 'first_refresh_cap' not in later and not later['start_now']


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
    assert runtime.event_coverage_mode=='event_only'
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


@pytest.mark.parametrize('reason',['denoising_variance','buffer','action_event'])
@pytest.mark.parametrize('coverage',['event_only','shared_refresh'])
@pytest.mark.parametrize('target,new_kind,commit_step,suppressed',[
    (129,'gripper_close',126,1),
    (136,'gripper_close',126,0),  # A materially moved transition is new evidence.
    (129,'gripper_open',126,0),   # Opposite commands cannot cover each other.
    (132,'gripper_close',131,0),  # Arrival after the old event does not cover it.
])
def test_fresh_prediction_covers_nearby_event_regardless_of_trigger(reason,coverage,target,new_kind,commit_step,suppressed):
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_scheduler_mode='deadline_high',m3_dvac_enabled=True,m3_event_coverage_mode=coverage,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(cfg)
    origin=117
    sign=1 if new_kind=='gripper_close' else -1
    actions=[[1.,0.,0.,0.,0.,0.,-sign if origin+i<target else sign] for i in range(50)]
    request=dict(prediction_id=1,observation_step=origin,observation_monotonic=10.,actions=actions,
        dvac=dict(prefix_actions=50),m3_timing=dict(reason=reason,deadline=10.9,critical_step=130,
            candidate_event_kinds=['gripper_close'],service_seconds=.3))
    # It misses the variance deadline in wall time, yet reaches the controller
    # before the action-event step. Those are distinct deadline checks.
    runtime.committed(request,dict(commit_step=commit_step,commit_monotonic=11.))
    queue={origin+i:(a,1,origin,i) for i,a in enumerate(actions)}
    plan=runtime.decide(now=11.,step=commit_step,queue=queue,rows=[],inflight=False)
    if coverage=='event_only':suppressed=0  # Frozen legacy rule rejects this late deadline.
    assert plan['suppressed_event_count']==suppressed
    assert plan['critical_step']==(None if suppressed else target)
    assert request['m3_timing']['arrival_lateness_seconds']==pytest.approx(.1)
    assert runtime.margin.seconds==.05


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


@pytest.mark.parametrize('critical,dvac,target,reason',[
    (16,12,16,'action_event'),(8,12,12,'denoising_variance'),
    (None,12,12,'denoising_variance'),(8,None,8,'action_event'),
    (None,None,50,'buffer'),(12,12,12,'action_event'),
])
def test_later_signal_selects_later_available_boundary(critical,dvac,target,reason):
    plan=timing_plan(now=10.,step=0,buffer_end_step=50,period=.05,service_seconds=.3,
        critical_step=critical,dvac_step=dvac,signal_policy='latest')
    assert plan.deadline==pytest.approx(10+target*.05) and plan.reason==reason
    assert plan.latest_start==pytest.approx(plan.deadline-.3-.05)


def test_later_signal_never_overrides_buffer_age_or_tie_behavior():
    args=dict(now=10.,step=0,period=.05,service_seconds=.3,critical_step=16,dvac_step=12,signal_policy='latest')
    plan=timing_plan(**args,buffer_end_step=10)
    assert plan.reason=='buffer' and plan.deadline==pytest.approx(10.5)
    assert timing_plan(**args,buffer_end_step=50,observation_deadline=10.2).reason=='observation_age'
    with pytest.raises(ValueError):timing_plan(**dict(args,signal_policy='unknown'),buffer_end_step=50)


def test_later_signal_retains_discrete_rtc_submission_guard():
    from types import SimpleNamespace
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_scheduler_mode='deadline_dvfs',m3_signal_policy='latest',m3_dvac_enabled=True,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.553]*20 for h in ('hot','short','long')},
        m3_recovery_samples=[.634],rtc_initial_control_period_seconds=.064)
    rt=DeadlineRuntime(cfg);rt.clocks=SimpleNamespace(profile='gpu_emc',pending=None)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    rt.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions,
        dvac={'prefix_actions':50}),dict(commit_step=0,commit_monotonic=.5))
    rt.events=[dict(kind='gripper_close',target_step=49)]
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    plan=rt.decide(now=38*.064,step=38,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert plan['signal_policy']=='latest' and plan['start_now']
    assert plan['start_constraint']=='rtc_prefix' and plan['rtc_latest_submit_step']==40
    assert plan['rtc_submit_guard_steps']==2


def test_nearby_merge_uses_only_current_chunk_and_preserves_variance(monkeypatch):
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_scheduler_mode='deadline_high',m3_dvac_enabled=True,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05)
    monkeypatch.setattr('perceptbudget.closed_loop.detect_events',lambda *a,**kw: [
        dict(kind='slowdown',offset=10),dict(kind='gripper_close',offset=18)])
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    req=dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions,dvac={'prefix_actions':15})
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    for merge,target,reason in ((None,10,'action_event'),(8,18,'denoising_variance')):
        rt=DeadlineRuntime(dict(cfg,m3_merge_slowdown_steps=merge))
        rt.committed(req,dict(commit_step=0,commit_monotonic=.1))
        plan=rt.decide(now=0.,step=0,queue=queue,rows=[],inflight=False)
        assert plan['critical_step']==target and plan['reason']==reason
        assert plan['dvac_step']==15 and plan['signal_policy']=='earliest'
        if merge:assert plan['merged_slowdowns'][0]['replacement_offset']==18


def test_no_turn_candidate_preserves_other_events_and_variance(monkeypatch):
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_scheduler_mode='deadline_high',m3_dvac_enabled=True,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05)
    monkeypatch.setattr('perceptbudget.closed_loop.detect_events',lambda *a,**kw: [
        dict(kind='turn',offset=10),dict(kind='slowdown',offset=20),dict(kind='gripper_close',offset=30)])
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    req=dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions,dvac={'prefix_actions':25})
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    for ignore,expected in ((False,10),(True,20)):
        rt=DeadlineRuntime(dict(cfg,m3_ignore_turn=ignore))
        rt.committed(req,dict(commit_step=0,commit_monotonic=.1))
        plan=rt.decide(now=0.,step=0,queue=queue,rows=[],inflight=False)
        assert plan['critical_step']==expected and plan['dvac_step']==25
        assert plan['signal_policy']=='earliest'
        assert 'gripper_close' in [e['kind'] for e in rt.events]


def test_later_signal_remembers_passed_first_event_instead_of_chasing_next_event():
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_scheduler_mode='deadline_high',m3_signal_policy='latest',m3_dvac_enabled=True,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05)
    rt=DeadlineRuntime(cfg)
    actions=[[1.,0.,0.,0.,0.,0.,1. if 8<=i<45 else -1.] for i in range(50)]
    rt.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions,
        dvac={'prefix_actions':40}),dict(commit_step=0,commit_monotonic=.1))
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    first=rt.decide(now=0.,step=0,queue=queue,rows=[],inflight=False)
    passed=rt.decide(now=.45,step=9,queue=queue,rows=[],inflight=False)
    assert first['critical_step']==passed['critical_step']==8
    assert first['deadline']==passed['deadline']==2.
    assert passed['candidate_event_kinds']==['gripper_close']
    fresh=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    rt.committed(dict(prediction_id=1,observation_step=10,observation_monotonic=.5,actions=fresh,
        dvac={'prefix_actions':5}),dict(commit_step=20,commit_monotonic=1.))
    new_queue={10+i:(a,1,10,i) for i,a in enumerate(fresh)}
    plan=rt.decide(now=1.,step=20,queue=new_queue,rows=[],inflight=False)
    assert plan['critical_step'] is None and plan['start_now']


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


def test_h50_starts_before_end_and_earlier_limits_still_win():
    args=dict(now=10.,step=0,buffer_end_step=50,period=.05,service_seconds=.5,margin_seconds=.05)
    plan=timing_plan(**args)
    assert plan.reason=='buffer'
    assert plan.deadline==pytest.approx(12.5)
    assert plan.latest_start==pytest.approx(11.95)
    assert timing_plan(**args,critical_step=20).reason=='action_event'
    assert timing_plan(**args,dvac_step=15).reason=='denoising_variance'
    assert timing_plan(**args,observation_deadline=12.).reason=='observation_age'


@pytest.mark.parametrize('events,dvac,event_step,expected_reason,expected_target',[
    (True,False,16,'action_event',16),
    (False,True,16,'denoising_variance',12),
    (True,True,16,'denoising_variance',12),
    (True,True,8,'action_event',8),
])
def test_three_trigger_strategies_select_enabled_earliest_boundary(events,dvac,event_step,expected_reason,expected_target):
    from perceptbudget.closed_loop import DeadlineRuntime
    config=dict(m3_scheduler_mode='deadline_high',m3_dvac_enabled=dvac,m3_action_events_enabled=events,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=2.,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(config)
    actions=[[1.,0.,0.,0.,0.,0.,-1. if i<event_step else 1.] for i in range(50)]
    req=dict(prediction_id=0,observation_step=0,observation_monotonic=10.,actions=actions,
        dvac=dict(prefix_actions=12))
    runtime.committed(req,dict(commit_step=0,commit_monotonic=10.))
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    plan=runtime.decide(now=10.,step=0,queue=queue,rows=[],inflight=False)
    assert plan['reason']==expected_reason
    assert plan['deadline']==pytest.approx(10.+expected_target*.05)
    assert plan['action_events_enabled']==events and plan['dvac_enabled']==dvac
    assert plan['admitted_remaining_actions']==50


def test_measured_recovery_history_is_available_before_first_live_request():
    from perceptbudget.closed_loop import DeadlineRuntime
    config=dict(m3_scheduler_mode='deadline_dvfs',m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05),
        m3_events={},m3_service_samples={h:[.553]*20 for h in ('hot','short','long')},
        m3_recovery_samples=[.634,.581,.604],rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(config)
    assert runtime.times.estimate('hot')==.553
    assert runtime.times.estimate('recovery')==.634
    old=timing_plan(now=0.,step=0,buffer_end_step=50,period=.05,observation_deadline=2.,
        service_seconds=.553,margin_seconds=.05)
    calibrated=timing_plan(now=0.,step=0,buffer_end_step=50,period=.05,observation_deadline=2.,
        service_seconds=runtime.times.estimate('recovery'),margin_seconds=.05)
    assert old.latest_start-calibrated.latest_start==pytest.approx(.081)
    assert runtime.margin.seconds==.05


def test_no_age_cutoff_allows_full_horizon_but_retains_buffer_and_variance_deadlines():
    from perceptbudget.closed_loop import DeadlineRuntime
    config=dict(m3_scheduler_mode='deadline_high',m3_dvac_enabled=True,m3_action_events_enabled=False,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05)
    rt=DeadlineRuntime(config)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    req=dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions,dvac={'prefix_actions':50})
    rt.committed(req,dict(commit_step=0,commit_monotonic=0.5))
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    assert rt.admitted(queue[49],49,3.)
    assert not rt.admitted((actions[0],0,0,50),50,3.)
    plan=rt.decide(now=2.5,step=40,queue=queue,rows=[],inflight=False)
    assert plan['reason']=='buffer' and plan['start_now'] and plan['admitted_remaining_actions']==10
    rt.chunks[0]['dvac_step']=43
    assert rt.decide(now=2.5,step=40,queue=queue,rows=[],inflight=False)['reason']=='denoising_variance'


@pytest.mark.parametrize('recovering,step,expect_start',[(False,38,False),(False,39,True),(True,37,False),(True,38,True)])
def test_rtc_prefix_bound_starts_before_discrete_overlap_is_too_short(recovering,step,expect_start):
    from types import SimpleNamespace
    from perceptbudget.closed_loop import DeadlineRuntime
    config=dict(H=50,m3_scheduler_mode='deadline_dvfs',m3_dvac_enabled=True,m3_action_events_enabled=False,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.553]*20 for h in ('hot','short','long')},
        m3_recovery_samples=[.634],rtc_initial_control_period_seconds=.064)
    rt=DeadlineRuntime(config)
    if recovering:rt.clocks=SimpleNamespace(profile='gpu_emc',pending=None)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    rt.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
        dict(commit_step=0,commit_monotonic=.5))
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    plan=rt.decide(now=step*.064,step=step,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert plan['rtc_latest_submit_step']==40
    assert plan['rtc_submit_guard_steps']==(2 if recovering else 1)
    assert plan['start_now']==expect_start
    if expect_start:assert plan['start_constraint']=='rtc_prefix'
    # With the later-tick HIGH acknowledgement, the next submission still
    # has >=10 old target steps. d must stay10, never clipped to fit s41.
    if recovering and expect_start:
        assert 50-(step+1)>=plan['rtc_delay_steps']
