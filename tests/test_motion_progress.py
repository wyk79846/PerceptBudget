from perceptbudget.motion_progress import DownwardProgress


def row(step,z0,z1,command=-.5,pid=0):
    return dict(step=step,action=[0.,0.,command,0.,0.,0.,-1.],action_start_monotonic=10.+step*.05,
        before={'raw_state':[0.,0.,z0]},after={'raw_state':[0.,0.,z1]},
        source={'prediction_id':pid})


def warm(watch):
    rows=[row(i,1.-i*.005,1.-(i+1)*.005) for i in range(8)]
    watch.update(rows,protected=lambda r:False)
    return rows


def test_contact_onset_requests_once_and_fresh_snapshot_does_not_rearm_hold():
    watch=DownwardProgress();rows=warm(watch)
    for i in range(8,13):
        rows.append(row(i,.96,.96));watch.update(rows,protected=lambda r:False)
    assert watch.onsets==1 and watch.pending['observed_step']==11
    assert watch.committed(dict(prediction_id=1,observation_step=10),dict(commit_step=12)) is None
    assert watch.pending is not None
    proof=watch.committed(dict(prediction_id=2,observation_step=11),dict(commit_step=12))
    assert proof['observed_step']==11 and watch.pending is None
    for i in range(13,30):
        rows.append(row(i,.96,.96));watch.update(rows,protected=lambda r:False)
    assert watch.onsets==1 and watch.pending is None


def test_grasp_guard_latches_ignored_block_until_motion_resumes():
    watch=DownwardProgress();rows=warm(watch)
    for i in range(8,15):
        rows.append(row(i,.96,.96));watch.update(rows,protected=lambda r:r['step']<=10)
    assert watch.onsets==watch.protected_onsets==1 and watch.pending is None
    for i in range(15,20):
        z=.96-(i-15)*.005
        rows.append(row(i,z,z-.005));watch.update(rows,protected=lambda r:False)
    assert not watch.active
    for i in range(20,24):
        rows.append(row(i,.935,.935));watch.update(rows,protected=lambda r:False)
    assert watch.onsets==2 and watch.pending is not None
    assert not watch.pending['grasp_protected']


def test_no_downward_command_or_reference_does_not_infer_contact():
    watch=DownwardProgress()
    rows=[row(i,1.,1.,command=0.) for i in range(30)]
    watch.update(rows,protected=lambda r:False)
    assert watch.reference is None and watch.pending is None
    watch=DownwardProgress();rows=warm(watch)
    for i in range(8,20):rows.append(row(i,.96,.96,command=.5))
    watch.update(rows,protected=lambda r:False)
    assert watch.pending is None and watch.onsets==0


def test_repeated_decision_does_not_count_same_physical_step_twice():
    watch=DownwardProgress();rows=warm(watch)
    first=watch.status()
    for _ in range(5):watch.update(rows,protected=lambda r:False)
    assert watch.status()==first


def test_new_progress_waits_even_if_old_forecast_phase_already_waited():
    from perceptbudget.critical_wait import PhaseWait,rtc_prefix_plan
    from perceptbudget.action_events import EventConfig
    guard=PhaseWait();guard.complete(dict(phase=dict(start_step=90,stop_step=120),kinds=['gripper_open']))
    request=dict(m3_timing=dict(reason='action_event',critical_step=105,
        candidate_event_kinds=['motion_blocked'],progress_alert=dict(observed_step=101)))
    review=guard.review(request,[],95,EventConfig())
    assert review['wait'] and review['reason']=='new_execution_progress'
    assert rtc_prefix_plan(request,105,10,review)['selected_delay_steps']==0


def test_runtime_alert_and_record_audit_use_only_completed_state_rows():
    import copy
    from perceptbudget.closed_loop import DeadlineRuntime
    from scripts.audit_rtc_system import audit_motion_progress
    cfg=dict(H=50,m3_phase_refresh=True,m3_motion_progress='refresh',m3_scheduler_mode='deadline_high',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(cfg);actions=[[0.,0.,-.5,0.,0.,0.,-1.]]*50
    request=dict(prediction_id=0,observation_step=0,observation_monotonic=9.,actions=actions)
    commit=dict(commit_step=0,commit_monotonic=9.5)
    runtime.committed(request,commit)
    trace=[dict(event='prediction_committed',request=copy.deepcopy(request),**commit)]
    rows=[];queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    for step in range(12):
        d=runtime.decide(now=10.+step*.05,step=step,queue=queue,rows=rows,inflight=False,rtc_delay_steps=10)
        trace.append(dict(event='m3_decision',step=step,**copy.deepcopy(d)))
        if step<8:r=row(step,1.-step*.005,1.-(step+1)*.005)
        else:r=row(step,.96,.96)
        rows.append(r);trace.append(dict(event='control_step',**r))
    alert=next(e for e in trace if e['event']=='m3_decision' and e.get('progress_alert'))
    assert alert['step']==11 and alert['critical_step']==11 and alert['start_now']
    assert alert['motion_progress']['processed_controls']==11
    failures=[]
    audit_motion_progress(cfg,trace,lambda name,ok,where=None:failures.append(name) if not ok else None)
    assert not failures
    damaged=copy.deepcopy(trace)
    for e in damaged:
        if e['event']=='control_step' and e['step']>=8:
            z=.96-(e['step']-8)*.005;e['before']['raw_state'][2]=z;e['after']['raw_state'][2]=z-.005
    audit_motion_progress(cfg,damaged,lambda name,ok,where=None:failures.append(name) if not ok else None)
    assert 'progress_known_state_replay' in failures
