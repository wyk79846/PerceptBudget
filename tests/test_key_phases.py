from perceptbudget.key_phases import intervals,absolute_intervals,coverage


def chunk(motion,grip_change=None):
    return [[float(v),0.,0.,0.,0.,0.,-1. if grip_change is None or i<grip_change else 1.] for i,v in enumerate(motion)]


def test_phase_uses_motion_shoulders_and_requires_anchor():
    motion=[1.,1.,.8,.5,.2,.2,.2,.5,.8,1.,1.]
    assert intervals(chunk(motion))==[]
    p=intervals(chunk(motion,4))
    assert len(p)==1 and p[0]['start_offset']==0 and p[0]['stop_offset']==10
    assert p[0]['anchors']==[dict(kind='gripper_close',offset=4)]
    assert p[0]['end_observed']


def test_no_recovery_protects_only_remaining_chunk():
    p=intervals(chunk([1.,1.,.7,.3,.1,.1,.1],3))
    assert p and p[0]['stop_offset']==7 and not p[0]['end_observed']


def test_execution_window_crossing_phase_blocks_even_when_start_outside():
    phases=absolute_intervals(chunk([1.,1.,.8,.5,.2,.2,.2,.5,.8,1.,1.],4),100)
    c=coverage(phases,step=80,now=0.,latest_start=.5,period=.05,horizon=50,prediction_id=3)
    assert c['start_step']==90 and not c['allowed']
    assert coverage(phases,step=110,now=0.,latest_start=0.,period=.05,horizon=50,prediction_id=3)['allowed']


def test_small_dips_and_constant_closed_grip_are_not_phases():
    assert intervals(chunk([1.,1.,.9,.8,.9,1.,1.],3))==[]
    assert intervals(chunk([.01]*10,4))==[]


def test_gate_changes_only_fast_admission_not_full_deadline():
    from test_m3_service_choice import fixture
    from perceptbudget.m3_service_choice import choose_service
    from scripts.audit_rtc_system import audit_service_formula
    costs,d=fixture('gpu_emc')
    kwargs=dict(observation_step=10,previous_mode='Full',cost_model=costs,
                quantiles={'Full':.55,'Fast':.35},latest_start_cap=1.95)
    old=choose_service(d,**kwargs)
    d=dict(d,fast_phase_guard=dict(prediction_id=2,horizon=50,
        intervals=[dict(start_step=43,stop_step=48)]))
    new=choose_service(d,**kwargs)
    assert old['mode']=='Fast' and new['mode']=='Full'
    assert new['candidates'][0]==old['candidates'][0]
    assert new['candidates'][1]['latest_start']==old['candidates'][1]['latest_start']
    assert new['candidates'][1]['key_phase_gate']['start_step']==41
    applied=dict(d,service_seconds=new['selected']['service_seconds'],latest_start=new['selected']['latest_start'])
    assert audit_service_formula(new,applied,costs,'Full')==[]


def test_replacement_chunk_does_not_inherit_old_phase():
    from perceptbudget.closed_loop import DeadlineRuntime
    from perceptbudget.action_events import EventConfig
    r=object.__new__(DeadlineRuntime)
    r.chunks={};r.key_phase_guard=True;r.fast_selection=False;r.dvac_enabled=False
    r.action_events_enabled=False;r.merge_gap=None;r.open_probe=None;r.event_coverage_mode='event_only'
    r.events_config=EventConfig()
    r.committed(dict(prediction_id=0,observation_step=100,observation_monotonic=1.,
        actions=chunk([1.,1.,.8,.5,.2,.2,.2,.5,.8,1.,1.],4)),dict(commit_step=103,commit_monotonic=1.2))
    assert r.chunks[0]['key_phases'][0]['start_step']==100
    r.committed(dict(prediction_id=1,observation_step=110,observation_monotonic=2.,
        actions=chunk([1.]*11)),dict(commit_step=113,commit_monotonic=2.2))
    assert r.chunks[1]['key_phases']==[]


def test_pending_fast_is_cancelled_if_updated_phase_gate_blocks():
    from types import SimpleNamespace
    from perceptbudget.closed_loop import DeadlineRuntime
    r=object.__new__(DeadlineRuntime)
    r.key_phase_guard=True;r.recovering_plan={'m3_fast_formula':{'mode':'Fast'}}
    r.clocks=SimpleNamespace(profile='baseline',pending=None,poll=lambda:None)
    decision=dict(m3_fast_formula=dict(candidates=[{},dict(key_phase_gate=dict(allowed=False))]))
    result=r.control_clocks(decision,inflight=False)
    assert r.recovering_plan is None and result['start_now'] is False


def test_runtime_decision_binds_current_phases_and_selects_full():
    from perceptbudget.closed_loop import DeadlineRuntime
    from test_m3_service_choice import fixture
    costs,_=fixture()
    for mode in ('Full','Fast'):
        for cell in costs['service'][mode].values():cell['samples_seconds']=[.55 if mode=='Full' else .3]
    cfg=dict(m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,maximum_action_offset=50,
        maximum_observation_age_seconds=None),m3_events={},m3_service_samples={h:[.55] for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05,H=50,m3_scheduler_mode='deadline_high',
        m3_fast_enabled=True,m3_fast_advance=True,m3_fast_key_phases=True,m3_fast_cost_model=costs)
    runtime=DeadlineRuntime(cfg)
    phases=[dict(start_step=35,stop_step=45)]
    runtime.chunks={0:dict(origin=0,observation=0.,dvac_step=None,key_phases=phases)}
    queue={i:([0.]*7,0,0,i) for i in range(5,50)}
    plan=runtime.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,rtc_delay_steps=8)
    assert plan['fast_phase_guard']['prediction_id']==0
    assert plan['fast_phase_guard']['intervals']==phases
    assert plan['m3_fast_formula']['mode']=='Full'
    assert plan['m3_fast_formula']['candidates'][1]['key_phase_gate']['allowed'] is False
