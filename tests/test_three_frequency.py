import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.three_frequency import select_full,ThreeFullRuntime
from perceptbudget.device_clocks import runtime_trace


@pytest.mark.parametrize('point,expected',[(30.2,True),(30.3,True),(30.6,False),(30.,True)])
def test_floor_applies_to_start_point_not_duration(point,expected):
    from perceptbudget.three_frequency import start_point
    #74ms per control step:30.2 ->30 is14.8ms early,30.6 is44.4ms.
    row=start_point(30,10.,10.+(point-30)*.074,.074,.025)
    assert row['continuous_start_step']==pytest.approx(point)
    assert row['floor_allowed'] is expected
    assert row['advance_seconds']==pytest.approx((point-30)*.074)


@pytest.mark.parametrize('ready_at,expected_end',[(.005,.005),(.1,.025)])
def test_intra_tick_clock_wait_obeys25ms_without_a_whole_control_step(ready_at,expected_end):
    from perceptbudget.three_frequency import wait_for_clock
    timer=SimpleNamespace(now=0.)
    clocks=SimpleNamespace(pending='gpu_emc')
    def poll():
        if timer.now>=ready_at:clocks.pending=None
    clocks.poll=poll
    with patch('perceptbudget.three_frequency.time.monotonic',lambda:timer.now), \
         patch('perceptbudget.three_frequency.time.sleep',lambda dt:setattr(timer,'now',timer.now+dt)):
        result=wait_for_clock(clocks,.025)
    assert result['seconds']==pytest.approx(expected_end)
    assert result['acknowledged']==(ready_at<=.025)
    assert result['simulation_frozen'] and not result['queue_consumed']


def decision(end=12.):
    return dict(decision_monotonic=10.,deadline=end,margin_seconds=.025,
        estimated_control_period_seconds=.05,service_seconds=.55,latest_start=end-.575)


@pytest.mark.parametrize('profile',['baseline','gpu_emc','deep_idle'])
def test_medium_is_preferred_even_when_high_is_faster(profile):
    f=select_full(decision(),medium_seconds=.75,medium_cap=20.,current_profile=profile,
                  transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018))
    assert f['selected']['profile']=='gpu_emc'
    assert f['candidates'][1]['feasible']
    assert f['selection_order']==['gpu_emc','baseline']


def test_high_only_when_medium_cannot_meet_deadline():
    args=dict(medium_seconds=.75,medium_cap=20.,current_profile='deep_idle',
              transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018))
    f=select_full(decision(10.7),**args)
    assert not f['candidates'][0]['feasible'] and f['selected']['profile']=='baseline'
    assert f['reason']=='high_medium_too_late'
    assert select_full(decision(10.3),**args)['reason']=='high_emergency'


def test_medium_rtc_prefix_and_first_refresh_caps_are_respected():
    args=dict(medium_seconds=.75,medium_cap=9.,current_profile='deep_idle',
              transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018))
    assert select_full(decision(),**args)['selected']['profile']=='baseline'
    d=decision();d['first_refresh_cap']={'latest_start':9.}
    f=select_full(d,**dict(args,medium_cap=20.))
    assert not any(r['feasible'] for r in f['candidates'])


def test_first_refresh_reserves_ack_tick_even_when_starting_from_high():
    d=decision();d['first_refresh_cap']=dict(latest_start=10.5,recovery_ticks=0)
    f=select_full(d,medium_seconds=.75,medium_cap=20.,current_profile='baseline',
                  transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018))
    assert f['candidates'][0]['first_cap']==pytest.approx(10.45)
    assert f['candidates'][1]['first_cap']==pytest.approx(10.5)


def config():
    costs=dict(cells=[dict(profile=p,from_profile='gpu_emc',history=h,
        service_samples_seconds=[.5 if p=='baseline' else .7]*20)
        for p in ('baseline','gpu_emc') for h in ('hot','short','long')])
    return dict(H=50,m3_low_full_costs=costs,m3_scheduler_mode='deadline_dvfs',
        m3_three_frequency_transitions=dict(medium_seconds=.0032,high_seconds=.019,high_to_medium_seconds=.018),
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,maximum_action_offset=50,
            maximum_observation_age_seconds=None,minimum_low_idle_seconds=.2),
        m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)


def test_no_inference_in_deep_idle_and_no_switch_during_compute():
    r=ThreeFullRuntime(config())
    changes=[]
    r.clocks=SimpleNamespace(profile='deep_idle',pending=None,poll=lambda:None,
                            switch=changes.append,last_low=9.,last_change=None)
    with pytest.raises(RuntimeError,match='Cannot infer'):r.requested({'observation_step':30})
    d=dict(start_now=True,compute_profile='gpu_emc')
    assert not r.control_clocks(d,inflight=False)['start_now']
    assert changes==['gpu_emc'] and r.recovery_target=='gpu_emc'
    assert not r.control_clocks(d,inflight=False)['start_now']
    r.clocks.profile='gpu_emc'
    assert r.control_clocks(d,inflight=False)['start_now']
    r.active_profile='gpu_emc'
    assert r.control_clocks(d,inflight=True)==d
    r.clocks.profile='deep_idle'
    with pytest.raises(RuntimeError):r.control_clocks(d,inflight=True)


def test_wait_can_enter_low_but_short_gap_does_not_switch():
    r=ThreeFullRuntime(config());changes=[]
    r.clocks=SimpleNamespace(profile='gpu_emc',pending=None,poll=lambda:None,switch=changes.append)
    d=dict(start_now=False,latest_start=10.1,decision_monotonic=10.)
    r.control_clocks(d,inflight=False);assert changes==[]
    d['latest_start']=11.
    r.control_clocks(d,inflight=False);assert changes==['deep_idle']


@pytest.mark.parametrize('acknowledged',[True,False])
def test_startup_wait_can_submit_in_same_tick_or_fall_back_without_changing_budget(acknowledged):
    r=ThreeFullRuntime(dict(config(),m3_start_floor_ms=25));changes=[]
    r.clocks=SimpleNamespace(profile='deep_idle',pending=None,poll=lambda:None,switch=changes.append)
    d=dict(start_now=True,compute_profile='gpu_emc',service_seconds=.8,margin_seconds=.025,
           latest_start=10.0148,decision_monotonic=10.,
           start_point_rounding=dict(floor_allowed=True))
    def waiting(clock,limit):
        assert limit==.025
        if acknowledged:clock.profile='gpu_emc'
        return dict(acknowledged=acknowledged)
    with patch('perceptbudget.three_frequency.wait_for_clock',side_effect=waiting) as wait:
        result=r.control_clocks(d,inflight=False)
        wait.assert_called_once()
    assert changes==['gpu_emc'] and result['start_now'] is acknowledged
    assert r.recovering_plan['service_seconds']==.8 and r.recovering_plan['latest_start']==10.0148
    assert r.recovering_plan['margin_seconds']==.025


def test_outside25ms_prepares_frequency_but_does_not_wait_or_submit_early():
    r=ThreeFullRuntime(dict(config(),m3_start_floor_ms=25));changes=[]
    r.clocks=SimpleNamespace(profile='deep_idle',pending=None,poll=lambda:None,switch=changes.append)
    d=dict(start_now=True,compute_profile='gpu_emc',start_point_rounding=dict(floor_allowed=False))
    with patch('perceptbudget.three_frequency.wait_for_clock') as wait:
        assert not r.control_clocks(d,inflight=False)['start_now']
        wait.assert_not_called()
    assert changes==['gpu_emc']


def test_three_state_runtime_protocol_and_acknowledgement():
    read,write=os.pipe();events=[]
    tokens={b'HIGH':'baseline',b'MID':'gpu_emc',b'IDLE':'deep_idle'}
    payloads=iter([b'MID\n',b'HIGH\n',b'STOP\n'])
    def emit(e):
        events.append(e)
        if e['status']=='APPLIED':os.write(write,next(payloads))
    try:
        os.write(write,b'IDLE\n')
        with patch('perceptbudget.device_clocks.observed_switch',return_value={}):
            runtime_trace(SimpleNamespace(actual=lambda:{}),emit,read,seconds=2,tokens=tokens)
    finally:os.close(write);os.close(read)
    assert [e['profile'] for e in events if e['status']=='APPLIED']==['deep_idle','gpu_emc','baseline']


def test_runtime_decision_does_not_modify_buffer_or_event_deadline():
    r=ThreeFullRuntime(config())
    r.clocks=SimpleNamespace(profile='deep_idle',pending=None,last_low=9.,poll=lambda:None)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=9.,actions=actions),
                dict(commit_step=0,commit_monotonic=9.5))
    q={i:(a,0,0,i) for i,a in enumerate(actions)}
    d=r.decide(now=10.,step=10,queue=q,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['compute_profile']=='gpu_emc'
    assert d['deadline']==12.  #40 remaining actions *50ms
    assert len(q)==50 and d['three_frequency_formula']['policy']=='medium_first_v1'


def test_resident_pair_keeps_original_policy_and_alternates_order():
    import copy
    from perceptbudget.device_quality import frequency_pair_views,configure_frequency_variant
    cfg=dict(m3_three_frequency=True,m3_low_full=True,H=50,N=5,state_enabled=True,
             m3_low_full_policy='shared_deadline_v4',rtc_beta=5)
    before=copy.deepcopy(cfg)
    assert frequency_pair_views(cfg,0)==('full_two','full_three')
    assert frequency_pair_views(cfg,1)==('full_three','full_two')
    for view,enabled in [('full_two',False),('full_three',True)]:
        child=copy.deepcopy(cfg)
        configure_frequency_variant(child,view)
        assert child['m3_three_frequency'] is enabled and child['m3_low_full']
        for key in ('H','N','state_enabled','m3_low_full_policy','rtc_beta'):
            assert child[key]==cfg[key]
    assert cfg==before
    assert frequency_pair_views({},0)==('full_high','full_low')


def test_original_high_low_pair_configuration_does_not_enable_three_states():
    from perceptbudget.device_quality import configure_frequency_variant
    for view,low in [('full_high',False),('full_low',True)]:
        cfg=dict(m3_three_frequency=False)
        configure_frequency_variant(cfg,view)
        assert cfg['m3_low_full'] is low and not cfg['m3_three_frequency']
    with pytest.raises(ValueError):configure_frequency_variant({},'full_three')


def test_three_frequency_pair_plan_is_six_episodes_one_model_without_launch(capsys):
    import json
    from scripts.run_m4 import main
    args=['probe','--acceptance-group','m3','--low-full','--three-frequency',
          '--full-frequency-pair','--task-indices','0','1','2','--critical-wait',
          '--critical-wait-phases','--first-refresh-step','30','--phase-refresh','--dry-run']
    with patch('sys.argv',args),patch('scripts.run_m4.subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['tasks']==[['libero_spatial',i] for i in (0,3,6)]
    assert plan['episodes']==6 and plan['model_loads']==1
    assert plan['three_frequency'] and plan['full_frequency_pair']
    assert plan['Fast']=='disabled' and plan['common_low_tail_seconds']==3.
    launch.assert_not_called()


def test_start_point_pair_changes_only_dispatch_option(capsys):
    import json,copy
    from scripts.run_m4 import main
    from perceptbudget.device_quality import frequency_pair_views,configure_frequency_variant
    cfg=dict(m3_three_frequency=True,m3_low_full=True,m3_start_floor_ms=25,H=50,N=5)
    assert frequency_pair_views(cfg,0)==('full_three_strict','full_three_start25')
    for label,limit in [('full_three_strict',0),('full_three_start25',25)]:
        c=copy.deepcopy(cfg);configure_frequency_variant(c,label)
        assert c['m3_three_frequency'] and c['m3_start_floor_ms']==limit
        assert c['H']==50 and c['N']==5
    args=['probe','--acceptance-group','m3','--low-full','--three-frequency',
          '--full-frequency-pair','--start-floor-ms','25','--task-indices','0',
          '--critical-wait','--critical-wait-phases','--first-refresh-step','30','--phase-refresh','--dry-run']
    with patch('sys.argv',args),patch('scripts.run_m4.subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes']==2 and plan['model_loads']==1 and plan['start_floor_ms']==25
    assert plan['eta_ms']==25
    launch.assert_not_called()
