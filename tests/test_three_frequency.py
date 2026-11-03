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


def test_shorter_idle_threshold_only_changes_continue_entry():
    original=config()
    candidate=config()
    candidate['m3_timing']['minimum_low_idle_seconds']=.1
    for cfg,expected in ((original,[]),(candidate,['deep_idle'])):
        r=ThreeFullRuntime(cfg);changes=[]
        r.clocks=SimpleNamespace(profile='baseline',pending=None,poll=lambda:None,switch=changes.append)
        d=dict(start_now=False,latest_start=10.15,decision_monotonic=10.,
               service_seconds=.8,deadline=11.,margin_seconds=.025,rtc_delay_steps=14,
               compute_profile='gpu_emc')
        out=r.control_clocks(d,inflight=False)
        assert changes==expected
        assert (out['latest_start'],out['deadline'],out['service_seconds'],out['rtc_delay_steps'])==(10.15,11.,.8,14)
        assert out['continue_idle_gate']['downclock']==bool(expected)
        r.clocks.profile='gpu_emc';r.active_profile='gpu_emc';changes.clear()
        r.control_clocks(dict(d),inflight=True)
        assert changes==[]
        r.control_clocks(dict(d,start_now=True,compute_profile='gpu_emc'),inflight=False)
        assert changes==[]


def test_short_idle_does_not_add_a_new_ack_tick_to_an_unchanged_compute_profile():
    cfg=config();cfg['m3_timing']['minimum_low_idle_seconds']=.1
    r=ThreeFullRuntime(cfg);changes=[]
    r.clocks=SimpleNamespace(profile='gpu_emc',pending=None,poll=lambda:None,switch=changes.append)
    d=dict(start_now=False,latest_start=10.18,decision_monotonic=10.,compute_profile='gpu_emc')
    out=r.control_clocks(d,inflight=False)
    assert changes==[] and out['continue_idle_gate']['reason']=='extra_ack_tick_not_budgeted'
    assert out['latest_start']==10.18
    out=r.control_clocks(dict(d,latest_start=10.3),inflight=False)
    assert changes==['deep_idle'] and out['continue_idle_gate']['reason']=='original_idle_rule'


@pytest.mark.parametrize('reference,candidate',[(200,100),(100,50),(100,40)])
def test_idle_threshold_pair_keeps_clock_policy_and_history_bins(reference,candidate):
    import copy
    from perceptbudget.device_quality import frequency_pair_views,configure_frequency_variant,camera_view_for_condition
    cfg=dict(config(),m3_three_frequency=True,m3_idle_min_ms=candidate,
             m3_idle_reference_ms=reference,m3_start_floor_ms=25)
    base=f'full_idle{reference}'
    assert frequency_pair_views(cfg,0)==(base,'full_idle_candidate')
    assert frequency_pair_views(cfg,1)==('full_idle_candidate',base)
    for label,value in ((base,reference/1000),('full_idle_candidate',candidate/1000)):
        child=copy.deepcopy(cfg);configure_frequency_variant(child,label)
        assert child['m3_timing']['minimum_low_idle_seconds']==value
        assert child['m3_three_frequency'] and child['m3_start_floor_ms']==25
        assert camera_view_for_condition(label)=='full'
        runtime=ThreeFullRuntime(child)
        assert runtime.settings['minimum_low_idle_seconds']==value
        assert runtime.low_histories.keys()==ThreeFullRuntime(config()).low_histories.keys()
    assert cfg['m3_timing']['minimum_low_idle_seconds']==.2


def test_idle_threshold_is_forwarded_to_actual_model_worker(tmp_path):
    from scripts import run_m4
    cfg=run_m4.load_config(repo_root=run_m4.ROOT)
    from perceptbudget.three_frequency import transition_costs
    transitions=transition_costs(cfg['output_root'])
    cfg['output_root']=str(tmp_path)
    calls=[]
    def stop_before_model(command,*unused,**kwargs):
        calls.append(command)
        if '--worker' in command:raise RuntimeError('test stopped before model load')
        assert '--prepare-environments' in command
        return 0
    args=['probe','--acceptance-group','m3','--low-full','--three-frequency',
          '--full-frequency-pair','--idle-min-ms','50','--idle-reference-ms','100',
          '--task-indices','2','--start-floor-ms','25']
    with patch('sys.argv',args),patch.object(run_m4,'load_config',return_value=cfg), \
         patch('perceptbudget.three_frequency.transition_costs',return_value=transitions), \
         patch('perceptbudget.three_frequency.preflight'),patch('perceptbudget.clock_access.authorize'), \
         patch('scripts.profile_clock_screen.monitored_worker',side_effect=stop_before_model):
        with pytest.raises(RuntimeError,match='test stopped before model load'):run_m4.main()
    worker=next(c for c in calls if '--worker' in c)
    assert worker[worker.index('--idle-min-ms')+1]=='50'
    assert worker[worker.index('--idle-reference-ms')+1]=='100'
    assert '--full-frequency-pair' in worker and '--three-frequency' in worker


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


def ordinary_costs():
    return dict(cells={p:dict(hold_power_w=8.,modes={
        'Full':dict(mean_seconds=.5,mean_energy_j=11.),
        'Fast':dict(mean_seconds=.3,mean_energy_j=7.)}) for p in ('baseline','gpu_emc')})


def test_ordinary_fast_keeps_the_full_schedule_and_phase_rules():
    a=ThreeFullRuntime(config())
    b=ThreeFullRuntime(dict(config(),m3_ordinary_fast=True,ordinary_fast_costs=ordinary_costs(),
        m3_phase_refresh=True,m3_first_refresh_step=30))
    # Use the same event/first-refresh settings in the comparison.
    a.phase_refresh=True;a.first_refresh_step=30
    for runtime in (a,b):
        runtime.clocks=SimpleNamespace(profile='deep_idle',pending=None,last_low=9.,poll=lambda:None)
        runtime.fast_delays={'baseline':6,'gpu_emc':8}
        actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
        runtime.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=9.,actions=actions),
                          dict(commit_step=0,commit_monotonic=9.5))
    queue={i:(x,0,0,i) for i,x in enumerate(actions)}
    args=dict(now=10.,step=10,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    reference=a.decide(**args);candidate=b.decide(**args)
    assert candidate['m3_fast_formula']['mode']=='Fast'
    assert {k:v for k,v in candidate.items() if k!='m3_fast_formula'}==reference
    b.ordinary_previous='Fast'
    assert b.decide(**args)['m3_fast_formula']['mode']=='Full'
    b.ordinary_previous='Full'
    b.events=[dict(kind='gripper_close',target_step=30)]
    critical=b.decide(**args)
    assert critical['reason']=='action_event' and critical['m3_fast_formula']['mode']=='Full'


def test_fast_service_cannot_replace_full_duration_samples():
    import copy
    r=ThreeFullRuntime(dict(config(),m3_ordinary_fast=True,ordinary_fast_costs=ordinary_costs()))
    original=copy.deepcopy((r.times.samples,r.times.recovery_samples,r.low_histories))
    request=dict(prediction_id=1,observation_step=10,observation_monotonic=1.,
        actions=[[0.,0.,0.,0.,0.,0.,-1.]]*50,m3_fast_choice=dict(mode='Fast'),
        m4_compute_profile='gpu_emc',m3_timing=dict(deadline=3.,critical_step=None,reason='buffer',history='hot',
            low_full_formula=dict(selected=dict(history='short'))))
    r.committed(request,dict(commit_step=16,commit_monotonic=1.3))
    assert (r.times.samples,r.times.recovery_samples,r.low_histories)==original
    assert r.ordinary_previous=='Fast' and r.chunks[1]['context_mode']=='Fast'


def test_cheaper_fast_request_can_still_lose_on_the_common_window():
    from perceptbudget.ordinary_fast import choose
    d=dict(decision_monotonic=1.,deadline=1.525,margin_seconds=.025,latest_start=1.,
        compute_profile='baseline',service_seconds=.5,rtc_delay_steps=10,
        estimated_control_period_seconds=.05,reason='buffer',deadline_already_unreachable=False,
        start_constraint='service_deadline')
    costs=ordinary_costs();costs['cells']['baseline']['hold_power_w']=10.
    costs['cells']['baseline']['modes']['Fast'].update(mean_energy_j=9.5,mean_seconds=.1)
    row=choose(d,step=30,previous_mode='Full',phases=[],events=[],costs=costs,fast_delay=5)
    assert row['mode']=='Full' and row['estimated_saving_j']<0
    assert row['candidates'][1]['service_energy_j']<row['candidates'][0]['service_energy_j']


def test_active_key_phase_and_slower_fast_delay_keep_full():
    from perceptbudget.ordinary_fast import choose
    d=dict(decision_monotonic=1.,deadline=2.,margin_seconds=.025,latest_start=1.4,
        compute_profile='gpu_emc',service_seconds=.5,rtc_delay_steps=10,
        estimated_control_period_seconds=.05,reason='buffer',deadline_already_unreachable=False,
        start_constraint='service_deadline')
    args=dict(step=30,previous_mode='Full',events=[],costs=ordinary_costs(),fast_delay=5)
    assert choose(d,phases=[dict(start_step=28,stop_step=35)],**args)['mode']=='Full'
    assert choose(d,phases=[],**dict(args,fast_delay=11))['mode']=='Full'


def test_clock_preparation_rechecks_current_phase_without_moving_full_plan():
    r=ThreeFullRuntime(dict(config(),m3_ordinary_fast=True,ordinary_fast_costs=ordinary_costs()))
    d=dict(decision_monotonic=1.,deadline=2.,margin_seconds=.025,latest_start=1.4,
        compute_profile='gpu_emc',service_seconds=.5,rtc_delay_steps=10,
        estimated_control_period_seconds=.05,reason='buffer',deadline_already_unreachable=False,
        start_constraint='service_deadline',start_now=True)
    r.recovering_plan=dict(d)
    r.recovery_target='gpu_emc'
    r.clocks=SimpleNamespace(profile='gpu_emc',pending=None,poll=lambda:None)
    r.ordinary_step=31;r.ordinary_phases=[dict(start_step=31,stop_step=40)]
    r.fast_delays={'gpu_emc':5}
    result=r.control_clocks(dict(d),inflight=False)
    assert result['m3_fast_formula']['mode']=='Full'
    assert result['m3_fast_formula']['observation_step']==31
    assert all(result[k]==v for k,v in d.items())
