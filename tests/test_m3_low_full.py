from types import SimpleNamespace
import pytest

from perceptbudget.m3_low_full import choose, LowFullRuntime


def costs():
    return dict(cells=[dict(profile=p,from_profile='gpu_emc',history=h,
        mean_service_seconds=.5 if p=='baseline' else .7,
        mean_absolute_service_energy_j=10. if p=='baseline' else 9.,
        service_samples_seconds=[.7]*20,p95_service_seconds=.7,source='test')
        for p in ('baseline','gpu_emc') for h in ('hot','short','long')],
        waits={p:[5.]*8 for p in ('baseline','gpu_emc')})


def decision(deadline=12.):
    return dict(decision_monotonic=10.,deadline=deadline,margin_seconds=.025,
        service_seconds=.6,latest_start=deadline-.625)


def test_low_full_needs_time_and_quality_and_still_records_energy():
    args=dict(costs=costs(),low_budget=.75,quiet_age=.3,ordinary=True,rtc_low_cap=20.)
    f=choose(decision(),**args)
    assert f['selected']['profile']=='gpu_emc'
    assert f['selected']['energy_j']==pytest.approx(sum(f['selected'][k] for k in
        ('wait_before_energy_j','service_energy_j','tail_energy_j')))
    assert choose(decision(10.7),**args)['selected']['profile']=='baseline'
    args['ordinary']=False
    assert choose(decision(),**args)['selected']['profile']=='baseline'


def test_low_full_is_used_even_if_service_energy_is_higher():
    c=costs()
    for cell in c['cells']:
        if cell['profile']=='gpu_emc':cell['mean_absolute_service_energy_j']=30.
    f=choose(decision(),costs=c,low_budget=.75,quiet_age=2.,ordinary=True,rtc_low_cap=20.)
    assert f['selected']['profile']=='gpu_emc'
    assert f['energy_used_for_selection'] is False


def test_runtime_preserves_high_service_history_and_never_switches_mid_low_inference():
    cfg=dict(H=50,m3_low_full_costs=costs(),m3_scheduler_mode='deadline_dvfs',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None,margin_updates=False,
            minimum_low_idle_seconds=.2),m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    r=LowFullRuntime(cfg)
    r.clocks=SimpleNamespace(profile='gpu_emc',pending=None,last_low=9.,poll=lambda:None)
    saved={k:list(v) for k,v in r.times.samples.items()}
    request=dict(prediction_id=1,observation_step=30,observation_monotonic=10.,
        actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50,m4_compute_profile='gpu_emc',
        m3_timing=dict(reason='buffer',deadline=12.,service_seconds=.75,history='hot',
            low_full_formula=dict(selected=dict(history='long'))))
    r.committed(request,dict(commit_step=44,commit_monotonic=10.8))
    assert saved=={k:list(v) for k,v in r.times.samples.items()}
    assert list(r.low_histories['long'])[-1]==pytest.approx(.8)
    r.active_profile='gpu_emc'
    assert r.control_clocks({'start_now':False},inflight=True)=={'start_now':False}
    r.clocks.pending='baseline'
    with pytest.raises(RuntimeError,match='Frequency changed'):
        r.control_clocks({},inflight=True)


def test_runtime_can_submit_admitted_low_full_but_keeps_first_refresh_high():
    cfg=dict(H=50,m3_low_full_costs=costs(),m3_scheduler_mode='deadline_dvfs',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None,margin_updates=False,
            minimum_low_idle_seconds=.2),m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    r=LowFullRuntime(cfg)
    r.clocks=SimpleNamespace(profile='gpu_emc',pending=None,last_low=9.,last_change=None,poll=lambda:None)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    for pid,origin,step in [(0,0,10),(1,30,40)]:
        r.committed(dict(prediction_id=pid,observation_step=origin,observation_monotonic=9.,actions=actions),
            dict(commit_step=origin,commit_monotonic=9.))
        queue={origin+i:(a,pid,origin,i) for i,a in enumerate(actions)}
        d=r.decide(now=10.,step=step,queue=queue,rows=[],inflight=False,rtc_delay_steps=16)
        if pid==0:
            assert not r.pending_low
        else:
            assert r.pending_low and d['low_full_formula']['selected']['profile']=='gpu_emc'
            assert d['path']=='low_full_compute'
            request=dict(observation_step=step,observation_monotonic=10.)
            r.requested(request)
            assert request['m4_compute_profile']=='gpu_emc'
            assert 'service_start_monotonic' not in request['m3_timing']


def test_measurement_history_uses_expected_service_start_not_decision_age():
    f=choose(decision(),costs=costs(),low_budget=.75,quiet_age=.1,ordinary=True,rtc_low_cap=20.)
    assert all(r['history']=='long' for r in f['candidates'])


def test_high_rtc_bound_is_not_shortened_by_low_delay():
    cfg=dict(H=50,m3_low_full_initial_delay=16,m3_low_full_costs=costs(),
        m3_scheduler_mode='deadline_dvfs',
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
            maximum_action_offset=50,maximum_observation_age_seconds=None,margin_updates=False,
            minimum_low_idle_seconds=.2),m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05)
    r=LowFullRuntime(cfg)
    r.clocks=SimpleNamespace(profile='gpu_emc',pending=None,last_low=9.,last_change=None,poll=lambda:None)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    r.committed(dict(prediction_id=1,observation_step=30,observation_monotonic=9.,actions=actions),
        dict(commit_step=40,commit_monotonic=9.))
    queue={30+i:(a,1,30,i) for i,a in enumerate(actions)}
    d=r.decide(now=10.,step=45,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert d['rtc_delay_steps']==16 and d['rtc_latest_submit_step']==64
    high=d['low_full_formula']['candidates'][0]
    # HIGH reserves its own10 steps +2 recovery/margin ticks, not LOW's16.
    assert high['latest_start']==pytest.approx(10.+(70-45-2)*.05)
    # A key event forces HIGH, and keeps exactly the same HIGH prefix bound.
    r.events=[dict(kind='turn',target_step=75,prediction_id=1)]
    high_d=r.decide(now=10.,step=45,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert high_d['rtc_delay_steps']==10 and high_d['rtc_latest_submit_step']==70
