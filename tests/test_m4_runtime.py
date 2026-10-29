from types import SimpleNamespace
from unittest.mock import patch
import pytest
from perceptbudget.m4_runtime import M4Runtime
from test_m4_candidates import CFG,WAIT,services


def runtime(tmp_path):
    cfg=dict(H=50,m3_scheduler_mode='deadline_dvfs',m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,
        maximum_action_offset=50,maximum_observation_age_seconds=None,margin_updates=False),
        m3_events={},m3_service_samples={h:[.5]*20 for h in ('hot','short','long')},
        rtc_initial_control_period_seconds=.05,m4_settings=dict(CFG,wait_cost_source='unused',directional_cost_source='unused'))
    adapter=SimpleNamespace(select_next_mode=lambda mode:None)
    with patch('perceptbudget.m4_runtime.load_service_costs',return_value=services()), \
         patch('perceptbudget.m4_runtime.wait_model',return_value=WAIT), \
         patch('perceptbudget.m4_runtime.load_transitions',return_value={}), \
         patch('perceptbudget.m4_runtime.Path.read_text',return_value='{}'):
        rt=M4Runtime(cfg,adapter)
    rt.log=tmp_path/'plans.jsonl'
    rt.clocks=SimpleNamespace(profile='baseline',pending=None,last_change=None,poll=lambda:None)
    return rt


def test_low_frequency_inference_does_not_force_high_or_switch(tmp_path):
    rt=runtime(tmp_path);rt.clocks.profile='gpu_emc'
    rt.phase='inflight'
    assert rt.control_clocks({'start_now':False},inflight=True)['start_now'] is False
    rt.clocks.pending='baseline'
    with pytest.raises(RuntimeError,match='during active inference'):
        rt.control_clocks({},inflight=True)


def test_locked_plan_is_not_postponed_every_control_tick(tmp_path):
    rt=runtime(tmp_path)
    rt.chunks[0]=dict(origin=0,observation=0)
    queue={i:(None,0,i) for i in range(50)}
    rt.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,rtc_delay_steps=5)
    initial=rt.active['prepare_at']
    rt.decide(now=.3,step=6,queue=queue,rows=[],inflight=False,rtc_delay_steps=5)
    assert rt.active['prepare_at']<=initial
    assert rt.plan_counter==1


def test_inclusive_service_entry_is_not_delayed_twice(tmp_path):
    rt=runtime(tmp_path)
    rt.active=dict(initial='gpu_emc',prepare_at=0,selected=dict(service_mode='Full',compute_profile='baseline',
        service_start_seconds=1,prepare_start_seconds=1,segments=[]))
    rt.phase='waiting';rt.clocks.profile='gpu_emc'
    def switch(profile):rt.clocks.profile=profile
    rt.clocks.switch=switch
    with patch('perceptbudget.m4_runtime.time.monotonic',return_value=10):
        result=rt.control_clocks({},inflight=False)
    assert result['start_now'] and rt.service_begin==10 and rt.entry_state=='gpu_emc'


def test_external_preparation_is_before_rtc_observation_limit(tmp_path):
    rt=runtime(tmp_path);rt.chunks[0]=dict(origin=0,observation=0)
    queue={i:(None,0,i) for i in range(50)}
    selected=dict(label='test',service_mode='Full',compute_profile='gpu_emc',next_state='gpu_emc',
        service_start_seconds=2,prepare_start_seconds=1.75,segments=[],energy_j=12)
    with patch('perceptbudget.m4_runtime.plan',return_value={'selected':selected}):
        rt.decide(now=.25,step=5,queue=queue,rows=[],inflight=False,rtc_delay_steps=16)
    assert rt.active['prepare_at']+.25<=rt.base['latest_start']+1e-9


def test_minimum_request_interval_does_not_start_service_timer(tmp_path):
    rt=runtime(tmp_path);rt.chunks[0]=dict(origin=0,observation=0)
    q={i:(None,0,i) for i in range(50)}
    d=rt.decide(now=.05,step=1,queue=q,rows=[],inflight=False,rtc_delay_steps=5)
    assert d['m4_phase']=='minimum_request_interval' and rt.active is None and rt.service_begin is None
    assert rt.control_clocks(d,inflight=False)['start_now'] is False


def test_emergency_is_replanned_when_old_event_passes(tmp_path):
    rt=runtime(tmp_path);rt.base={'reason':'action_event','critical_step':93,'candidate_event_kinds':['slowdown']}
    rt.active=dict(selected={'label':'emergency-Full-high'},trigger=('action_event',83,('gripper_open',)))
    rt.phase='preparing'
    d=rt.control_clocks({},inflight=False)
    assert not d['start_now'] and rt.active is None and rt.phase is None


def test_bootstrap_does_not_poison_steady_p95_and_poll_is_budgeted(tmp_path):
    rt=runtime(tmp_path)
    request=dict(prediction_id=0,observation_step=0,observation_monotonic=10.,ready_monotonic=12.,
        actions=[[0.]*7 for _ in range(50)])
    rt.committed(request,dict(commit_step=0,commit_monotonic=12.))
    assert all(len(v)==1 for v in rt.ready_histories.values())
    assert not rt.poll_delays
    for c in rt.services:
        rt.ready_histories[rt.cost_key(c)].clear();rt.ready_histories[rt.cost_key(c)].extend([.3]*60)
    rt.poll_delays.extend([.055,.06])
    c=next(c for c in rt.services if c['mode']=='Fast' and c['profile']=='baseline')
    rt.calibration_ready_samples[rt.cost_key(c)]=[.3]*60
    rt.end_to_end_histories[rt.cost_key(c)].append(1.5)
    measured=next(c for c in rt.planning_services(.05) if c['mode']=='Fast' and c['profile']=='baseline')
    assert measured['p95_service_seconds']==pytest.approx(.35) # percentile of whole-service samples, not component quantile sum or running max
