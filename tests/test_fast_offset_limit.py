import pytest
from perceptbudget.closed_loop import DeadlineRuntime


def runtime():
    costs=dict(waits={'baseline':[5.]*4,'gpu_emc':[2.]*4},service={
        mode:{h:dict(energy_j=e,mean_seconds=t,samples_seconds=[t],from_profile='baseline' if h=='hot' else 'gpu_emc') for h in ('hot','short','long')}
        for mode,e,t in [('Full',10.,.5),('Fast',5.,.3)]})
    cfg=dict(m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025,maximum_action_offset=50,maximum_observation_age_seconds=None),
        m3_events={},m3_service_samples={h:[.5] for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05,H=50,
        m3_scheduler_mode='deadline_high',m3_fast_enabled=True,m3_fast_cost_model=costs,m3_fast_max_offset=30)
    r=DeadlineRuntime(cfg);r.fast_last_mode='Fast'
    r.chunks={2:dict(origin=64,observation=0.,dvac_step=None,context_mode='Fast')}
    return r


def test_limit_counts_inference_steps_and_full_starts_before_boundary():
    r=runtime();queue={i:([0.]*7,2,0,i-64) for i in range(72,114)}
    plan=r.decide(now=3.6,step=72,queue=queue,rows=[],inflight=False,rtc_delay_steps=11)
    assert plan['admitted_remaining_actions']==22  # 30 minus8 consumed before commit
    assert plan['m3_fast_formula']['mode']=='Full'
    assert plan['deadline']==pytest.approx(3.6+22*.05)
    assert plan['latest_start']==pytest.approx(plan['deadline']-.5-.025)
    assert r.admitted(queue[93],93,4.)
    assert not r.admitted(queue[94],94,4.)


def test_full_keeps_fifty_and_overdue_fast_is_not_extended():
    r=runtime();item=([0.]*7,2,0,30)
    assert not r.admitted(item,94,4.)
    r.chunks[2]['context_mode']='Full'
    assert r.admitted(item,94,4.)
    assert not r.admitted(item,114,4.)
