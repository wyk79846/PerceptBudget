from types import SimpleNamespace

import pytest
import torch

from perceptbudget.safe_paper import make_model
from perceptbudget.safe_runtime import SafeFailureMonitor, cap_refresh


def monitor(tmp_path):
    model=make_model(4,'cpu')
    for p in model.parameters():p.data.zero_()
    path=tmp_path/'probe.pt'
    torch.save(dict(record=dict(width=4,horizon='first',denoise='last'),
        state_dict=model.state_dict(),normalization='none',alpha=.2,
        cp_band=torch.tensor([.5,.9])),path)
    layer=torch.nn.Linear(4,2).eval().requires_grad_(False)
    expert=SimpleNamespace(suffix=SimpleNamespace(action_out_proj=layer))
    state=torch.random.get_rng_state().clone()
    probe=SafeFailureMonitor(expert,torch,path)
    assert torch.equal(torch.random.get_rng_state(),state)
    return probe,layer


def observe(probe,layer):
    probe.begin_request()
    for _ in range(5):layer(torch.ones(1,3,4))
    return probe.observe(5)


def test_alarm_uses_saved_band_once_and_resets_per_episode(tmp_path):
    probe,layer=monitor(tmp_path)
    try:
        first=observe(probe,layer)
        assert first['probability']==.5 and first['new_alarm']
        second=observe(probe,layer)
        assert second['cumulative_score']==1. and second['exceeded'] and not second['new_alarm']
        third=observe(probe,layer)
        assert third['threshold'] is None and not third['exceeded'] and not third['new_alarm']
        probe.reset()
        assert observe(probe,layer)['new_alarm']
    finally:probe.close()


def test_online_feature_matches_saved_corner_contract(tmp_path):
    probe,layer=monitor(tmp_path)
    try:
        inputs=[]
        hook=probe.model.projector.register_forward_pre_hook(lambda m,x:inputs.append(x[0].clone()))
        probe.begin_request()
        for i in range(5):layer(torch.arange(12.).reshape(1,3,4)+100*i)
        probe.observe(5)
        hook.remove()
        assert inputs[0].tolist()==[[400.,401.,402.,403.]]
    finally:probe.close()


def test_alarm_refresh_keeps_an_earlier_existing_deadline():
    d=dict(estimated_control_period_seconds=.05,latest_start=11.,
        first_refresh_cap=dict(latest_start=9.9),start_now=False)
    pending=dict(request_step=22,prediction_id=1)
    result=cap_refresh(d,pending,now=10.,step=20,current_profile='deep_idle')
    assert result['latest_start']==pytest.approx(10.05)
    assert result['first_refresh_cap']==d['first_refresh_cap']
    assert d['latest_start']==11.
    d['latest_start']=9.8
    assert cap_refresh(d,pending,now=10.,step=20,current_profile='baseline')['latest_start']==9.8


def test_safe_full_overrides_critical_fast_without_changing_normal_choice():
    from perceptbudget.ordinary_fast import choose,CRITICAL_POLICY
    d=dict(compute_profile='baseline',estimated_control_period_seconds=.05,
        service_seconds=.5,margin_seconds=.025,reason='action_event',
        deadline=12.,decision_monotonic=10.,latest_start=11.475,
        deadline_already_unreachable=False,rtc_delay_steps=10,start_constraint='service_deadline')
    costs=dict(cells=dict(baseline=dict(hold_power_w=8.,modes=dict(
        Full=dict(mean_seconds=.5,mean_energy_j=12.),Fast=dict(mean_seconds=.3,mean_energy_j=8.)))))
    kwargs=dict(step=20,previous_mode='Full',phases=[],events=[],costs=costs,
        fast_delay=6,policy=CRITICAL_POLICY,every_event=True)
    assert choose(d,**kwargs)['mode']=='Fast'
    selected=choose(dict(d,safe_refresh=dict(prediction_id=1)),**kwargs)
    assert selected['mode']=='Full' and selected['reason']=='safe_failure_probe'
