from copy import deepcopy
import pytest
from perceptbudget.m3_service_choice import choose_service
from scripts.audit_rtc_system import audit_service_formula


def fixture(profile='baseline'):
    costs=dict(waits={'baseline':[10.]*4,'gpu_emc':[3.]*4},
        transitions={'down':dict(seconds=.25,max_readback_seconds=.02)},
        service={mode:{h:dict(from_profile=state,energy_j=energy,mean_seconds=mean)
          for h,state,energy in [('hot','baseline',hot),('short','gpu_emc',low),('long','gpu_emc',low)]}
          for mode,hot,low,mean in [('Full',12.,9.,.5),('Fast',100.,5.,.3)]})
    d=dict(decision_monotonic=0.,deadline=2.5,margin_seconds=.025,
        estimated_control_period_seconds=.05,current_profile=profile,reason='buffer',fast_full_only=False)
    return costs,d


def choose(profile='baseline',**kwargs):
    costs,d=fixture(profile);d.update(kwargs)
    return choose_service(d,observation_step=10,previous_mode='Full',cost_model=costs,
        quantiles={'Full':.55,'Fast':.35},latest_start_cap=1.95),d,costs


def test_original39_is_advanced_by_fast8_not_full12_and_no_double_subtraction():
    result,d,costs=choose()
    assert result['mode']=='Fast'
    c=result['selected']
    assert c['advance_steps']==8 and c['latest_start']==pytest.approx(31*.05)
    assert c['entry_profile']=='gpu_emc' and c['service_energy_j']==5.
    assert c['energy_j']==c['service_energy_j']
    assert c['continue_after']['forecast'] is False
    applied=dict(d,service_seconds=c['service_seconds'],latest_start=c['latest_start'])
    assert audit_service_formula(result,applied,costs,'Full')==[]


def test_lowering_clock_does_not_replace_fast_cost_with_hot_fast():
    high,_,_=choose();low,_,_=choose('gpu_emc')
    assert high['mode']==low['mode']=='Fast'
    assert high['selected']['service_energy_j']==low['selected']['service_energy_j']==5.
    assert high['selected']['latest_start']==low['selected']['latest_start']


def test_no_cap_subtracts_fast_service_only_once():
    costs,d=fixture('gpu_emc')
    r=choose_service(d,observation_step=10,previous_mode='Full',cost_model=costs,
        quantiles={'Full':.55,'Fast':.35})
    assert r['selected']['latest_start']==pytest.approx(2.125)


def test_continue_cost_does_not_get_ranked_as_inference_energy():
    costs,d=fixture();a=choose_service(d,observation_step=10,previous_mode='Full',cost_model=costs,
        quantiles={'Full':.55,'Fast':.35})
    costs['waits']['baseline']=[1000.]*4
    b=choose_service(d,observation_step=10,previous_mode='Full',cost_model=costs,
        quantiles={'Full':.55,'Fast':.35})
    assert a['mode']==b['mode']=='Fast'
    assert a['selected']['energy_j']==b['selected']['energy_j']
    assert a['selected']['continue_before']['forecast_energy_j']!=b['selected']['continue_before']['forecast_energy_j']


def test_short_high_wait_cannot_produce_hot_fast_and_no_late_fast_fallback():
    r,d,costs=choose(decision_monotonic=1.4)
    assert r['mode']=='Full' and not r['candidates'][1]['low_continue_admissible']
    r,_,_=choose('gpu_emc',decision_monotonic=2.5)
    assert r['mode']=='Full' and r['reason']=='deadline_unreachable'


def test_fast_after_fast_forbidden_and_event_requires_full():
    costs,d=fixture('gpu_emc')
    for previous,event in [('Fast','buffer'),('Full','action_event')]:
        r=choose_service(dict(d,reason=event),observation_step=10,previous_mode=previous,
            cost_model=costs,quantiles={'Full':.55,'Fast':.35})
        assert r['mode']=='Full'
