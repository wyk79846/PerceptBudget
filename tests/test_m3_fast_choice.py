from copy import deepcopy
import pytest
from perceptbudget.m3_fast_choice import choose_mode


def model():
    return dict(waits={'baseline':[5.]*4,'gpu_emc':[2.]*4},service={
        mode:{'hot':dict(energy_j=energy,mean_seconds=duration,from_profile='baseline'),
              'short':dict(energy_j=energy+1,mean_seconds=duration+.2,from_profile='gpu_emc'),
              'long':dict(energy_j=energy,mean_seconds=duration+.1,from_profile='gpu_emc')}
        for mode,energy,duration in [('Full',12.,2.),('Fast',5.,1.)]})


def decision(**changes):
    return dict(dict(decision_monotonic=0.,deadline=10.,margin_seconds=.1,
        current_profile='baseline',reason='buffer',critical_step=None,
        estimated_control_period_seconds=.05),**changes)


def choose(d=None,c=None,**kw):
    return choose_mode(d or decision(),observation_step=20,cost_model=c or model(),
        quantiles=kw.pop('quantiles',{'Full':2.,'Fast':1.}),**kw)


def test_both_candidates_use_own_latest_start_and_common_energy_window():
    result=choose();a,b=result['candidates']
    assert a['latest_start']==pytest.approx(7.9)
    assert b['latest_start']==pytest.approx(8.9)
    assert a['energy_j']==pytest.approx(52.)
    assert b['energy_j']==pytest.approx(50.)
    assert a['window_end']==b['window_end']==10.
    assert result['mode']=='Fast'


def test_more_expensive_fast_is_not_selected_just_because_it_is_fast():
    costs=model();costs['service']['Fast']['hot']['energy_j']=8.
    assert choose(c=costs)['mode']=='Full'


def test_infeasible_full_is_filtered_before_energy_comparison():
    result=choose(decision(deadline=2.),quantiles={'Full':3.,'Fast':1.})
    assert result['mode']=='Fast' and not result['candidates'][0]['feasible']


def test_critical_operation_requires_full_and_validated_quality_limits_fast():
    assert choose(decision(reason='action_event'))['mode']=='Full'
    assert choose(decision(critical_step=21))['mode']=='Full'
    assert choose(previous_mode='Fast')['mode']=='Full'


def test_recovery_energy_is_in_complete_service_not_added_twice():
    result=choose(decision(current_profile='gpu_emc'),quantiles={'Full':2.2,'Fast':1.2})
    r=result['selected']
    assert r['recovery_included'] and r['additional_transition_energy_j']==0.
    assert r['service_energy_j']==6.
    assert r['energy_j']==pytest.approx(8.7*2+6+.1*5)


def test_rtc_constraint_caps_prepare_time_and_prices_new_wait():
    result=choose(latest_start_cap=3.)
    assert all(r['latest_start']==3. and r['start_constraint']=='rtc_prefix' for r in result['candidates'])
    assert all(r['wait_seconds']==3. for r in result['candidates'])


def test_no_feasible_mode_is_logged_not_declared_safe():
    result=choose(decision(deadline=.5))
    assert result['reason']=='deadline_unreachable'
    assert not result['selected']['feasible']
    assert not any(r['feasible'] for r in result['candidates'])


def test_input_and_cost_measurements_are_not_modified():
    d,c=decision(),model();before=deepcopy((d,c));choose(d,c)
    assert (d,c)==before


def test_independent_audit_rejects_forged_energy_or_selected_budget():
    from scripts.audit_rtc_system import audit_fast_formula
    d=decision();c=model();result=choose(d,c)
    selected=result['selected']
    applied=dict(d,service_seconds=selected['service_seconds'],latest_start=selected['latest_start'])
    assert audit_fast_formula(result,applied,c,'Full')==[]
    changed=deepcopy(result);changed['candidates'][0]['energy_j']=0.
    assert 'absolute_energy_once' in audit_fast_formula(changed,applied,c,'Full')
    applied['service_seconds']+=.1
    assert 'selected_budget_applied' in audit_fast_formula(result,applied,c,'Full')
