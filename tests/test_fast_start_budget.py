from copy import deepcopy

import pytest

from perceptbudget.fast_start_budget import POLICY, select
from perceptbudget.three_frequency import select_full


def scenario(reason='action_event', now=10.):
    decision=dict(reason=reason,decision_monotonic=now,deadline=11.,margin_seconds=.025,
        estimated_control_period_seconds=.05,service_seconds=.6,latest_start=10.375,
        current_profile='deep_idle')
    formula=select_full(decision,medium_seconds=.75,medium_cap=12.,current_profile='deep_idle',
        transitions=dict(medium_seconds=.003,high_seconds=.018,high_to_medium_seconds=.018))
    kwargs=dict(choices={p:dict(mode='Fast') for p in ('baseline','gpu_emc')},
        fast_delays=dict(gpu_emc=10,baseline=7),
        costs=dict(cells={p:dict(modes=dict(Fast=dict(samples_seconds=[s]*12)))
            for p,s in [('gpu_emc',.5),('baseline',.35)]}),probability=.95,
        prefix_caps=dict(gpu_emc=12.,baseline=12.))
    return decision,formula,kwargs


def test_fast_can_observe_later_without_moving_deadline_or_changing_full_budget():
    decision,formula,kwargs=scenario()
    saved=deepcopy(formula)
    result=select(formula,decision,**kwargs)
    assert formula==saved
    assert result['policy']==POLICY and result['deadline']==decision['deadline']
    assert result['selected']['profile']=='gpu_emc'
    assert result['selected']['latest_start']>formula['selected']['latest_start']
    assert result['selected']['latest_start']+result['selected']['service_seconds']==pytest.approx(10.975)
    assert result['full_candidates']==formula['candidates']


def test_mode_budget_keeps_medium_feasible_after_full_medium_window_has_passed():
    decision,formula,kwargs=scenario(now=10.3)
    assert formula['selected']['profile']=='baseline'
    result=select(formula,decision,**kwargs)
    assert result['selected']['profile']=='gpu_emc'
    assert result['selected']['feasible']


def test_ordinary_full_and_mandatory_full_after_fast_retain_original_timing():
    decision,formula,kwargs=scenario(reason='buffer')
    assert select(formula,decision,**kwargs) is formula
    decision['reason']='action_event'
    kwargs['choices']={p:dict(mode='Full') for p in ('baseline','gpu_emc')}
    assert select(formula,decision,**kwargs) is formula


def test_later_fast_start_cannot_overrun_rtc_prefix_or_validated_action_window():
    decision,formula,kwargs=scenario()
    kwargs['prefix_caps']['gpu_emc']=10.25
    formula['candidates'][0]['first_cap']=10.24
    result=select(formula,decision,**kwargs)
    assert result['candidates'][0]['latest_start']==10.24
    assert result['candidates'][0]['rtc_cap']==10.25
