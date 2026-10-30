from copy import deepcopy

import pytest

from perceptbudget.m3_fast_choice import choose_mode


def make_decision(**updates):
    decision = dict(
        reason='buffer',
        critical_step=None,
        service_seconds=.08,
        margin_seconds=.07,
        estimated_control_period_seconds=.05,
    )
    decision.update(updates)
    return decision


def test_bootstrap_is_full_even_after_fast_mode():
    result = choose_mode(make_decision(), observation_step=0, previous_mode='Fast')

    assert result == dict(mode='Full', reason='bootstrap_full')


def test_action_event_uses_full_without_requiring_budget_fields():
    result = choose_mode({'reason': 'action_event'}, observation_step=12, previous_mode='Fast')

    assert result == dict(mode='Full', reason='action_event_full')


def test_critical_step_inclusive_ceil_budget_boundary_uses_full():
    # (.08 + .07) / .05 rounds up to three steps; service alone covers two.
    decision = make_decision(critical_step=23)

    result = choose_mode(decision, observation_step=20, previous_mode='Fast')

    assert result == dict(mode='Full', reason='critical_within_full_budget')


def test_critical_step_just_outside_budget_uses_fast():
    decision = make_decision(critical_step=24)

    result = choose_mode(decision, observation_step=20)

    assert result == dict(mode='Fast', reason='m3_refresh_fast')


def test_past_critical_step_does_not_count_as_within_budget():
    decision = make_decision(critical_step=19)

    result = choose_mode(decision, observation_step=20)

    assert result == dict(mode='Fast', reason='m3_refresh_fast')


def test_fast_is_followed_by_full_then_fast_can_recur():
    decision = make_decision()

    first = choose_mode(decision, observation_step=1, previous_mode='Full')
    second = choose_mode(decision, observation_step=2, previous_mode=first['mode'])
    third = choose_mode(decision, observation_step=3, previous_mode=second['mode'])

    assert [first['mode'], second['mode'], third['mode']] == ['Fast', 'Full', 'Fast']
    assert [first['reason'], second['reason'], third['reason']] == [
        'm3_refresh_fast', 'return_to_full_after_fast', 'm3_refresh_fast'
    ]


def test_previous_fast_returns_to_full_when_other_rules_do_not_apply():
    result = choose_mode(make_decision(), observation_step=1, previous_mode='Fast')

    assert result == dict(mode='Full', reason='return_to_full_after_fast')


def test_decision_is_not_modified():
    decision = make_decision(critical_step=23, metadata={'source': ['runtime']})
    original = deepcopy(decision)

    choose_mode(decision, observation_step=20)

    assert decision == original


@pytest.mark.parametrize('change', [
    {'observation_step': -1},
    {'observation_step': 1.5},
])
def test_invalid_observation_step_is_rejected(change):
    with pytest.raises(ValueError):
        choose_mode(make_decision(), **change)


def test_invalid_previous_mode_is_rejected():
    with pytest.raises(ValueError):
        choose_mode(make_decision(), observation_step=1, previous_mode='Continue')


def test_critical_budget_requires_actual_timing_plan_fields():
    decision = {'reason': 'buffer', 'critical_step': 5}

    with pytest.raises(ValueError, match='finite service, margin and control-period'):
        choose_mode(decision, observation_step=1)
