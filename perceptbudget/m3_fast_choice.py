"""M3 Full/Fast selection based on the existing deadline decision."""

import math
from collections.abc import Mapping


def choose_mode(decision, *, observation_step, previous_mode='Full'):
    """Choose the compute mode for an M3 refresh without changing its decision.

    Event refreshes and bootstrap always use Full. A nearby critical step also
    uses Full when it falls within the existing service-plus-margin budget.
    After a Fast refresh, the next refresh returns to Full.
    """
    if not isinstance(decision, Mapping):
        raise TypeError('decision must be a mapping')
    if type(observation_step) is not int or observation_step < 0:
        raise ValueError('observation_step must be a nonnegative integer')
    if previous_mode not in ('Full', 'Fast'):
        raise ValueError("previous_mode must be 'Full' or 'Fast'")

    if observation_step == 0:
        return dict(mode='Full', reason='bootstrap_full')

    if decision.get('reason') == 'action_event':
        return dict(mode='Full', reason='action_event_full')

    critical_step = decision.get('critical_step')
    if critical_step is not None:
        if type(critical_step) is not int or critical_step < 0:
            raise ValueError('critical_step must be a nonnegative integer or None')
        if critical_step >= observation_step:
            service_seconds = decision.get('service_seconds')
            margin_seconds = decision.get('margin_seconds')
            period_seconds = decision.get('estimated_control_period_seconds')
            values = (service_seconds, margin_seconds, period_seconds)
            if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in values):
                raise ValueError('Decision must contain finite service, margin and control-period seconds')
            if service_seconds <= 0 or margin_seconds < 0 or period_seconds <= 0:
                raise ValueError('Decision contains an invalid service budget, margin or control period')

            # Match the controller's step-boundary roundoff convention.
            covered_steps = math.ceil((service_seconds + margin_seconds) / period_seconds - 1e-9)
            if critical_step - observation_step <= covered_steps:
                return dict(mode='Full', reason='critical_within_full_budget')

    if previous_mode == 'Fast':
        return dict(mode='Full', reason='return_to_full_after_fast')

    return dict(mode='Fast', reason='m3_refresh_fast')
