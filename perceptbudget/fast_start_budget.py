"""Opt-in Fast service budgets; preserve the existing deadlines and RTC caps."""
from perceptbudget.deadline_controller import percentile

POLICY = 'critical_fast_own_budget_v1'


def select(formula, decision, *, choices, fast_delays, costs, probability, prefix_caps):
    """Use observed Fast delay plus a commit tick, with the measured prior.

    Clock transitions remain explicit. Full candidates and the original Full
    prefix bounds are retained; only eligible action-event Fast gets a shorter
    service budget. No outcome, task identifier or seed enters this formula.
    """
    if decision['reason'] != 'action_event':
        return formula
    period = decision['estimated_control_period_seconds']
    end = decision['deadline'] - decision['margin_seconds']
    rows = []
    for original in formula['candidates']:
        row = dict(original)
        profile = row['profile']
        if choices[profile]['mode'] == 'Fast':
            prior = percentile(costs['cells'][profile]['modes']['Fast']['samples_seconds'], probability)
            compute = max(prior, fast_delays[profile] * period) + period
            transition = (row['switch_seconds'] + row['ack_reserve_seconds']
                          if decision['current_profile'] != profile else 0.)
            service = min(original['service_seconds'], compute + transition)
            row.update(service_seconds=service, compute_seconds=service-transition,
                       fast_delay_steps=fast_delays[profile],
                       fast_prior_seconds=prior, mode='Fast', rtc_cap=prefix_caps[profile])
            row['latest_start'] = min(end-service, row['rtc_cap'], row['first_cap'])
            row['feasible'] = row['latest_start'] >= decision['decision_monotonic'] - 1e-9
        else:
            row['mode'] = 'Full'
        rows.append(row)
    if not any(row['mode'] == 'Fast' for row in rows):
        return formula
    chosen = rows[0] if rows[0]['feasible'] else rows[1]
    return dict(formula, policy=POLICY, full_candidates=formula['candidates'],
                candidates=rows, selected=chosen,
                reason='medium_meets_deadline' if chosen is rows[0] else
                       'high_medium_too_late' if chosen['feasible'] else 'high_emergency')
