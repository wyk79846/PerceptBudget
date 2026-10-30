"""Two perception candidates on M3 clocks: conditional deadlines and total energy."""
import math


def load_cost_model(config):
    """Reuse measured data, not the former M4 search or its candidate templates."""
    import json
    from pathlib import Path
    from perceptbudget.m4_costs import load_service_costs
    from perceptbudget.m4_candidates import wait_model
    rows=load_service_costs(config)
    service={mode:{} for mode in ('Full','Fast')}
    for mode in service:
        for history in ('hot','short','long'):
            r=next(x for x in rows if x['mode']==mode and x['profile']=='baseline' and x['history']==history)
            service[mode][history]=dict(energy_j=r['mean_absolute_service_energy_j'],
                mean_seconds=r['mean_service_seconds'],samples_seconds=r['service_samples_seconds'],
                from_profile=r['from_profile'],source=r['source'])
    waits=wait_model(json.loads(Path(config['wait_cost_source']).read_text()))
    return dict(service=service,waits=waits,wait_source=config['wait_cost_source'],
                scope='Fixed high-frequency compute; measured complete service includes recovery')


def choose_mode(decision, *, observation_step, previous_mode='Full', cost_model,
                quantiles, quiet_age_seconds=0., latest_start_cap=None):
    """Evaluate Full/Fast on the same [now, deadline] window.

    c and post-service state are fixed high. Before preparation, retain the
    current M3 state; replan when that state changes. Complete service costs
    already include any measured recovery, so no second transition is charged.
    """
    from perceptbudget.m4_candidates import hold_energy
    now=decision['decision_monotonic'];deadline=decision['deadline']
    eta=decision['margin_seconds'];profile=decision['current_profile']
    if previous_mode not in ('Full','Fast') or profile not in ('baseline','gpu_emc'):
        raise ValueError('Unknown previous mode or current device state')
    if not all(math.isfinite(x) for x in (now,deadline,eta,quiet_age_seconds)) or eta<0 or quiet_age_seconds<0:
        raise ValueError('Invalid deadline/energy inputs')
    critical=decision.get('critical_step')
    critical_deadline=(now+(critical-observation_step)*decision['estimated_control_period_seconds']
                       if critical is not None else None)
    requires_full=(decision.get('fast_full_only',False) or observation_step==0 or decision['reason']=='action_event'
                   or critical_deadline is not None and critical_deadline<=deadline+1e-9)
    proposals=[]
    for mode in ('Full','Fast'):
        measured=cost_model['service'][mode]
        entry=[r for r in measured.values() if r['from_profile']==profile]
        if not entry:raise ValueError('No measured service from current device state')
        # Short/long are measurement histories, not extra online state choices.
        # Use the larger measured energy with its own paired mean duration.
        cell=max(entry,key=lambda r:r['energy_j'])
        q=float(quantiles[mode]);mean=float(cell['mean_seconds']);energy=float(cell['energy_j'])
        if not all(math.isfinite(x) and x>0 for x in (q,mean,energy)):
            raise ValueError('Invalid measured service budget')
        raw_latest=deadline-q-eta
        latest=min(raw_latest,latest_start_cap) if latest_start_cap is not None else raw_latest
        start=max(now,latest)
        quality=(mode=='Full' or not requires_full and previous_mode!='Fast')
        feasible=latest>=now-1e-9 and start+mean<=deadline+1e-9
        before=max(0.,start-now);after=max(0.,deadline-start-mean)
        wait_energy=hold_energy(cost_model['waits'],profile,quiet_age_seconds,before)
        tail_energy=hold_energy(cost_model['waits'],'baseline',0.,after)
        proposals.append(dict(mode=mode,quality_allowed=quality,feasible=feasible,
            quality_reason=('full_only_ablation' if decision.get('fast_full_only',False) else 'full_required_by_event' if requires_full else 'validated_single_fast_then_full'),
            service_seconds=q,mean_service_seconds=mean,latest_start=latest,
            formula_latest_start=raw_latest,prepare_start=start,window_end=deadline,
            wait_seconds=before,tail_seconds=after,wait_energy_j=wait_energy,
            service_energy_j=energy,tail_energy_j=tail_energy,
            additional_transition_energy_j=0.,recovery_included=profile!='baseline',
            energy_j=wait_energy+energy+tail_energy,
            start_constraint='rtc_prefix' if latest<raw_latest else 'service_deadline'))
    available=[p for p in proposals if p['quality_allowed'] and p['feasible']]
    if available:
        chosen=min(available,key=lambda p:(p['energy_j'],p['mode']!='Full'))
        status='feasible_minimum_energy'
    else:
        # Record the missed timing constraint; do not label the fallback feasible
        # or extend the old action horizon. Existing underflow handling remains.
        chosen=min((p for p in proposals if p['quality_allowed']),key=lambda p:p['service_seconds'])
        status='deadline_unreachable'
    return dict(observation_step=observation_step,mode=chosen['mode'],reason=status,selected=chosen,candidates=proposals,
        window_start=now,window_end=deadline,margin_seconds=eta,
        current_profile=profile,quiet_age_seconds=quiet_age_seconds,
        latest_start_cap=latest_start_cap,previous_mode=previous_mode,
        requires_full=requires_full,policy=decision.get('fast_policy','m3_two_mode_deadline_energy_v2'))
