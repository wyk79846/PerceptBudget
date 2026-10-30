"""Two perception candidates on M3 clocks: conditional deadlines and total energy."""
import math


def load_cost_model(config):
    """Reuse measured data, not the former M4 search or its candidate templates."""
    import json
    from pathlib import Path
    from perceptbudget.m4_costs import load_service_costs,load_transitions
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
    measured=load_transitions(config['directional_cost_source'])
    transitions={'down':measured['baseline','gpu_emc'],'up':measured['gpu_emc','baseline']}
    return dict(service=service,waits=waits,wait_source=config['wait_cost_source'],
                continue_energy_model='measured_roundtrip_v1',transitions=transitions,
                scope='Fixed high-frequency compute; measured complete service includes recovery')


def continue_energy(costs, seconds, *, minimum_low_seconds=.2):
    """Price a Continue interval, including measured down/up preparation windows.

    Both candidates start and end high. The 250ms measured preparation cells
    already contain hold energy: their durations must not be charged twice.
    This is a cost forecast, not a promise about unobserved future events.
    """
    from perceptbudget.m4_candidates import hold_energy
    high=hold_energy(costs['waits'],'baseline',0.,seconds)
    result=dict(activity='Continue',path='high_hold',seconds=seconds,
        hold_energy_j=high,transition_energy_j=0.,energy_j=high)
    if costs.get('continue_energy_model')!='measured_roundtrip_v1':return result
    down,up=(costs['transitions'][k] for k in ('down','up'))
    low_seconds=seconds-down['seconds']-up['seconds']
    if low_seconds<minimum_low_seconds:return result
    def transition_energy(cell,age):
        reference=cell.get('reference_quiet_age_seconds',0.)
        extra=max(0.,hold_energy(costs['waits'],cell['source'],age,cell['seconds'])-
                  hold_energy(costs['waits'],cell['source'],reference,cell['seconds']))
        return cell['energy_j']+extra
    switching=transition_energy(down,0.)+transition_energy(up,down['seconds']+low_seconds)
    holding=hold_energy(costs['waits'],'gpu_emc',down['seconds'],low_seconds)
    if switching+holding<high:
        result=dict(activity='Continue',path='low_hold_roundtrip',seconds=seconds,
            down_seconds=down['seconds'],low_seconds=low_seconds,up_seconds=up['seconds'],
            hold_energy_j=holding,transition_energy_j=switching,energy_j=switching+holding)
    return result


def choose_mode(decision, *, observation_step, previous_mode='Full', cost_model,
                quantiles, quiet_age_seconds=0., latest_start_cap=None):
    """Evaluate Full/Fast on the same [now, deadline] window.

    c and post-service state are fixed high. Before preparation, retain the
    current M3 state; replan when that state changes. Complete service costs
    already include any measured recovery, so no second transition is charged.
    """
    if decision.get('fast_policy')=='m3_service_ledger_fast_advance_v4':
        from perceptbudget.m3_service_choice import choose_service
        return choose_service(decision,observation_step=observation_step,previous_mode=previous_mode,
            cost_model=cost_model,quantiles=quantiles,quiet_age_seconds=quiet_age_seconds,latest_start_cap=latest_start_cap)
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
        base_latest=latest
        reserve_steps=0
        if mode=='Fast' and decision.get('fast_reserve_full',False):
            period=decision['estimated_control_period_seconds']
            full_budget=decision['fast_full_reserve_service_seconds']
            if not math.isfinite(full_budget) or full_budget<=0 or not math.isfinite(period) or period<=0:
                raise ValueError('Full reservation requires positive service budget and control period')
            reserve_steps=math.ceil((full_budget+eta)/period-1e-9)
            latest-=reserve_steps*period
        start=max(now,latest)
        quality=(mode=='Full' or not requires_full and previous_mode!='Fast')
        feasible=latest>=now-1e-9 and start+mean<=deadline+1e-9
        before=max(0.,start-now);after=max(0.,deadline-start-mean)
        wait_energy=hold_energy(cost_model['waits'],profile,quiet_age_seconds,before)
        tail=continue_energy(cost_model,after,minimum_low_seconds=decision.get('minimum_low_idle_seconds',.2))
        tail_energy=tail['energy_j']
        proposals.append(dict(mode=mode,quality_allowed=quality,feasible=feasible,
            quality_reason=('full_only_ablation' if decision.get('fast_full_only',False) else 'full_required_by_event' if requires_full else 'validated_single_fast_then_full'),
            service_seconds=q,mean_service_seconds=mean,latest_start=latest,
            formula_latest_start=raw_latest,base_latest_start=base_latest,
            full_reserve_steps=reserve_steps,full_reserve_seconds=reserve_steps*decision['estimated_control_period_seconds'],
            prepare_start=start,window_end=deadline,
            wait_seconds=before,tail_seconds=after,tail_activity='Continue',continue_plan=tail,wait_energy_j=wait_energy,
            service_energy_j=energy,tail_energy_j=tail_energy,
            additional_transition_energy_j=0.,recovery_included=profile!='baseline',
            energy_j=wait_energy+energy+tail_energy,
            start_constraint='rtc_prefix' if base_latest<raw_latest else 'service_deadline'))
    available=[p for p in proposals if p['quality_allowed'] and p['feasible']]
    if available:
        chosen=min(available,key=lambda p:(p['energy_j'],p['mode']!='Full'))
        status='feasible_minimum_energy'
    else:
        # Record the missed timing constraint; do not label the fallback feasible
        # or extend the old action horizon. Existing underflow handling remains.
        chosen=(next(p for p in proposals if p['mode']=='Full') if decision.get('fast_reserve_full',False) else
                min((p for p in proposals if p['quality_allowed']),key=lambda p:p['service_seconds']))
        status='deadline_unreachable'
    return dict(observation_step=observation_step,mode=chosen['mode'],reason=status,selected=chosen,candidates=proposals,
        window_start=now,window_end=deadline,margin_seconds=eta,
        current_profile=profile,quiet_age_seconds=quiet_age_seconds,
        latest_start_cap=latest_start_cap,previous_mode=previous_mode,
        requires_full=requires_full,policy=decision.get('fast_policy','m3_two_mode_deadline_energy_v2'))
