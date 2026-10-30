"""Service-only choice with separately recorded Continue forecasts.

Fast is planned from LOW, and all neural computation executes HIGH. No
post-service wait, guessed next recovery or invented terminal state is charged
to an inference. Whole-task efficiency is assessed from measured energy.
"""
import math

POLICY='m3_service_ledger_fast_advance_v4'


def choose_service(decision, *, observation_step, previous_mode, cost_model,
                   quantiles, quiet_age_seconds=0., latest_start_cap=None):
    from perceptbudget.m4_candidates import hold_energy
    now=decision['decision_monotonic'];deadline=decision['deadline']
    eta=decision['margin_seconds'];period=decision['estimated_control_period_seconds']
    profile=decision['current_profile']
    if profile not in ('baseline','gpu_emc') or previous_mode not in ('Full','Fast'):
        raise ValueError('Unknown device state or previous service')
    if not all(math.isfinite(v) for v in (now,deadline,eta,period,quiet_age_seconds)) or period<=0 or min(eta,quiet_age_seconds)<0:
        raise ValueError('Invalid scheduling inputs')
    require_full=bool(observation_step==0 or decision.get('fast_full_only') or decision['reason']=='action_event')
    candidates=[]
    for mode in ('Full','Fast'):
        entry_state='gpu_emc' if mode=='Fast' else profile
        cells=[c for c in cost_model['service'][mode].values() if c['from_profile']==entry_state]
        if not cells:raise ValueError('Missing measured service entry state')
        cell=max(cells,key=lambda c:c['energy_j'])
        q=float(quantiles[mode]);mean=float(cell['mean_seconds']);energy=float(cell['energy_j'])
        if not all(math.isfinite(v) and v>0 for v in (q,mean,energy)):
            raise ValueError('Invalid complete service measurements')
        raw=deadline-q-eta
        advance_steps=math.ceil((q+eta)/period-1e-9) if mode=='Fast' else 0
        # Original RTC submission boundary (e.g.39), minus Fast's complete
        # recovery-to-commit duration (e.g.7). Deadline subtraction occurs ONCE.
        cap=(latest_start_cap-advance_steps*period if mode=='Fast' else latest_start_cap) if latest_start_cap is not None else None
        latest=min(raw,cap) if cap is not None else raw
        wait=max(0.,latest-now)
        down=cost_model.get('transitions',{}).get('down',{})
        down_seconds=down.get('max_readback_seconds',down.get('seconds',0.))
        low_admission=(mode!='Fast' or profile=='gpu_emc' or
                       wait>=decision.get('minimum_low_idle_seconds',.2)+down_seconds)
        quality=mode=='Full' or (not require_full and previous_mode!='Fast' and low_admission)
        feasible=latest>=now-1e-9
        # Waiting is a separate ledger. A HIGH->LOW transition forecast is not
        # fabricated here: the runtime logs and meters that interval separately.
        waiting=hold_energy(cost_model['waits'],profile,quiet_age_seconds,wait)
        candidates.append(dict(mode=mode,entry_profile=entry_state,
            compute_profile='baseline',service_path=('recover_'+mode if entry_state=='gpu_emc' else 'hot_Full'),
            quality_allowed=quality,low_continue_admissible=low_admission,feasible=feasible,
            service_seconds=q,mean_service_seconds=mean,service_energy_j=energy,
            energy_j=energy,energy_scope='complete_service_only',
            formula_latest_start=raw,latest_start=latest,
            advance_steps=advance_steps,advance_seconds=advance_steps*period,
            original_rtc_latest_start=latest_start_cap,effective_rtc_latest_start=cap,
            start_constraint='rtc_prefix' if cap is not None and cap<raw else 'service_deadline',
            continue_before=dict(activity='Continue',seconds=wait,profile_at_decision=profile,
                forecast_energy_j=waiting,scope='current-state hold forecast; excluded from service ranking'),
            continue_after=dict(activity='Continue',forecast=False,reason='new chunk not observed; meter actual runtime'),
            recovery_included=entry_state=='gpu_emc',additional_transition_energy_j=0.))
    feasible=[c for c in candidates if c['quality_allowed'] and c['feasible']]
    chosen=min(feasible,key=lambda c:(c['service_energy_j'],c['mode']!='Full')) if feasible else candidates[0]
    return dict(policy=POLICY,mode=chosen['mode'],selected=chosen,candidates=candidates,
        reason='feasible_minimum_service_energy' if feasible else 'deadline_unreachable',
        observation_step=observation_step,window_start=now,window_end=deadline,
        margin_seconds=eta,current_profile=profile,previous_mode=previous_mode,
        quiet_age_seconds=quiet_age_seconds,latest_start_cap=latest_start_cap,
        requires_full=require_full,energy_scope='service_only; Continue accounted separately')
