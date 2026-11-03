"""Substitute one ordinary Full request; retain the complete Full timing plan."""
import json
import math
from pathlib import Path

POLICY = 'ordinary_fast_same_full_schedule_v1'


def load_costs(path, config):
    path=Path(path)
    costs=json.loads(path.read_text())
    recorded=json.loads((path.parent/'run.json').read_text())['config_snapshot']
    if costs.get('status')!='completed' or not costs.get('restored'):
        raise ValueError('Ordinary Fast needs completed restored clock costs')
    for field in ('H','N','precision','checkpoint_provenance_revision','rtc_beta'):
        if recorded[field]!=config[field]:raise ValueError('Fast cost model mismatch: '+field)
    for profile in ('baseline','gpu_emc'):
        cell=costs['cells'][profile]
        for mode in ('Full','Fast'):
            sample=cell['modes'][mode]
            values=[cell['hold_power_w'],sample['mean_energy_j'],sample['mean_seconds'],*sample['samples_seconds']]
            if len(sample['samples_seconds'])<4 or any(not math.isfinite(v) or v<=0 for v in values):
                raise ValueError('Missing positive measured Full/Fast costs')
    return dict(costs,source=str(path))


def choose(decision, *, step, previous_mode, phases, events, costs, fast_delay):
    """The same clock preparation and initial waiting cancel between candidates.

    Compare absolute request energy plus the remaining hold on the SAME window.
    Hold is an explicit Continue term, never part of the Fast request energy.
    This fixed-input ranking prior is checked against measured closed-loop J.
    """
    profile=decision['compute_profile'];cell=costs['cells'][profile]
    period=decision['estimated_control_period_seconds']
    near=[e for e in events if e['target_step']>=step and
          (e['target_step']-step)*period<=decision['service_seconds']+decision['margin_seconds']]
    active=[p for p in phases if p['start_step']<=step<p['stop_step']]
    ordinary=decision['reason']=='buffer' and not near and not active
    window=max(0.,decision['deadline']-decision['margin_seconds']-
               max(decision['decision_monotonic'],decision['latest_start']))
    rows=[]
    for mode in ('Full','Fast'):
        c=cell['modes'][mode];tail=max(0.,window-c['mean_seconds'])
        rows.append(dict(mode=mode,service_energy_j=c['mean_energy_j'],
            service_mean_seconds=c['mean_seconds'],continue_seconds=tail,
            continue_energy_j=tail*cell['hold_power_w'],
            comparable_window_energy_j=c['mean_energy_j']+tail*cell['hold_power_w']))
    allowed=(ordinary and previous_mode!='Fast' and
             not decision['deadline_already_unreachable'] and
             fast_delay<=decision['rtc_delay_steps'] and
             rows[1]['service_mean_seconds']<=rows[0]['service_mean_seconds'])
    saving=rows[0]['comparable_window_energy_j']-rows[1]['comparable_window_energy_j']
    selected='Fast' if allowed and saving>0 else 'Full'
    return dict(policy=POLICY,mode=selected,
        reason='ordinary_cheaper_fast' if selected=='Fast' else
               'full_after_fast' if previous_mode=='Fast' else
               'full_key_stage' if not ordinary else 'full_cost_or_timing',
        observation_step=step,previous_mode=previous_mode,ordinary=ordinary,
        active_phases=active,near_events=near,fast_delay_steps=fast_delay,
        profile=profile,window_seconds=window,candidates=rows,
        estimated_saving_j=saving,cost_source=costs.get('source'),
        shared_preparation='Same start and selected frequency; common preparation energy cancels',
        hold_scope='Measured fixed-input hold power; ranking prior, task energy measured separately',
        full_schedule={k:decision[k] for k in ('deadline','latest_start','service_seconds',
            'margin_seconds','compute_profile','rtc_delay_steps','start_constraint')})
