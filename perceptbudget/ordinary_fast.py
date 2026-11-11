"""Substitute one ordinary Full request; retain the complete Full timing plan."""
import json
import math
from pathlib import Path

POLICY = 'ordinary_fast_same_full_schedule_v1'
FEEDBACK_POLICY_V2 = 'ordinary_fast_feedback_buffer_v2'
FEEDBACK_POLICY = 'ordinary_fast_feedback_buffer_v3'
CRITICAL_POLICY = 'critical_fast_feedback_buffer_v1'
ALTERNATING_POLICY = 'full_fast_alternating_state_v1'
ALTERNATING_NO_STATE_POLICY = 'full_fast_alternating_no_state_v1'


def load_costs(path, config):
    path=Path(path)
    costs=json.loads(path.read_text())
    recorded=json.loads((path.parent/'run.json').read_text())['config_snapshot']
    if costs.get('status')!='completed' or not costs.get('restored'):
        raise ValueError('Ordinary Fast needs completed restored clock costs')
    for field in ('H','N','precision','checkpoint_provenance_revision','rtc_beta'):
        if recorded[field]!=config[field]:raise ValueError('Fast cost model mismatch: '+field)
    contract=costs.get('fast_input_contract',dict(visual_mode='selective320',visual_tokens=320,state_enabled=False))
    mode=config.get('m3_ordinary_fast_visual_mode','selective320')
    if mode not in ('selective320','selective464','wrist','pool2_agent'):
        raise ValueError('Unsupported ordinary Fast input mode')
    expected=dict(visual_mode=mode,visual_tokens=256 if mode=='wrist' else 320 if mode=='pool2_agent' else int(mode[9:]),
        state_enabled=not config.get('m2_fast_no_state',mode!='wrist'))
    proxy=(mode=='pool2_agent' and contract.get('visual_mode')=='selective320'
        and (config.get('m3_fast_alternation') or config.get('m3_fast_stable') or config.get('m3_fast_critical')) and contract.get('visual_tokens')==320
        and contract.get('state_enabled') is False)
    if not proxy and any(contract.get(k)!=v for k,v in expected.items()):
        raise ValueError('Fast cost input geometry/state mismatch')
    positions='original_camera_and_text_coordinates_v1' if config.get('m3_ordinary_fast_positions') else 'contiguous_v1'
    if not proxy and contract.get('position_policy','contiguous_v1')!=positions:
        raise ValueError('Fast cost position policy mismatch')
    for profile in ('baseline','gpu_emc'):
        cell=costs['cells'][profile]
        for mode in ('Full','Fast'):
            sample=cell['modes'][mode]
            values=[cell['hold_power_w'],sample['mean_energy_j'],sample['mean_seconds'],*sample['samples_seconds']]
            if len(sample['samples_seconds'])<4 or any(not math.isfinite(v) or v<=0 for v in values):
                raise ValueError('Missing positive measured Full/Fast costs')
    return dict(costs,source=str(path),cost_geometry_scope=(
        '320-token stateless costs reused as an initial external64/full-wrist prior, not measured stateful pool2_agent energy. Actual request timing and whole-task energy are recorded separately.'
        if proxy else 'matching measured input contract'))


def choose(decision, *, step, previous_mode, phases, events, costs, fast_delay,
           policy=POLICY, match_steps=5, full_per_fast=1, full_streak=None, every_event=False):
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
    protected=[]
    if policy in (FEEDBACK_POLICY_V2,FEEDBACK_POLICY,CRITICAL_POLICY):
        # The energy controller has already removed predictable motion cues.
        # Do not resurrect them through the legacy raw phase detector. Keep a
        # local gripper margin; every unhandled future key still chooses Full
        # through the unchanged action-event deadline.
        near=[e for e in events if e['kind'].startswith('gripper_')
              and abs(e['target_step']-step)<=match_steps]
        if policy in (FEEDBACK_POLICY,CRITICAL_POLICY):
            # A predictable bend may be ordinary while a bounded grasp/place
            # operation is still active. Keep only gripper-anchored intervals;
            # do not re-enable all legacy motion-phase refreshes.
            protected=[p for p in active if any(a['kind'].startswith('gripper_')
                for a in p.get('anchors',[]))]
        ordinary=decision['reason']=='buffer' and not near and not protected
    elif policy not in (POLICY,ALTERNATING_POLICY,ALTERNATING_NO_STATE_POLICY):
        raise ValueError('Unknown ordinary Fast policy')
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
    streak=int(previous_mode=='Full') if full_streak is None else full_streak
    if policy in (ALTERNATING_POLICY,ALTERNATING_NO_STATE_POLICY):
        # Strict cadence counts completed Full requests, including bootstrap.
        # No phase, quality, energy or timing admission gate changes the order.
        # Scheduling and clock selection still use the complete Full plan.
        if full_per_fast not in (1,2):raise ValueError('Use one or two Full requests per Fast')
        selected='Fast' if streak>=full_per_fast else 'Full'
    if policy==CRITICAL_POLICY:
        # Invert only the existing stable/critical admission. A missed Full
        # deadline must not forbid the faster critical response. Keep the Full
        # schedule, RTC reservation and no-consecutive-Fast rule unchanged.
        selected='Fast' if (not ordinary and previous_mode!='Fast'
            and fast_delay<=decision['rtc_delay_steps']
            and rows[1]['service_mean_seconds']<=rows[0]['service_mean_seconds']) else 'Full'
    approach=(decision.get('reason')=='action_event'
              and decision.get('critical_step') in decision.get('full_approach_target_steps',[]))
    if approach:selected='Full'
    force_event=every_event and decision['reason']=='action_event'
    if force_event:selected='Fast'
    if decision.get('safe_refresh'):selected='Full'
    if decision.get('visual_refresh'):selected='Full'
    if policy==CRITICAL_POLICY:
        mode_reason=('critical_fast' if selected=='Fast' else 'full_stable_stage' if ordinary else
                     'full_after_fast' if previous_mode=='Fast' else 'full_fast_timing')
    else:
        mode_reason=('alternating_refresh' if policy in (ALTERNATING_POLICY,ALTERNATING_NO_STATE_POLICY) else
                     'ordinary_cheaper_fast' if selected=='Fast' else 'full_after_fast' if previous_mode=='Fast' else
                     'full_key_stage' if not ordinary else 'full_cost_or_timing')
    result=dict(policy=policy,mode=selected,
        reason='visual_monitor' if decision.get('visual_refresh') else 'safe_failure_probe' if decision.get('safe_refresh') else 'critical_event_fast' if force_event else 'full_downward_approach' if approach else mode_reason,
        observation_step=step,previous_mode=previous_mode,ordinary=ordinary,
        full_per_fast=full_per_fast,full_streak=streak,
        active_phases=active,near_events=near,fast_delay_steps=fast_delay,
        profile=profile,window_seconds=window,candidates=rows,
        gripper_margin_steps=match_steps if policy in (FEEDBACK_POLICY_V2,FEEDBACK_POLICY,CRITICAL_POLICY) else None,
        protected_gripper_phases=protected,
        estimated_saving_j=saving,cost_source=costs.get('source'),
        shared_preparation='Same start and selected frequency; common preparation energy cancels',
        hold_scope='Measured fixed-input hold power; ranking prior, task energy measured separately',
        full_schedule={k:decision[k] for k in ('deadline','latest_start','service_seconds',
            'margin_seconds','compute_profile','rtc_delay_steps','start_constraint')})
    if decision.get('fast_start_budget'):
        result['selected_schedule']=result.pop('full_schedule')
        result['timing_basis']=decision['fast_start_budget']['policy']
    if every_event:result['event_fast_override']=force_event
    return result
