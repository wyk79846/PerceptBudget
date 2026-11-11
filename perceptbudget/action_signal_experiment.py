"""Optional, budgeted action-cue experiment. CPU arithmetic, no new perception.

All features use the currently available chunk, the preceding prediction on
the SAME absolute action steps, and already executed controls. No task labels,
success labels, object poses, contacts or future observations enter the rule.
"""
import json
import math
from pathlib import Path

VERSION = 'budgeted_action_signals_v1'
ALIGNED_VERSION = 'aligned_action_signals_v2'
PAIRED_VERSION = 'paired_action_signals_v3'
METRICS = ('translation_change', 'rotation_change',
           'translation_disagreement', 'rotation_disagreement')


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def mean(values):
    return [sum(v[k] for v in values)/len(values) for k in range(7)]


def difference(a, b, start, end):
    return norm([a[k]-b[k] for k in range(start, end)])


def grip(action, threshold):
    return 1 if action[6] >= threshold else -1 if action[6] <= -threshold else 0


def validate(settings):
    if settings.get('version') not in (VERSION, ALIGNED_VERSION, PAIRED_VERSION):
        raise ValueError('Unknown action-signal experiment version')
    thresholds = settings.get('thresholds', {})
    if set(thresholds) != set(METRICS) or any(
            not math.isfinite(v) or v <= 0 for v in thresholds.values()):
        raise ValueError('Action signals require four positive pooled thresholds')
    if settings.get('minimum_request_steps') != 20 or settings.get('minimum_mean_steps') != 22:
        raise ValueError('This experiment keeps the 20-step / 22-step refresh budget')
    if type(settings.get('refresh_budget_enabled', True)) is not bool:
        raise ValueError('refresh_budget_enabled must be a boolean')
    if type(settings.get('skip_covered_candidates', False)) is not bool:
        raise ValueError('skip_covered_candidates must be a boolean')
    if type(settings.get('pair_components', True)) is not bool:
        raise ValueError('pair_components must be a boolean')
    if type(settings.get('gripper_conflict', False)) is not bool:
        raise ValueError('gripper_conflict must be a boolean')
    if type(settings.get('post_close_feedback', False)) is not bool:
        raise ValueError('post_close_feedback must be a boolean')
    if type(settings.get('gripper_lift_feedback',False)) is not bool:
        raise ValueError('gripper_lift_feedback must be a boolean')
    if settings.get('gripper_lift_feedback') and settings.get('post_close_feedback'):
        raise ValueError('Use gated gripper feedback or blanket post-close feedback, not both')
    age = settings.get('critical_freshness_max_age_steps')
    if age is not None and (settings.get('version') != PAIRED_VERSION
            or type(age) is not int or not 1 <= age < 50
            or settings.get('post_close_feedback', False)):
        raise ValueError('Critical freshness requires standalone v3 and an age in1..49')
    wait = settings.get('critical_freshness_max_wait_steps', 0)
    if type(wait) is not int or not 0 <= wait <= 2 or (wait and age is None):
        raise ValueError('Critical freshness wait requires an age and0..2 estimated steps')
    correction = settings.get('critical_freshness_correction_steps', 0)
    if type(correction) is not int or not 0 <= correction <= 2 or (correction and age is None):
        raise ValueError('Critical freshness correction requires an age and0..2 steps')
    return settings


def load_settings(path):
    return validate(json.loads(Path(path).read_text()))


def extract(actions, *, origin, source_step, history, reference, config,
            within_plan_only=False):
    """Two-step sustained changes; unmatched old tails stay missing, not zero.

    Rotations are decoded controller commands, NOT world-frame angular speed.
    The old and new actions are compared before either is executed, excluding
    the consumed prefix. Differences indicate changed plans, not proven error.
    """
    n = config.confirm_steps
    # V2 measures changes inside the unexecuted part of ONE prediction, both
    # offline and online. Its first two windows have no preceding planned
    # controls. The original event detector still handles handoff events.
    history = {} if within_plan_only else {
        int(k): v for k, v in history.items() if k < source_step}

    def action_at(step):
        if step < source_step:
            return history.get(step)
        i = step-origin
        return actions[i] if 0 <= i < len(actions) else None

    rows = []
    last_grip = next((grip(history[k], config.gripper_threshold)
                     for k in sorted(history, reverse=True)
                     if grip(history[k], config.gripper_threshold)), 0)
    last_switch = None
    for step in range(source_step, origin+len(actions)-n+1):
        future = [action_at(s) for s in range(step, step+n)]
        before = [action_at(s) for s in range(step-n, step)]
        if any(a is None for a in future):
            continue
        a = mean(future)
        b = mean(before) if all(v is not None for v in before) else None
        command = grip(future[0], config.gripper_threshold)
        stable_grip = bool(command and all(grip(v, config.gripper_threshold) == command for v in future))
        switch = bool(stable_grip and last_grip and command != last_grip)
        reversal_steps = step-last_switch if switch and last_switch is not None else None
        if stable_grip:
            if switch:
                last_switch = step
            last_grip = command
        ref_offset = step-reference['origin'] if reference else -1
        overlap = bool(reference and ref_offset >= 0 and ref_offset+n <= len(reference['actions']))
        old = reference['actions'][ref_offset:ref_offset+n] if overlap else None
        # Minimum over the confirmation window prevents one noisy action from
        # creating a "sustained" plan disagreement.
        row = dict(target_step=step, translation_norm=norm(a[:3]),
            rotation_norm=norm(a[3:6]), vertical_command=a[2],
            translation_change=difference(a, b, 0, 3) if b else None,
            rotation_change=difference(a, b, 3, 6) if b else None,
            slowdown_ratio=norm(a[:3])/max(norm(b[:3]), 1e-9) if b else None,
            gripper_command=command, gripper_switch=switch,
            gripper_reversal_steps=reversal_steps, reference_overlap=overlap,
            translation_disagreement=min(difference(x, y, 0, 3) for x, y in zip(future, old)) if overlap else None,
            rotation_disagreement=min(difference(x, y, 3, 6) for x, y in zip(future, old)) if overlap else None,
            gripper_disagreement=bool(overlap and stable_grip and all(
                grip(v, config.gripper_threshold) == -command for v in old)),
            vertical_reversal=bool(b and a[2]*b[2] < 0
                and abs(a[2]-b[2]) >= config.minimum_motion))
        rows.append(row)
    return rows


def observe_execution(rows):
    """Record available robot feedback; do not invent a grasp-success label."""
    if not rows:
        return dict(available=False)
    first, last = rows[0], rows[-1]
    before = first.get('before', {}).get('raw_state')
    after = last.get('after', {}).get('raw_state')
    result = dict(available=bool(before and after), first_step=first.get('step'),
                  last_step=last.get('step'), controls=len(rows))
    if before and after and len(before) >= 8 and len(after) >= 8:
        result.update(translation_distance=norm([after[k]-before[k] for k in range(3)]),
            orientation_delta=[after[k]-before[k] for k in range(3, 6)],
            gripper_before=list(before[6:8]), gripper_after=list(after[6:8]),
            use='record_only; state differences do not identify grasp/contact')
    return result


def overlap_summary(features):
    """Risk from the available overlap; never extend the old predicted tail.

    The old chunk often ends BEFORE a budget-compatible new event. Its current
    disagreement is therefore a chunk-level risk signal, not a fictitious
    action-by-action comparison at that later event.
    """
    overlap = [f for f in features if f['reference_overlap']]
    return dict(translation_disagreement=max((f['translation_disagreement'] for f in overlap), default=None),
        rotation_disagreement=max((f['rotation_disagreement'] for f in overlap), default=None),
        gripper_disagreement=any(f['gripper_disagreement'] for f in overlap),
        first_step=overlap[0]['target_step'] if overlap else None,
        last_step=overlap[-1]['target_step'] if overlap else None)


def disagreement_kinds(evidence, thresholds):
    kinds = [k for k in METRICS[2:]
             if evidence[k] is not None and evidence[k] > thresholds[k]]
    if evidence['gripper_disagreement']:
        kinds.append('gripper_disagreement')
    return kinds


def prepare(*, actions, origin, source_step, history, reference, config,
            settings, period, response_seconds, margin_seconds, prediction_count,
            covered_phases=(), covered_events=()):
    """Freeze candidate and budget once at commit, just like the old forecast."""
    phase_aware = settings['version'] == PAIRED_VERSION
    paired = phase_aware and settings.get('pair_components', True)
    aligned = settings['version'] in (ALIGNED_VERSION, PAIRED_VERSION)
    features = extract(actions, origin=origin, source_step=source_step,
                       history=history, reference=reference, config=config,
                       within_plan_only=aligned)
    minimum = settings['minimum_request_steps']
    budget_enabled = settings.get('refresh_budget_enabled', True)
    # prediction_count includes bootstrap: one upcoming interval per committed
    # chunk. Existing urgent events can break this mean; new cues cannot.
    earliest = max(origin+minimum, settings['minimum_mean_steps']*prediction_count) if budget_enabled else source_step
    # Include a clock-ACK tick and the existing early-start rounding allowance.
    lead = math.ceil((response_seconds+margin_seconds)/period-1e-9)+2
    earliest_target = earliest+lead if budget_enabled else source_step
    thresholds = settings['thresholds']
    overlap = overlap_summary(features)
    chunk_disagreement = disagreement_kinds(overlap, thresholds)
    opportunities = []
    for f in features:
        # V1 used disagreement anywhere in the overlap. V2 requires evidence
        # at this exact future window; a missing old tail supplies no evidence.
        disagreement = disagreement_kinds(f, thresholds) if aligned else chunk_disagreement
        geometry = [k for k in METRICS[:2] if f[k] is not None and f[k] > thresholds[k]]
        if f['gripper_switch']:
            geometry.append('gripper_switch')
        if f['gripper_reversal_steps'] is not None and f['gripper_reversal_steps'] <= minimum:
            geometry.append('gripper_reversal')
        if paired:
            pairs = dict(translation_change='translation_disagreement',
                         rotation_change='rotation_disagreement',
                         gripper_switch='gripper_disagreement',
                         gripper_reversal='gripper_disagreement')
            geometry = [k for k in geometry if pairs[k] in disagreement]
            disagreement = [k for k in disagreement if k in {pairs[g] for g in geometry}]
        # A persistent, opposite open/close plan is a discrete disagreement
        # even when neither command changes inside this new chunk. This is an
        # opt-in uncertainty cue, not evidence that an object was grasped.
        if aligned and settings.get('gripper_conflict', False) and f['gripper_disagreement']:
            geometry.append('gripper_plan_conflict')
            if 'gripper_disagreement' not in disagreement:
                disagreement.append('gripper_disagreement')
        if not geometry or not disagreement:
            continue
        opportunities.append(dict(target_step=f['target_step'],
            geometry=geometry, disagreement=disagreement,
            admitted=f['target_step'] >= earliest_target,
            reason='two_signal_families' if f['target_step'] >= earliest_target else 'refresh_budget'))
        if aligned:
            opportunities[-1]['evidence_steps'] = [f['target_step'],
                f['target_step']+config.confirm_steps-1]
        if phase_aware and opportunities[-1]['admitted']:
            from perceptbudget.critical_wait import covered_by_phase
            cue = dict(kind='action_revision', target_step=f['target_step'])
            if any(covered_by_phase(cue, phase) for phase in covered_phases):
                opportunities[-1].update(admitted=False, reason='phase_already_refreshed')
        if settings.get('skip_covered_candidates', False) and opportunities[-1]['admitted']:
            if any(event['kind'] == 'action_revision'
                    and abs(event['target_step'] - f['target_step']) <= config.match_steps
                    for event in covered_events):
                opportunities[-1].update(admitted=False, reason='point_already_refreshed')
    selected = next((f for f in opportunities if f['admitted']), None)
    event = (dict(kind='action_revision', target_step=selected['target_step'],
                  evidence=dict(geometry=selected['geometry'], disagreement=selected['disagreement']))
             if selected else None)
    if event and aligned:
        event['evidence']['steps'] = selected['evidence_steps']
    review = dict(version=settings['version'], observation_step=origin, source_step=source_step,
        reference_origin=reference['origin'] if reference else None,
        feature_steps=len(features), minimum_request_steps=minimum,
        minimum_mean_steps=settings['minimum_mean_steps'], prediction_count=prediction_count,
        earliest_new_request_step=earliest, earliest_new_target_step=earliest_target,
        service_reserve_steps=lead, period_seconds=period, response_seconds=response_seconds,
        margin_seconds=margin_seconds, opportunity_count=len(opportunities),
        overlap_evidence=overlap,
        budget_rejected=sum(f['reason']=='refresh_budget' for f in opportunities), candidate=event)
    if aligned:
        review.update(change_scope='unexecuted_current_plan_only',
                      disagreement_scope='same_target_window',
                      overlap_summary_use='diagnostic_only')
    if phase_aware:
        review.update(evidence_pairing='same_command_component' if paired else 'same_target_window',
                      covered_phases=[dict(p) for p in covered_phases],
                      phase_rejected=sum(f['reason']=='phase_already_refreshed' for f in opportunities))
    if settings.get('gripper_conflict', False):
        review['gripper_conflict'] = True
    if not budget_enabled:
        # Keep old records/replays identical when the opt-out is absent.
        # Imminent cues now reach the existing scheduler, which may wait for
        # inference at the critical action instead of dropping the cue here.
        review['refresh_budget_enabled'] = False
    if settings.get('skip_covered_candidates', False):
        review.update(skip_covered_candidates=True,
            covered_events=[dict(e) for e in covered_events],
            point_rejected=sum(f['reason']=='point_already_refreshed' for f in opportunities))
    return dict(review=review, features=features, opportunities=opportunities)


def apply_candidate(candidates, *, packet, step, end, prediction_id, covered, tolerance, period=None):
    """Never undo old filtering/coverage. Add at most one new, typed cue."""
    cue = packet['review']['candidate']
    if not cue or not step <= cue['target_step'] < end:
        return candidates
    review = packet['review']
    # Hard gate at the actual decision step. A faster/slower control tick must
    # not turn the frozen forecast into an early request outside the budget.
    if review.get('refresh_budget_enabled', True) and step < review['earliest_new_request_step']:
        return candidates
    if any(c['kind'] == cue['kind'] and abs(c['target_step']-cue['target_step']) <= tolerance for c in covered):
        return candidates
    # An earlier existing event already obtains feedback; no duplicate cue.
    if any(e['target_step'] <= cue['target_step'] for e in candidates):
        return candidates
    if 'activation_step' not in packet:
        lead = math.ceil((review['response_seconds']+review['margin_seconds']) /
                         (period or review['period_seconds'])-1e-9)+2
        if review.get('refresh_budget_enabled', True) and cue['target_step']-step < lead:
            return candidates
        packet['activation_step'] = step
    # Once admitted, retain the same deadline while the Full/Fast selector
    # waits for its own launch point. Do not retract it on the next tick.
    return [*candidates, dict(cue, prediction_id=prediction_id)]
