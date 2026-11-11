"""Optional pre-event observation renewal for an already covered action cue."""
from perceptbudget.critical_wait import covered_by_phase


def propose(*, events, candidates, phases, covered, used, ordinary_turns,
            observation_step, step, end, prediction_id, maximum_age, tolerance):
    """Propose one stale covered cue; the chosen service must still fit later.

    The age belongs to the image that generated the current actions. This is
    neither an object/grasp detector nor permission to re-use expired actions.
    """
    for event in sorted(events, key=lambda e: e['target_step']):
        target, kind = event['target_step'], event['kind']
        if not step < target < end or target-observation_step <= maximum_age:
            continue
        if kind == 'turn' and target in ordinary_turns:
            continue
        if any(e['target_step'] <= target for e in candidates):
            continue
        phase = next((p for p in phases if covered_by_phase(event, p)), None)
        point = any(e['kind'] == kind and abs(e['target_step']-target) <= tolerance
                    for e in covered)
        if phase is None and not point:
            continue
        if any(p['kind'] == kind and p['start_step'] <= target < p['stop_step']
               for p in used):
            continue
        return dict(policy='critical_observation_freshness_v1', kind=kind,
                    target_step=target, prediction_id=prediction_id,
                    source_observation_step=observation_step,
                    source_age_steps=target-observation_step,
                    maximum_age_steps=maximum_age,
                    start_step=phase['start_step'] if phase else target-tolerance,
                    stop_step=phase['stop_step'] if phase else target+tolerance+1,
                    was_phase_covered=phase is not None, was_point_covered=point)
    return None
