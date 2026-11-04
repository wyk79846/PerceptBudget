"""Pause only the uncompleted part of an event-triggered refresh."""


def wait_target(request):
    if not request:
        return None
    timing = request.get('m3_timing', {})
    return timing.get('critical_step') if timing.get('reason') == 'action_event' else None


def should_wait(request, step):
    target = wait_target(request)
    return target is not None and step >= target


def rtc_prefix_plan(request, step, estimated_delay, review):
    """Only actions executable before a known pause belong to RTC's prefix.

    A latency estimate counts control ticks while inference runs. When control
    will stop at an earlier critical step, later old actions are not committed.
    Keep the latency history for scheduling; cap only RTC's action constraint.
    """
    if type(estimated_delay) is not int or estimated_delay < 0:
        raise ValueError('RTC delay must be a nonnegative integer')
    target = wait_target(request)
    cap = max(0, target-step) if target is not None and review['wait'] else None
    return dict(policy='committed_stop_v1', estimated_delay_steps=estimated_delay,
        stop_step=target, maximum_consumed_steps=cap, wait_review=review,
        selected_delay_steps=min(estimated_delay, cap) if cap is not None else estimated_delay)


class PhaseWait:
    """One completed wait per finite predicted motion phase; never slide its end."""

    def __init__(self):
        self.completed = None

    def review(self, request, actions, origin, config):
        from perceptbudget.key_phases import intervals_for_timing
        target = wait_target(request)
        kinds = request.get('m3_timing', {}).get('candidate_event_kinds', [])
        old = self.completed
        opposite = bool(old and (
            ('gripper_open' in kinds and 'gripper_close' in old['kinds']) or
            ('gripper_close' in kinds and 'gripper_open' in old['kinds'])))
        same_kind = (not old or old.get('event_source')!='remaining_v2'
                     or bool(kinds) and set(kinds)<=set(old['kinds']))
        if old and target is not None and old['start_step'] <= target < old['stop_step'] and not opposite and same_kind:
            return dict(wait=False, reason='phase_already_waited', covered_phase=dict(old))
        phases = intervals_for_timing(actions, origin, config, request.get('m3_timing', {}))
        phase = next((p for p in phases if p['end_observed'] and target is not None
                      and p['start_step'] <= target < p['stop_step']), None)
        result=dict(wait=True, reason='new_phase' if phase else 'point_event', phase=phase,
                    kinds=list(kinds))
        if request.get('m3_timing',{}).get('phase_event_source')=='remaining_v2':
            result['event_source']='remaining_v2'
        return result

    def complete(self, review):
        phase = review.get('phase')
        self.completed = (dict(start_step=phase['start_step'], stop_step=phase['stop_step'],
                               kinds=review['kinds']) if phase else None)
        if self.completed and review.get('event_source')=='remaining_v2':
            self.completed['event_source']='remaining_v2'


def covered_by_phase(event, phase):
    if not phase or not phase['start_step'] <= event['target_step'] < phase['stop_step']:
        return False
    kinds = phase['kinds']
    if phase.get('event_source')=='remaining_v2' and event['kind'] not in kinds:
        return False
    return not (('gripper_open' == event['kind'] and 'gripper_close' in kinds) or
                ('gripper_close' == event['kind'] and 'gripper_open' in kinds))
