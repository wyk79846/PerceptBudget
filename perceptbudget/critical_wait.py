"""Pause only the uncompleted part of an event-triggered refresh."""


def bounded_result_wait(future, seconds):
    """No physics or queue operations; timeout is an experiment outcome."""
    import time
    from concurrent.futures import TimeoutError
    if seconds!=.025:raise ValueError('Experimental short wait must be25ms')
    start=time.monotonic()
    try:
        future.result(timeout=seconds)
        ready=True
    except TimeoutError:
        # A completed worker may itself have raised TimeoutError. Propagate that
        # failure instead of calling it a permissible timing shortfall.
        if future.done():future.result()
        ready=False
    stop=time.monotonic()
    return dict(start_monotonic=start,stop_monotonic=stop,seconds=stop-start,
                limit_seconds=seconds,ready=ready,budget_exceeded=not ready,
                simulation_frozen=True,queue_consumed=False)


def wait_target(request):
    if not request:
        return None
    timing = request.get('m3_timing', {})
    return timing.get('critical_step') if timing.get('reason') == 'action_event' else None


def should_wait(request, step):
    target = wait_target(request)
    return target is not None and step >= target


class PhaseWait:
    """One completed wait per finite predicted motion phase; never slide its end."""

    def __init__(self):
        self.completed = None

    def review(self, request, actions, origin, config):
        from perceptbudget.key_phases import absolute_intervals
        target = wait_target(request)
        kinds = request.get('m3_timing', {}).get('candidate_event_kinds', [])
        old = self.completed
        opposite = bool(old and (
            ('gripper_open' in kinds and 'gripper_close' in old['kinds']) or
            ('gripper_close' in kinds and 'gripper_open' in old['kinds'])))
        if old and target is not None and old['start_step'] <= target < old['stop_step'] and not opposite:
            return dict(wait=False, reason='phase_already_waited', covered_phase=dict(old))
        phases = absolute_intervals(actions, origin, config)
        phase = next((p for p in phases if p['end_observed'] and target is not None
                      and p['start_step'] <= target < p['stop_step']), None)
        return dict(wait=True, reason='new_phase' if phase else 'point_event', phase=phase,
                    kinds=list(kinds))

    def complete(self, review):
        phase = review.get('phase')
        self.completed = (dict(start_step=phase['start_step'], stop_step=phase['stop_step'],
                               kinds=review['kinds']) if phase else None)


def covered_by_phase(event, phase):
    if not phase or not phase['start_step'] <= event['target_step'] < phase['stop_step']:
        return False
    kinds = phase['kinds']
    return not (('gripper_open' == event['kind'] and 'gripper_close' in kinds) or
                ('gripper_close' == event['kind'] and 'gripper_open' in kinds))
