"""Pause only the uncompleted part of an event-triggered refresh."""


def wait_target(request):
    if not request:
        return None
    timing = request.get('m3_timing', {})
    return timing.get('critical_step') if timing.get('reason') == 'action_event' else None


def should_wait(request, step):
    target = wait_target(request)
    return target is not None and step >= target
