"""Opt-in common LOW endpoint, separate from task completion and inference drain."""
import time


def settle_low(clocks, *, seconds=3., timeout=3.):
    if seconds < 3.:
        raise ValueError('Common LOW tail requires at least three seconds')
    start = time.monotonic()
    clocks.poll()
    until = start + timeout
    while clocks.pending is not None:
        if time.monotonic() >= until:
            raise TimeoutError('Pending terminal frequency change timed out')
        time.sleep(.002)
        clocks.poll()
    if clocks.profile != 'gpu_emc':
        clocks.switch('gpu_emc')
        while clocks.pending is not None:
            if time.monotonic() >= until:
                raise TimeoutError('Terminal LOW acknowledgement timed out')
            time.sleep(.002)
            clocks.poll()
    acknowledged = time.monotonic()
    end = acknowledged + seconds
    while time.monotonic() < end:
        time.sleep(min(.02, max(0., end-time.monotonic())))
        clocks.poll()
        if clocks.pending is not None or clocks.profile != 'gpu_emc':
            raise RuntimeError('Frequency changed during common LOW tail')
    return dict(start_monotonic=start, low_acknowledged_monotonic=acknowledged,
                stop_monotonic=time.monotonic(), low_hold_seconds=seconds,
                profile='gpu_emc', scope='Post-inference return to LOW plus fixed hold; not Continue')
