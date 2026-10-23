"""Experimental camera suspension for committed-action execution only."""
from contextlib import contextmanager


@contextmanager
def suspend_cameras(env):
    """Disable image sensors, not physics/proprioception; restore even on failure.

    Caller MUST force a fresh observation after exiting, before any inference.
    Never use cached images from the suspended interval as new observations.
    """
    cameras = [o for o in env.env._observables.values() if o.modality == 'image']
    if not cameras:
        raise RuntimeError('No image observables found')
    previous = [(o, o.is_enabled()) for o in cameras]
    try:
        for observable, _ in previous:
            observable.set_enabled(False)
        yield {'disabled_image_observables': len(cameras)}
    finally:
        for observable, enabled in previous:
            observable.set_enabled(enabled)
