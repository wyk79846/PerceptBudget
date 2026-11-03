"""Experimental camera suspension for committed-action execution only."""
from contextlib import contextmanager


def fresh_camera_observation(env, observation):
    """Refresh image sensors only; keep the control loop's proprioception intact."""
    import hashlib
    import time
    import numpy as np
    start = time.monotonic()
    inner = env.env
    before = inner.sim.get_state().flatten().copy()
    for observable in inner._observables.values():
        if observable.modality == 'image' and observable.is_enabled():
            observable.update(timestep=inner.model_timestep, obs_cache=inner._obs_cache, force=True)
    fresh = inner._get_observations(force_update=False)
    after = inner.sim.get_state().flatten()
    if not np.array_equal(before, after):
        raise RuntimeError('Camera acquisition advanced or changed physics')
    result = dict(observation)
    result.update((k, v) for k, v in fresh.items() if 'image' in k)
    return result, dict(start_monotonic=start, stop_monotonic=time.monotonic(),
        physics_sha256=hashlib.sha256(np.asarray(after,dtype=np.float64).tobytes()).hexdigest(),
        image_sensors_force_updated=True, proprioception_force_updated=False, physics_advanced=False)


class ContinueCapture:
    """Skip unused Continue images; request inputs always come from live sensors."""

    def __init__(self):
        self.skipped_steps = []
        self.acquisitions = []
        self.frame_steps = []

    def for_request(self, env, observation, step):
        keys = ('agentview_image', 'robot0_eye_in_hand_image')
        acquisition = None
        if not all(key in observation for key in keys):
            observation, acquisition = fresh_camera_observation(env, observation)
            if not all(key in observation for key in keys):
                raise RuntimeError('Fresh request observation is missing camera images')
            self.acquisitions.append(dict(step=step, **acquisition))
        self.frame_steps.append(step)
        return observation, acquisition

    def step(self, env, action, *, control_step, inference_active):
        if inference_active:
            return env.step(action)
        with suspend_cameras(env):
            result = env.step(action)
            if any('image' in key for key in result[0]):
                raise RuntimeError('Continue camera suspension still returned images')
        self.skipped_steps.append(control_step)
        return result

    def summary(self):
        return dict(policy='continue_only_camera_suspend_request_frames_v1',
            skipped_control_steps=self.skipped_steps,
            skipped_count=len(self.skipped_steps), acquisitions=self.acquisitions,
            frame_steps=self.frame_steps, frame_sampling='request observations only',
            physics_and_state_logging='unchanged',
            scope='Continue rendering/recording optimization; no visual disturbance monitor is enabled')


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
