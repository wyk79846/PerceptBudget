"""Sparse wrist-image feedback, independent of the VLA and simulator labels.

First establish visible texture that moves with a closed hand. Then report a
persistent disappearance of that texture. This deliberately cannot diagnose an
empty grasp that never established a visible carried object.
"""
import time


class WristSlipMonitor:
    def __init__(self, *, interval=5, confirmation=2, loss_fraction=.75):
        import cv2
        cv2.setNumThreads(1)
        self.cv = cv2
        self.interval = interval
        self.confirmation = confirmation
        self.loss_fraction = loss_fraction
        self.previous = None
        self.candidates = None
        self.anchor = None
        self.bad = 0
        self.alarmed = False
        self.samples = []
        self.last_step = -interval

    def due(self, step):
        return step > self.last_step and step > 0 and step % self.interval == 0

    def reset_grasp(self, *, reset_alarm=True):
        self.candidates = None
        self.anchor = None
        self.bad = 0
        if reset_alarm:self.alarmed = False

    def on_control(self, action):
        """Observe commanded release every tick, including between image samples."""
        if float(action[6]) <= -.2:
            self.reset_grasp()
            self.previous = None

    def texture_loss(self, gray, reference):
        scores = []
        for x, y, template in reference['templates']:
            search = gray[y-8:y+9, x-8:x+9]
            scores.append(float(self.cv.matchTemplate(search, template,
                                    self.cv.TM_CCOEFF_NORMED).max()))
        return sum(v < .65 for v in scores)/len(scores), min(scores)

    def observe(self, image, state, action, *, step, eligible=True):
        import numpy as np
        start = time.monotonic()
        cv = self.cv
        self.on_control(action)
        gray = (cv.cvtColor(image, cv.COLOR_RGB2GRAY) if image.ndim == 3 else image)
        gray = cv.resize(gray, (256, 256), interpolation=cv.INTER_AREA)
        state = np.asarray(state, dtype=float)
        closed = float(action[6]) >= .2
        width = abs(float(state[6]-state[7]))
        row = dict(step=int(step), eligible=bool(eligible), closed=closed,
                   width=width, locked=False, alarm=False, reason='warming_up')
        if not closed:
            self.reset_grasp()
            row['reason'] = 'gripper_open'
        elif self.previous is not None:
            old = self.previous
            aperture_change = abs(width-old['width'])
            travel = float(np.linalg.norm(state[:3]-old['state'][:3]))
            row.update(hand_travel_m=travel, aperture_change_m=aperture_change)
            reference = self.anchor or (self.candidates or {}).get('reference')
            collapse = old['width'] > .01 and width < .5*old['width']
            if (collapse and reference is not None and eligible and not self.alarmed
                    and state[2] >= old['state'][2]-.003):
                loss, correlation = self.texture_loss(gray, reference)
                if loss >= self.loss_fraction:
                    row.update(alarm=True, reason='closed_jaw_collapse_and_texture_loss',
                               loss_fraction=loss, correlation_min=correlation,
                               reference_step=reference['step'])
                    self.alarmed = True
            if aperture_change > .003 and not row['alarm']:
                self.reset_grasp(reset_alarm=False)
                row['reason'] = 'fingers_still_moving'
            elif self.anchor is not None:
                loss, correlation = self.texture_loss(gray, self.anchor)
                self.bad = self.bad+1 if loss >= self.loss_fraction else 0
                row.update(locked=True, texture_count=len(self.anchor['templates']), loss_fraction=loss,
                           correlation_min=correlation, consecutive_loss=self.bad,
                           locked_step=self.anchor['step'])
                if not row['alarm']:row['reason'] = 'tracking_carried_texture'
                if eligible and self.bad >= self.confirmation and not self.alarmed:
                    row.update(alarm=True, reason='persistent_carried_texture_loss')
                    self.alarmed = True
            elif not row['alarm']:
                # Raw LIBERO wrist images have the hand at the top edge.
                # Exclude the palm/fingers; inspect the space beyond their tips.
                mask = np.zeros_like(gray)
                mask[64:112, 88:168] = 255
                points = cv.goodFeaturesToTrack(old['gray'], maxCorners=24,
                    qualityLevel=.04, minDistance=6, mask=mask, blockSize=5)
                stable = []
                if points is not None:
                    now, good, _ = cv.calcOpticalFlowPyrLK(old['gray'], gray, points, None,
                        winSize=(15, 15), maxLevel=2)
                    if now is not None:
                        for p, q, ok in zip(points[:, 0], now[:, 0], good[:, 0]):
                            if ok and np.linalg.norm(p-q) <= 1.25:
                                x, y = np.rint(q).astype(int)
                                if 88 <= x < 168 and 64 <= y < 112:
                                    stable.append((int(x), int(y)))
                # A stationary camera alone is not evidence of carrying.
                # Require hand movement plus background optical flow.
                bg_mask = np.full_like(gray, 255)
                bg_mask[:120, 56:200] = 0
                background = cv.goodFeaturesToTrack(old['gray'], maxCorners=48,
                    qualityLevel=.03, minDistance=10, mask=bg_mask)
                flow = 0.
                if background is not None:
                    moved, ok, _ = cv.calcOpticalFlowPyrLK(old['gray'], gray, background,
                        None, winSize=(21, 21), maxLevel=2)
                    if moved is not None and np.any(ok):
                        flow = float(np.median(np.linalg.norm(
                            moved[:, 0]-background[:, 0], axis=1)[ok[:, 0].astype(bool)]))
                row.update(stable_texture_count=len(stable), background_flow_px=flow,
                           reason='seeking_carried_texture')
                if travel >= .008 and flow >= 2. and len(stable) >= 4:
                    # Require a second observation of the same stable region.
                    supported = (self.candidates is not None and
                        sum(any((x-a)**2+(y-b)**2 <= 16 for a,b in self.candidates['points'])
                            for x,y in stable) >= 4)
                    templates = [(x,y,gray[y-5:y+6,x-5:x+6].copy())
                                 for x,y in stable]
                    if supported:
                        self.anchor = dict(step=step, templates=templates)
                        row.update(locked=True, reason='carried_texture_established',
                                   texture_count=len(templates))
                    self.candidates = dict(points=stable,
                        reference=dict(step=step, templates=templates))
                else:
                    self.candidates = None
        self.previous = dict(gray=gray.copy(), state=state.copy(), width=width)
        self.last_step = step
        row['detector_seconds'] = time.monotonic()-start
        self.samples.append(row)
        return row


def continuation_eligible(decision, *, inflight):
    """No future labels or final task outcome in the online sampling gate."""
    return bool(not inflight and decision and not decision.get('start_now', True)
                and decision.get('wait_seconds', 0.) > 0)


def fresh_wrist_image(env):
    """Acquire one image without advancing physics or replacing policy inputs."""
    start = time.monotonic()
    inner = env.env
    sim_time = float(inner.sim.data.time)
    sensor = inner._observables['robot0_eye_in_hand_image']
    if not sensor.is_enabled():
        raise RuntimeError('Wrist camera must be restored before monitor acquisition')
    sensor.update(timestep=inner.model_timestep, obs_cache=inner._obs_cache, force=True)
    observed = inner._get_observations(force_update=False)
    if float(inner.sim.data.time) != sim_time:
        raise RuntimeError('Visual monitoring advanced simulation')
    return observed['robot0_eye_in_hand_image'].copy(), dict(
        start_monotonic=start, stop_monotonic=time.monotonic(),
        sim_time=sim_time, camera='robot0_eye_in_hand', physics_advanced=False,
        policy_observation_replaced=False)


def cap_visual_refresh(decision, pending, *, now, step, current_profile):
    """Advance a Full request while retaining earlier mandatory deadlines."""
    d = dict(decision)
    ticks = int(current_profile != 'baseline')
    latest = now+(pending['request_step']-step-ticks)*d['estimated_control_period_seconds']
    cap = dict(scope='visual_monitor', request_step=pending['request_step'],
               recovery_ticks=ticks, latest_start=latest, alarm=pending)
    if latest < d.get('first_refresh_cap', {}).get('latest_start', float('inf')):
        d['first_refresh_cap'] = cap
    if latest < d['latest_start']:
        d.update(latest_start=latest, start_now=latest <= now,
                 wait_seconds=max(0.,latest-now), start_constraint='first_refresh')
    d['visual_refresh'] = dict(pending)
    return d
