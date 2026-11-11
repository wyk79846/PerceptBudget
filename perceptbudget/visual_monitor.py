"""Sparse wrist-image feedback, independent of the VLA and simulator labels.

Track carried texture and optionally check a nearly empty closed hand lifting
while the visible texture near its tips fails to move with it.
"""
import time


def detector_name(value):
    """Semantic method names; accept recorded commands from older experiments."""
    aliases={'original_v1':'fixed_reference','updated_v2':'updated_reference'}
    value=aliases.get(value,value)
    if value not in ('fixed_reference','updated_reference'):
        raise ValueError('Unknown wrist monitor detector')
    return value


class WristSlipMonitor:
    def __init__(self, *, interval=5, confirmation=2, loss_fraction=.75, detector='fixed_reference',grasp_check=False):
        import cv2
        cv2.setNumThreads(1)
        self.cv = cv2
        self.detector=detector_name(detector)
        if grasp_check and self.detector!='updated_reference':
            raise ValueError('Grasp check requires updated reference monitoring')
        self.grasp_check=grasp_check
        self.open_width=0.
        self.close=None
        self.control_history=[]
        self.last_control=None
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

    def on_control(self, action, state=None, *, step=None):
        """Observe commanded release every tick, including between image samples."""
        g=float(action[6])
        if g<=-.2:self.close=None
        if self.grasp_check and state is not None:
            width=abs(float(state[6]-state[7]))
            if g<=-.2:
                self.open_width=max(self.open_width,width)
                self.close=None
            elif g>=.2 and self.close is None:
                self.close=dict(step=step,z=self.last_control['z'] if self.last_control else float(state[2]),checks=0)
            current=dict(step=step,z=float(state[2]),width=width,g=g)
            if self.last_control is None or step!=self.last_control['step']:
                self.control_history.append(current)
                self.control_history=self.control_history[-5:]
            self.last_control=current
        if float(action[6]) <= -.2:
            self.reset_grasp()
            self.previous = None

    def empty_lift_candidate(self):
        if not self.grasp_check or self.close is None or self.open_width<=0 or len(self.control_history)<2:
            return None
        recent=self.control_history[-2:];current=recent[-1]
        lift=current['z']-self.close['z']
        if (all(r['g']>=.2 for r in recent) and current['width']<=.05*self.open_width
                and lift>=.02 and current['z']>=recent[0]['z']-.003):
            return dict(close_step=self.close['step'],open_width_m=self.open_width,
                width_m=current['width'],width_fraction=current['width']/self.open_width,
                eef_lift_m=lift,control_step=current['step'])
        return None

    def grasp_check_due(self):
        return (not self.alarmed and self.empty_lift_candidate() is not None
                and self.close['checks']<2)

    def note_grasp_sample(self):
        if self.empty_lift_candidate() is not None:self.close['checks']+=1

    def grasp_motion(self, gray, state, candidate, step):
        """Image motion supplies positive evidence; missing features are unknown."""
        import numpy as np
        result=dict(state_candidate=candidate,available=False,confirmed=False)
        old=self.previous
        if old is None or step-self.last_step>self.interval:return result
        rise=float(state[2]-old['state'][2])
        result['image_pair_rise_m']=rise
        if rise<.008:return result
        cv=self.cv;mask=np.zeros_like(gray);mask[64:112,88:168]=255
        points=cv.goodFeaturesToTrack(old['gray'],maxCorners=24,qualityLevel=.04,
            minDistance=6,mask=mask,blockSize=5)
        if points is None:return result
        now,valid,_=cv.calcOpticalFlowPyrLK(old['gray'],gray,points,None,winSize=(21,21),maxLevel=3)
        if now is None:return result
        back,back_valid,_=cv.calcOpticalFlowPyrLK(gray,old['gray'],now,None,winSize=(21,21),maxLevel=3)
        if back is None:return result
        good=(valid[:,0].astype(bool)&back_valid[:,0].astype(bool)
              &(np.linalg.norm(back[:,0]-points[:,0],axis=1)<=1.25))
        flow=np.linalg.norm(now[:,0]-points[:,0],axis=1)[good]
        result['tracked_points']=len(flow)
        if len(flow)<4:return result
        moving=float(np.mean(flow>=2.));static=int(sum(flow<=1.25))
        result.update(available=True,moving_fraction=moving,static_points=static,
            median_motion_px=float(np.median(flow)),
            confirmed=moving>=self.loss_fraction and static<4)
        return result

    def texture_loss(self, gray, reference):
        scores = []
        for x, y, template in reference['templates']:
            search = gray[y-8:y+9, x-8:x+9]
            scores.append(float(self.cv.matchTemplate(search, template,
                                    self.cv.TM_CCOEFF_NORMED).max()))
        return sum(v < .65 for v in scores)/len(scores), min(scores)

    def update_visible_reference(self, gray, reference, step):
        """Update only strongly matched patches after a coherent observation.

        The same1.25px motion test used to establish carrying is retained.
        Missing patches never become new reference patches, and a mostly lost
        object cannot be learned as background.
        """
        cv=self.cv;visible=[]
        for x,y,template in reference['templates']:
            search=gray[y-8:y+9,x-8:x+9]
            _,score,_,location=cv.minMaxLoc(cv.matchTemplate(search,template,cv.TM_CCOEFF_NORMED))
            dx,dy=location[0]-3,location[1]-3
            visible.append(score>=.65 and dx*dx+dy*dy<=1.25**2)
        coherent=sum(visible)/len(visible)>=self.loss_fraction
        if coherent:
            reference['templates']=[(x,y,gray[y-5:y+6,x-5:x+6].copy() if ok else template)
                for (x,y,template),ok in zip(reference['templates'],visible)]
            reference['updated_step']=step
        return coherent

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
        if self.detector=='updated_reference':
            gap=step-self.last_step if self.last_step>=0 else None
            row['sample_gap_steps']=gap
            if self.previous is not None and gap is not None and gap>self.interval:
                # An unobserved interval does not demonstrate persistent loss.
                # Re-establish the reference instead of carrying stale evidence.
                self.reset_grasp(reset_alarm=False)
                self.previous=None
                row.update(reference_reacquired=True,reason='observation_gap_reacquire')
        candidate=self.empty_lift_candidate()
        if candidate is not None:
            evidence=self.grasp_motion(gray,state,candidate,step)
            row['grasp_check']=evidence
            self.note_grasp_sample()
            if evidence['confirmed'] and eligible and not self.alarmed:
                row.update(alarm=True,reason='empty_closed_grip_visual_motion')
                self.alarmed=True
        if not closed:
            self.reset_grasp()
            row['reason'] = 'gripper_open'
        elif self.previous is not None and not row['alarm']:
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
                if self.detector=='updated_reference' and loss<self.loss_fraction:
                    row['reference_updated']=self.update_visible_reference(gray,self.anchor,step)
                    row['reference_step']=self.anchor.get('updated_step',self.anchor['step'])
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


def stop_response(config, request=None):
    settings=config.get('visual_monitor',{})
    enabled=settings.get('mode')=='refresh' and settings.get('response')=='stop_and_replan'
    return enabled and (request is None or bool(request.get('visual_monitor_response')))
