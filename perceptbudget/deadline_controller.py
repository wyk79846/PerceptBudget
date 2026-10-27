"""Small time calculations for M3; clocks and inference are owned by the runner."""
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path


def percentile(values, probability=.95):
    values = sorted(float(v) for v in values)
    if not values or not 0 < probability <= 1 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Use positive measured service times and probability in(0,1]')
    return values[math.ceil(len(values)*probability)-1]


class FullServiceTimes:
    def __init__(self, samples, *, probability=.95, capacity=128):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be positive')
        self.samples = {h: deque(samples[h], maxlen=capacity) for h in ('hot','short','long')}
        self.recovery_samples = deque(maxlen=capacity)
        self.probability = probability
        for values in self.samples.values():
            percentile(values, probability)

    @classmethod
    def from_m2_summary(cls, path, **kwargs):
        summary = json.loads(Path(path).read_text())
        samples = {h: [] for h in ('hot','short','long')}
        for source in summary['selected_summary_sources']:
            full = json.loads(Path(source['full']).read_text())
            if full['status'] != 'completed':
                raise ValueError('M2 source is incomplete')
            for row in full['requests']:
                if row['profile'] == 'baseline':
                    samples[row['state_history']].append(row['service_ms']/1000)
        return cls(samples, **kwargs)

    def estimate(self, history):
        if history == 'recovery':
            # Unknown dwell between calibrated points: use the larger estimate.
            estimates = [self.estimate('hot'), self.estimate('short'), self.estimate('long')]
            if self.recovery_samples:
                estimates.append(percentile(self.recovery_samples,self.probability))
            return max(estimates)
        return percentile(self.samples[history], self.probability)

    def observe(self, history, seconds):
        percentile([seconds], self.probability)
        (self.recovery_samples if history=='recovery' else self.samples[history]).append(float(seconds))


@dataclass(frozen=True)
class TimingPlan:
    deadline: float
    reason: str
    latest_start: float
    wait_seconds: float
    start_now: bool
    service_seconds: float
    margin_seconds: float
    deadline_already_unreachable: bool


def timing_plan(*, now, step, buffer_end_step, period, service_seconds,
                margin_seconds=.05, critical_step=None, observation_deadline=None, dvac_step=None,
                signal_policy='earliest'):
    """buffer_end_step is the caller's admitted exclusive end, not raw H.

    All deadlines use monotonic wall time. service_seconds already includes
    device recovery, so no extra restore term belongs in this calculation.
    """
    if any(not math.isfinite(v) for v in (now, period, service_seconds, margin_seconds)):
        raise ValueError('Timing inputs must be finite')
    if period <= 0 or service_seconds <= 0 or margin_seconds < 0:
        raise ValueError('Invalid period, service time or margin')
    if any(type(v) is not int for v in (step, buffer_end_step)):
        raise ValueError('Control indices must be integers')
    if signal_policy not in ('earliest','latest'):
        raise ValueError('Signal policy must be earliest or latest')
    limits = [(now+(buffer_end_step-step)*period, 'buffer')]
    signals=[]
    if dvac_step is not None:
        if type(dvac_step) is not int or dvac_step < 0:
            raise ValueError('DVAC boundary must be a nonnegative absolute action index')
        signals.append((now+(dvac_step-step)*period,'denoising_variance'))
    if critical_step is not None:
        if type(critical_step) is not int or critical_step < 0 or (signal_policy=='earliest' and critical_step<step):
            raise ValueError('Critical step must be nonnegative; earliest policy requires a current or future index')
        signals.append((now+(critical_step-step)*period, 'action_event'))
    if signals:
        selected=(min if signal_policy=='earliest' else max)(t for t,_ in signals)
        # Preserve the original tie break, including event-coverage bookkeeping.
        limits.append(min(s for s in signals if s[0]==selected))
    if observation_deadline is not None:
        if not math.isfinite(observation_deadline):
            raise ValueError('Observation deadline must be finite')
        limits.append((observation_deadline, 'observation_age'))
    deadline, reason = min(limits)
    latest = deadline-service_seconds-margin_seconds
    return TimingPlan(deadline, reason, latest, max(0.,latest-now), latest<=now,
                      service_seconds, margin_seconds, now+service_seconds>deadline)


def full_continue_plan(*, hot_service, recovery_service, minimum_low_idle=.2, **timing):
    """First compare timing feasibility; low-idle payback uses an initial gate.

    Return both paths for auditing. Stage-three execution supplies measured
    energy checks; this helper does not relabel the gate as net energy proof.
    """
    hot = timing_plan(service_seconds=hot_service, **timing)
    low = timing_plan(service_seconds=recovery_service, **timing)
    can_lower = not low.start_now and low.wait_seconds >= minimum_low_idle
    return dict(path='low_continue' if can_lower else 'hot_continue',
                selected=low if can_lower else hot, hot=hot, recovery=low,
                low_idle_candidate=can_lower)


class SafetyMargin:
    """Increase after uncovered residuals; reduce gradually over stable windows."""
    def __init__(self, initial=.05, floor=.02, step=.01, stable_window=20):
        if not 0 <= floor <= initial or step <= 0 or stable_window < 1:
            raise ValueError('Invalid margin parameters')
        self.seconds, self.floor, self.step = initial, floor, step
        self.window, self.stable = stable_window, 0
        self.excess = deque(maxlen=max(32,stable_window))

    def observe(self, *, predicted, actual, start_lateness=0.):
        if any(not math.isfinite(v) for v in (predicted,actual,start_lateness)) or min(predicted,actual) <= 0 or start_lateness < 0:
            raise ValueError('Use finite measured times and nonnegative start lateness')
        residual = max(0.,actual-predicted)+start_lateness
        self.excess.append(residual)
        required = max(self.excess) if len(self.excess)<20 else sorted(self.excess)[math.ceil(.95*len(self.excess))-1]
        before = self.seconds
        if residual > self.seconds:
            self.seconds = max(self.seconds+self.step, residual+self.step)
            self.stable = 0
        else:
            self.stable += 1
            if self.stable >= self.window:
                self.seconds = max(self.floor, required+self.step, self.seconds-self.step)
                self.stable = 0
        return dict(before_seconds=before, after_seconds=self.seconds,
                    excess_seconds=residual, samples=len(self.excess))
