"""Common-window energy choice; callers supply measured, aligned segments."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Segment:
    kind: str
    seconds: float
    energy_j: float

    def __post_init__(self):
        if self.kind not in ('hold','service','transition'):
            raise ValueError('Unknown energy segment')
        if any(not math.isfinite(v) or v<0 for v in (self.seconds,self.energy_j)):
            raise ValueError('Invalid segment measurement')


@dataclass(frozen=True)
class Candidate:
    label: str
    mode: str
    compute_profile: str
    next_state: str
    service_start_seconds: float | None
    service_p95_seconds: float | None
    segments: tuple
    quality_allowed: bool=True
    costs_complete: bool=True


def choose(candidates, *, window_seconds,deadline_seconds,margin_seconds=.025,fast_allowed=False):
    """All times relative to decision; no missing cost becomes zero.

    Service segment includes device preparation; do not add its entry transition
    again. Outside-service transitions and holds occupy disjoint window segments.
    P95 is for deadline feasibility; segment durations are for expected energy.
    """
    if any(not math.isfinite(v) or v<0 for v in (window_seconds,deadline_seconds,margin_seconds)) or window_seconds==0:
        raise ValueError('Invalid decision window')
    accepted=[];rejected=[]
    for c in candidates:
        reason=None
        if c.mode not in ('Full','Fast','Continue'):raise ValueError('Unknown mode')
        if not c.costs_complete:reason='missing_measurement'
        elif not c.quality_allowed or (c.mode=='Fast' and not fast_allowed):reason='quality'
        elif abs(sum(s.seconds for s in c.segments)-window_seconds)>1e-6:reason='window_mismatch'
        elif c.mode=='Continue':
            if any(s.kind=='service' for s in c.segments):reason='continue_has_compute'
            elif window_seconds+margin_seconds>deadline_seconds:reason='continue_exceeds_deadline'
        else:
            services=[s for s in c.segments if s.kind=='service']
            if len(services)!=1:reason='service_accounting'
            elif c.service_start_seconds is None or c.service_p95_seconds is None:reason='missing_service_budget'
            elif not all(math.isfinite(v) and v>=0 for v in (c.service_start_seconds,c.service_p95_seconds)):reason='invalid_service_budget'
            elif c.service_p95_seconds==0:reason='invalid_service_budget'
            elif c.service_start_seconds+c.service_p95_seconds+margin_seconds>deadline_seconds:reason='deadline'
            else:
                elapsed=0.
                for segment in c.segments:
                    if segment.kind=='service':break
                    elapsed+=segment.seconds
                if abs(elapsed-c.service_start_seconds)>1e-6:reason='service_start_mismatch'
        if reason:rejected.append(dict(label=c.label,reason=reason))
        else:accepted.append(dict(label=c.label,mode=c.mode,compute_profile=c.compute_profile,
            next_state=c.next_state,energy_j=sum(s.energy_j for s in c.segments)))
    selected=min(accepted,key=lambda x:(x['energy_j'],x['label'])) if accepted else None
    return dict(selected=selected,candidates=accepted,rejected=rejected,
        status='selected' if selected else 'no_feasible_candidate')
