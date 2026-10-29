"""Small measured candidate set for mechanism3; no new clock combinations."""
import math
from perceptbudget.m4_selector import Candidate,Segment,choose

AGE_EDGES=(0.,.2,.5,1.,float('inf'))


def wait_model(summary):
    raw=next(r for r in summary['results'] if r['guard_seconds']==0.)
    result={}
    for profile in ('baseline','gpu_emc'):
        bins=[]
        for left,right in zip(AGE_EDGES,AGE_EDGES[1:]):
            ws=[w for e in raw['episodes'] for w in e['windows']
                if w['profile']==profile and left<=w['quiet_age_seconds']<right]
            seconds=sum(w['seconds'] for w in ws)
            bins.append(sum(w['energy_j'] for w in ws)/seconds if seconds else None)
        result[profile]=bins
    return result


def hold_energy(model,profile,age,seconds):
    if seconds==0:return 0.
    end=age+seconds;energy=0.
    for index,(left,right) in enumerate(zip(AGE_EDGES,AGE_EDGES[1:])):
        duration=max(0.,min(end,right)-max(age,left))
        if duration:
            watts=model[profile][index]
            if watts is None:raise ValueError('Missing wait-age measurement')
            energy+=watts*duration
    return energy


def templates(config):
    return [(m,c,s) for m in ('Full','Fast') for c in config['compute_profiles'] for s in config['hold_profiles']]+[
        ('Continue','none',s) for s in config['hold_profiles']]


def build(config,services,waits,*,current_profile,quiet_age_seconds,window_seconds,deadline_seconds,
          critical_seconds=None,unexpected_change=False,bootstrap=False,consecutive_fast=0,inflight=False,transitions=None):
    if current_profile not in config['hold_profiles']:raise ValueError('Unsupported current state')
    if any(not math.isfinite(v) or v<0 for v in (quiet_age_seconds,window_seconds,deadline_seconds)) or window_seconds==0:
        raise ValueError('Invalid runtime timing')
    if critical_seconds is not None and (not math.isfinite(critical_seconds) or critical_seconds<0):
        raise ValueError('Invalid critical time')
    transitions=transitions or {}
    def change(source,target,age=None):
        if source==target:return None
        if (source,target) not in transitions:raise ValueError('missing_directional_preparation')
        measured=transitions[source,target]
        result=dict(measured)
        if age is not None:
            duration=measured['seconds'];reference=measured.get('reference_quiet_age_seconds',0.)
            extra=max(0.,hold_energy(waits,source,age,duration)-hold_energy(waits,source,reference,duration))
            result['energy_j']+=extra
            result['history_adjustment_j']=extra
        return result
    eta=config['timing']['safety_margin_seconds'];candidates=[];catalog=[]
    full_budget=max(c['p95_service_seconds'] for c in services if c['mode']=='Full' and c['profile']=='baseline')
    fast_allowed=(not bootstrap and not unexpected_change and consecutive_fast<1 and
        (critical_seconds is None or critical_seconds>full_budget+eta))
    deadline=min(deadline_seconds,critical_seconds) if critical_seconds is not None else deadline_seconds
    for mode,profile,next_state in templates(config):
        label=f'{mode}:{profile}->{next_state}';segments=[];start=p95=None;missing=None;source=[]
        allowed=not inflight and (mode!='Fast' or fast_allowed)
        try:
            if mode=='Continue':
                transition=change(current_profile,next_state,quiet_age_seconds)
                duration=transition['seconds'] if transition else 0.
                if duration>window_seconds:raise ValueError('preparation_exceeds_window')
                next_budgets=[]
                for future_mode in ('Full','Fast') if fast_allowed else ('Full',):
                    for future_profile in config['compute_profiles']:
                        measured=[c['p95_service_seconds'] for c in services if c['mode']==future_mode and c['profile']==future_profile and c['from_profile']==next_state]
                        if measured:next_budgets.append(max(measured))
                reserve=min(next_budgets) if next_budgets else float('inf')
                allowed=not bootstrap and not unexpected_change and window_seconds+reserve+eta<=deadline
                if transition:segments.append(Segment('transition',duration,transition['energy_j']))
                segments.append(Segment('hold',window_seconds-duration,hold_energy(waits,next_state,
                    duration if transition else quiet_age_seconds,window_seconds-duration)))
            else:
                # Complete service includes its measured recovery. Do not add it again.
                available=[c for c in services if c['mode']==mode and c['profile']==profile and c['from_profile']==current_profile]
                entry=None
                if not available:
                    entry=change(current_profile,profile)
                    available=[c for c in services if c['mode']==mode and c['profile']==profile and c['from_profile']==profile]
                if not available:raise ValueError('missing_service_entry_state')
                post=change(profile,next_state,0.)
                if next_state not in waits:raise ValueError('missing_hold_profile')
                # Short/long recovery are calibration histories, not hardware states.
                # Use the larger observed recovery estimate until dwell-specific online updates exist.
                p95=max(c['p95_service_seconds'] for c in available)
                mean=max(c['mean_service_seconds'] for c in available)
                energy=max(c['mean_absolute_service_energy_j'] for c in available)
                source=[c['history'] for c in available]
                entry_seconds=entry['seconds'] if entry else 0.
                post_seconds=post['seconds'] if post else 0.
                start=max(entry_seconds,min(deadline-p95-eta,window_seconds-mean-post_seconds))
                remaining=window_seconds-start-mean-post_seconds
                if remaining < -1e-9:raise ValueError('service_exceeds_window')
                remaining=max(0.,remaining)
                before=start-entry_seconds
                segments=[Segment('hold',before,hold_energy(waits,current_profile,quiet_age_seconds,before))]
                if entry:
                    entry=change(current_profile,profile,quiet_age_seconds+before)
                    segments.append(Segment('transition',entry_seconds,entry['energy_j']))
                segments.append(Segment('service',mean,energy))
                if post:segments.append(Segment('transition',post_seconds,post['energy_j']))
                segments.append(Segment('hold',remaining,hold_energy(waits,next_state,post_seconds,remaining)))
        except (ValueError,KeyError) as exc:missing=str(exc)
        c=Candidate(label,mode,profile,next_state,start,p95,tuple(segments),allowed,missing is None)
        candidates.append(c);catalog.append(dict(label=label,costs_complete=c.costs_complete,
            missing_reason=missing,quality_allowed=allowed,service_histories=source,
            service_start_seconds=start,service_p95_seconds=p95,
            segments=[dict(kind=s.kind,seconds=s.seconds,energy_j=s.energy_j) for s in segments]))
    decision=choose(candidates,window_seconds=window_seconds,deadline_seconds=deadline,
        margin_seconds=eta,fast_allowed=fast_allowed)
    decision.update(catalog=catalog,template_count=len(candidates),measured_count=sum(c.costs_complete for c in candidates),
        eligible_count=len(decision['candidates']),fast_allowed=fast_allowed)
    return decision
