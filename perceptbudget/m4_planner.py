"""Complete one-refresh plans: immediate, deferred, or wait then mandatory Full."""
import json
from perceptbudget.m4_candidates import build,hold_energy
from perceptbudget.m4_selector import Candidate,Segment,choose

POLICY_REVISION='wait-first-absolute-prepare-v2'


def price(segments,waits,transitions,initial_profile,initial_age):
    state,age=initial_profile,initial_age;priced=[]
    for s in segments:
        s=dict(s);duration=s['seconds']
        if s['kind']=='hold':
            s.update(profile=state,energy_j=hold_energy(waits,state,age,duration));age+=duration
        elif s['kind']=='transition':
            target=s['target'];cost=transitions[state,target]
            extra=max(0.,hold_energy(waits,state,age,duration)-hold_energy(waits,state,cost.get('reference_quiet_age_seconds',0.),duration))
            s.update(source=state,energy_j=cost['energy_j']+extra,history_adjustment_j=extra)
            state=target;age=duration
        else:
            state=s['compute_profile'];age=0.
        priced.append(s)
    return priced


def refresh_plan(row,*,style,window,deadline,eta,current,age,waits,transitions,earliest_observation=0.,earliest_prepare=0.):
    raw=row['segments'];service_index=next(i for i,s in enumerate(raw) if s['kind']=='service')
    entry=sum(s['seconds'] for s in raw[:service_index] if s['kind']=='transition')
    post=sum(s['seconds'] for s in raw[service_index+1:] if s['kind']=='transition')
    p95=row['service_p95_seconds'];latest=min(row['service_start_seconds'],deadline-p95-eta,window-p95-eta-post)
    first_start=max(entry+earliest_prepare,earliest_observation)
    start=first_start if style=='now' else latest
    if start<first_start-1e-9:return None
    if start<entry-1e-9 or start+p95+eta>deadline+1e-9 or start+p95+eta+post>window+1e-9:return None
    start=max(start,entry);segments=[];mode,tail=row['label'].split(':');compute,next_state=tail.split('->')
    for i,s in enumerate(raw):
        s=dict(s)
        if i==0:s['seconds']=start-entry
        elif i==len(raw)-1:s['seconds']=window-sum(x['seconds'] for x in segments)
        if s['seconds']< -1e-9:return None
        s['seconds']=max(0.,s['seconds'])
        if s['kind']=='transition':s['target']=compute if i<service_index else next_state
        if s['kind']=='service':s['compute_profile']=compute
        segments.append(s)
    if style=='wait-first-latest':
        # Move a necessary downclock ahead of the wait, without moving neural
        # inference earlier or changing its geometry/profile/quality obligation.
        # The previous catalog only offered this wait-state choice before Full.
        if (len(segments)<3 or segments[0]['kind']!='hold' or segments[0]['seconds']<=1e-6
                or segments[1]['kind']!='transition' or segments[1]['target']!='gpu_emc'):
            return None
        segments=[segments[1],segments[0],*segments[2:]]
    segments=price(segments,waits,transitions,current,age)
    return dict(label=row['label']+'@'+style,policy_type=row['label'],service_mode=mode,
        compute_profile=compute,next_state=next_state,service_start_seconds=start,service_p95_seconds=p95,
        prepare_start_seconds=max(0.,start-entry),segments=segments,
        energy_j=sum(s['energy_j'] for s in segments))


def plan(config,services,waits,*,current_profile,quiet_age_seconds,window_seconds,deadline_seconds,
         transitions=None,critical_seconds=None,unexpected_change=False,bootstrap=False,consecutive_fast=0,inflight=False,earliest_observation_seconds=0.,earliest_prepare_seconds=0.,fast_permitted=True):
    if inflight:return dict(status='waiting_for_inflight',selected=None,plans=[],rejected=[])
    transitions=transitions or {};eta=config['timing']['safety_margin_seconds']
    deadline=min(deadline_seconds,critical_seconds) if critical_seconds is not None else deadline_seconds
    base=build(config,services,waits,current_profile=current_profile,quiet_age_seconds=quiet_age_seconds,
        window_seconds=window_seconds,deadline_seconds=deadline,critical_seconds=critical_seconds,
        unexpected_change=unexpected_change,bootstrap=bootstrap,consecutive_fast=consecutive_fast if fast_permitted else max(1,consecutive_fast),transitions=transitions)
    proposals=[]
    for row in base['catalog']:
        if not row['costs_complete'] or not row['quality_allowed'] or row['label'].startswith('Continue:'):continue
        for style in ('now',) if bootstrap else ('now','latest','wait-first-latest'):
            p=refresh_plan(row,style=style,window=window_seconds,deadline=deadline,eta=eta,
                current=current_profile,age=quiet_age_seconds,waits=waits,transitions=transitions,
                earliest_observation=earliest_observation_seconds,earliest_prepare=earliest_prepare_seconds)
            if p:proposals.append(p)
    # Continue is not free: price a concrete later Full on the SAME window.
    if not bootstrap and not unexpected_change:
        for held in config['hold_profiles']:
            transition=transitions.get((current_profile,held)) if held!=current_profile else None
            if held!=current_profile and transition is None:continue
            prefix=transition['seconds'] if transition else 0.
            if prefix>=min(window_seconds,deadline):continue
            sub=build(config,services,waits,current_profile=held,
                quiet_age_seconds=prefix if transition else quiet_age_seconds,
                window_seconds=window_seconds-prefix,deadline_seconds=deadline-prefix,
                consecutive_fast=1,transitions=transitions)
            following=[]
            for row in sub['catalog']:
                if not row['costs_complete'] or not row['label'].startswith('Full:'):continue
                p=refresh_plan(row,style='latest',window=window_seconds-prefix,deadline=deadline-prefix,eta=eta,
                    current=held,age=prefix if transition else quiet_age_seconds,waits=waits,transitions=transitions,
                    earliest_observation=max(0.,earliest_observation_seconds-prefix),
                    earliest_prepare=max(0.,earliest_prepare_seconds-prefix))
                if not p or p['prepare_start_seconds']<=1e-6:continue
                segments=([dict(kind='transition',seconds=prefix,target=held)] if transition else [])+p['segments']
                segments=price(segments,waits,transitions,current_profile,quiet_age_seconds)
                p.update(label='Continue:'+held+'->'+p['policy_type'],policy_type='Continue:'+held,
                    service_start_seconds=p['service_start_seconds']+prefix,
                    prepare_start_seconds=p['prepare_start_seconds']+prefix,
                    segments=segments,energy_j=sum(s['energy_j'] for s in segments))
                following.append(p)
            if following:proposals.append(min(following,key=lambda p:(p['energy_j'],p['label'])))
    # An event-deadline Fast must still deliver a Full before that event.
    # Replacing only a hold tail with the concrete Full makes quality obligations equal.
    complete=[]
    for p in proposals:
        if p['service_mode']!='Fast' or critical_seconds is None:
            complete.append(p);continue
        prefix=p['segments'][:-1];used=sum(s['seconds'] for s in prefix)
        remaining=window_seconds-used
        if remaining<=0:continue
        post=sum(s['seconds'] for s in prefix[next(i for i,s in enumerate(prefix) if s['kind']=='service')+1:])
        earliest=max(0.,p['service_start_seconds']+p['service_p95_seconds']+eta+post-used)
        future=plan(config,services,waits,current_profile=p['next_state'],quiet_age_seconds=post,
            window_seconds=remaining,deadline_seconds=max(0.,deadline-used),critical_seconds=max(0.,deadline-used),
            consecutive_fast=1,transitions=transitions,earliest_prepare_seconds=earliest)
        full=future['selected']
        if full is None:continue
        segments=price(prefix+full['segments'],waits,transitions,current_profile,quiet_age_seconds)
        p=dict(p,segments=segments,energy_j=sum(s['energy_j'] for s in segments),
            terminal_state=full['next_state'],followup=dict(mode='Full',compute_profile=full['compute_profile'],
                prepare_start_seconds=used+full['prepare_start_seconds'],
                service_start_seconds=used+full['service_start_seconds'],service_p95_seconds=full['service_p95_seconds']),
            obligation='Full_before_critical')
        complete.append(p)
    proposals=complete
    unique={}
    for p in proposals:
        signature=json.dumps([p['service_mode'],p['compute_profile'],p['next_state'],
            [(s['kind'],round(s['seconds'],7),s.get('target'),round(s['energy_j'],7)) for s in p['segments']]])
        if signature in unique:unique[signature].setdefault('equivalent_labels',[]).append(p['label'])
        else:unique[signature]=p
    plans=list(unique.values())
    candidates=[Candidate(p['label'],p['service_mode'],p['compute_profile'],p['next_state'],
        p['service_start_seconds'],p['service_p95_seconds'],tuple(Segment(s['kind'],s['seconds'],s['energy_j']) for s in p['segments']),
        additional_services=((p['followup']['service_start_seconds'],p['followup']['service_p95_seconds']),) if p.get('followup') else ()) for p in plans]
    decision=choose(candidates,window_seconds=window_seconds,deadline_seconds=deadline,margin_seconds=eta,fast_allowed=base['fast_allowed'])
    selected=next((p for p in plans if decision['selected'] and p['label']==decision['selected']['label']),None)
    if selected:
        selected=dict(selected,action_now='Continue' if selected['prepare_start_seconds']>1e-6 else selected['service_mode'])
    return dict(status=decision['status'],selected=selected,plans=plans,rejected=decision['rejected'],
        window_seconds=window_seconds,deadline_seconds=deadline,fast_allowed=base['fast_allowed'])
