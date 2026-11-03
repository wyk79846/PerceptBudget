"""Compare two Full frequencies from the retained M3 preparation trigger."""
import json
import math
from pathlib import Path

POLICY = 'full_energy_fixed_prepare_v1'
HISTORIES = ('under_0.2s','0.2_to_1s','at_least_1s')


def history(seconds):
    return HISTORIES[0] if seconds<.2 else HISTORIES[1] if seconds<1. else HISTORIES[2]


def load_costs(path,config):
    path=Path(path);data=json.loads(path.read_text())
    captured=json.loads((path.parent/'run.json').read_text())['config_snapshot']
    if data.get('status')!='completed':raise ValueError('Incomplete Full energy data')
    for key in ('H','N','precision','checkpoint_provenance_revision','state_enabled'):
        if captured[key]!=config[key]:raise ValueError('Full energy configuration mismatch: '+key)
    result=dict(source=str(path),cells=data['cells'],transitions=data['transitions'])
    for row in result['cells']:
        if not all(math.isfinite(row[k]) and row[k]>0 for k in ('mean_seconds','mean_power_w','total_energy_j')):
            raise ValueError('Invalid measured energy cell')
    return result


def hold(costs,profile,after_full,seconds):
    """Integrate measured post-Full response bins; not constant idle power."""
    energy=0.;parts=[]
    for a,b,h in ((0.,.2,HISTORIES[0]),(.2,1.,HISTORIES[1]),(1.,math.inf,HISTORIES[2])):
        duration=max(0.,min(seconds,b)-a)
        if not duration:continue
        rows=[r for r in costs['cells'] if r['mode']=='Continue' and r['profile']==profile
              and r['after_full_profile']==after_full and r['history']==h]
        if len(rows)!=1:raise ValueError('Missing measured Continue response')
        watts=rows[0]['mean_power_w'];energy+=watts*duration
        parts.append(dict(history=h,seconds=duration,power_w=watts))
    return dict(activity='Continue',profile=profile,seconds=seconds,energy_j=energy,bins=parts)


def price(costs,cell,window_seconds,minimum_idle):
    """Full includes entry/recovery. A post-Full lowering is counted once."""
    seconds=cell['mean_seconds'];tail=window_seconds-seconds
    if tail < -1e-9:raise ValueError('Measured Full mean exceeds common window')
    tail=max(0.,tail);profile=cell['profile']
    parts=[dict(activity='Full',profile=profile,seconds=seconds,
        energy_j=cell['mean_request_energy_j'],entry_recovery_included=True,
        from_profile=cell['from_profile'],history=cell['history'])]
    next_profile=profile
    if tail>=minimum_idle:
        rows=[r for r in costs['transitions'] if r['from_profile']==profile
              and r['profile']=='deep_idle' and r.get('during_task')]
        if not rows:raise ValueError('Missing measured post-Full lowering')
        duration=sum(r['milliseconds']/1000 for r in rows)/len(rows)
        energy=sum(r['switch_window_energy_j'] for r in rows)/len(rows)
        if duration>tail:raise ValueError('Lowering exceeds remaining time')
        parts.append(dict(activity='lower_to_deep',profile='deep_idle',seconds=duration,
            energy_j=energy,samples=len(rows),scope='Mean absolute switch-window energy; replaces hold during this interval'))
        tail-=duration;next_profile='deep_idle'
    parts.append(hold(costs,next_profile,profile,tail))
    return dict(profile=profile,from_profile=cell['from_profile'],cell_history=cell['history'],
        observations=cell['observations'],seconds=window_seconds,
        energy_j=sum(p['energy_j'] for p in parts),next_profile=next_profile,segments=parts)


def select(formula,decision,costs,*,quiet_age,minimum_idle):
    """Original preparation trigger, same start/end, original Continue rule.

    Only compare when the original M3 plan is due. Do not delay inference to
    the faster profile's later standalone start. Replan after its real result.
    """
    start=decision['decision_monotonic']
    # Critical arrival is the admission deadline, not the end of the energy
    # accounting window. A fresh Full produces a new buffer; its actual next
    # refresh is replanned after receipt, never cut off at this old boundary.
    end=start+decision['admitted_remaining_actions']*decision['estimated_control_period_seconds']
    window=max(0.,end-start);h=history(quiet_age);rows=[]
    for full in formula['candidates']:
        row=dict(profile=full['profile'],feasible=full['feasible'],complete=False)
        try:
            if not full['feasible']:raise ValueError('Cannot meet original deadline/RTC bounds')
            candidates=[r for r in costs['cells'] if r['mode']=='Full' and r['profile']==full['profile']
                        and r['from_profile']==decision['current_profile']]
            exact=[r for r in candidates if r['history']==h]
            available=exact or candidates
            if not available:raise ValueError('Missing measured Full entry route')
            estimates=[price(costs,c,window,minimum_idle) for c in available]
            # Keep each measured energy paired with its own duration. Unknown
            # dwell uses the more expensive whole-window estimate on this route.
            chosen=max(estimates,key=lambda r:r['energy_j'])
            row.update(chosen,complete=True,history_match=bool(exact),history_options=estimates)
        except ValueError as exc:row['missing_reason']=str(exc)
        rows.append(row)
    eligible=[r for r in rows if r['complete'] and r['feasible']]
    chosen=min(eligible,key=lambda r:(r['energy_j'],r['profile']!=formula['selected']['profile'])) if eligible else None
    return dict(policy=POLICY,profile=chosen['profile'] if chosen else formula['selected']['profile'],
        reason='minimum_measured_window_energy' if chosen else 'retain_original_missing_costs',
        energy_used_for_selection=chosen is not None,selected=chosen,candidates=rows,
        window_start=start,window_end=end,window_seconds=window,
        window_basis='current_admitted_buffer',arrival_deadline=decision['deadline'],
        quiet_age_seconds=quiet_age,requested_history=h,cost_source=costs['source'],
        scope='Two Full profiles at the original preparation trigger; original0.2s post-service lowering rule; no Fast or camera change')


def audit_record(record,costs,*,current_profile,minimum_idle):
    """Validate recorded segments against raw calibration cells, independently of price()."""
    errors=[]
    if abs(record['window_seconds']-(record['window_end']-record['window_start']))>1e-8:errors.append('window_endpoints')
    for route in record['candidates']:
        if not route['complete']:continue
        for option in route['history_options']:
            segments=option['segments'];service=segments[0]
            cells=[c for c in costs['cells'] if c['mode']=='Full' and c['profile']==route['profile']
                and c['from_profile']==current_profile and c['history']==option['cell_history']]
            if len(cells)!=1:errors.append('service_source');continue
            c=cells[0]
            if abs(service['seconds']-c['mean_seconds'])>1e-8 or abs(service['energy_j']-c['mean_request_energy_j'])>1e-8:
                errors.append('absolute_service')
            tail=record['window_seconds']-service['seconds']
            down=[p for p in segments if p['activity']=='lower_to_deep']
            if bool(down)!=(tail>=minimum_idle):errors.append('unchanged_idle_rule')
            if down:
                source=[s for s in costs['transitions'] if s['from_profile']==route['profile'] and s['profile']=='deep_idle' and s.get('during_task')]
                if not source:errors.append('lowering_source')
                elif (abs(down[0]['seconds']-sum(s['milliseconds']/1000 for s in source)/len(source))>1e-8
                      or abs(down[0]['energy_j']-sum(s['switch_window_energy_j'] for s in source)/len(source))>1e-8):errors.append('lowering_once')
            for segment in segments[1:]:
                if segment['activity']!='Continue':continue
                total=0.
                for a,b,h in ((0.,.2,HISTORIES[0]),(.2,1.,HISTORIES[1]),(1.,math.inf,HISTORIES[2])):
                    duration=max(0.,min(segment['seconds'],b)-a)
                    if not duration:continue
                    cell=next(x for x in costs['cells'] if x['mode']=='Continue' and x['profile']==segment['profile']
                        and x['after_full_profile']==route['profile'] and x['history']==h)
                    total+=duration*cell['mean_power_w']
                if abs(total-segment['energy_j'])>1e-8:errors.append('holding_response')
            if abs(sum(p['seconds'] for p in segments)-record['window_seconds'])>1e-8:errors.append('common_time_window')
            if abs(sum(p['energy_j'] for p in segments)-option['energy_j'])>1e-8:errors.append('energy_sum')
        if abs(route['energy_j']-max(x['energy_j'] for x in route['history_options']))>1e-8:errors.append('paired_history_fallback')
    eligible=[r for r in record['candidates'] if r['complete'] and r['feasible']]
    if record['energy_used_for_selection']!=bool(eligible):errors.append('selection_status')
    if eligible and record['profile']!=record['selected']['profile']:errors.append('selected_profile')
    if eligible and (record['selected'] not in eligible or record['selected']['energy_j']>min(r['energy_j'] for r in eligible)+1e-8):
        errors.append('minimum_energy')
    return sorted(set(errors))
