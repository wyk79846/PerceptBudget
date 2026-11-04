"""Engineering PACE-inspired phase hints from decoded commands, not true speed.

Only pronounced low regions associated with a predicted gripper/turn event
become protected intervals. Active Fast chunks can request one Full response.
"""
import math
from statistics import median
from perceptbudget.action_events import EventConfig,detect_events,_actions


def intervals(actions,config=None,*,events=None):
    config=config or EventConfig()
    actions=_actions(actions)
    if len(actions)<3:return []
    amplitude=[math.sqrt(sum(v*v for v in a[:3])) for a in actions]
    smooth=[median(amplitude[max(0,i-1):min(len(actions),i+2)]) for i in range(len(actions))]
    runs=[]
    for i,value in enumerate(smooth):
        if runs and runs[-1][2]==value:runs[-1][1]=i
        else:runs.append([i,i,value])
    peaks=[];valleys=[]
    for i,(start,stop,value) in enumerate(runs):
        left=runs[i-1][2] if i else None
        right=runs[i+1][2] if i+1<len(runs) else None
        if (left is None or value>left) and (right is None or value>right):peaks.append((start+stop)//2)
        if left is not None and value<left and (right is None or value<right):valleys.append((start+stop)//2)
    events=[e for e in (detect_events(actions,config) if events is None else events)
            if e['kind'] in ('gripper_close','gripper_open','turn')]
    phases=[]
    for valley in valleys:
        left=next((p for p in reversed(peaks) if p<valley),None)
        right=next((p for p in peaks if p>valley),None)
        if left is None:continue
        shoulder=min(smooth[left],smooth[right]) if right is not None else smooth[left]
        if shoulder<config.minimum_motion or smooth[valley]>shoulder*config.slowdown_ratio:continue
        stop=right+1 if right is not None else len(actions)
        anchors=[dict(kind=e['kind'],offset=e['offset']) for e in events if left<=e['offset']<stop]
        if not anchors:continue
        phase=dict(start_offset=left,stop_offset=stop,valley_offsets=[valley],anchors=anchors,
                   minimum_command_norm=smooth[valley],shoulder_command_norm=shoulder,
                   end_observed=right is not None)
        if phases and phase['start_offset']<phases[-1]['stop_offset']:
            old=phases[-1];old['stop_offset']=max(old['stop_offset'],stop)
            old['valley_offsets']+=phase['valley_offsets']
            old['anchors']=[dict(kind=k,offset=o) for k,o in sorted({(e['kind'],e['offset']) for e in old['anchors']+anchors})]
            old['minimum_command_norm']=min(old['minimum_command_norm'],smooth[valley])
            old['end_observed'] &= phase['end_observed']
        else:phases.append(phase)
    return phases


def absolute_intervals(actions,origin,config=None,*,events=None):
    return [dict(p,start_step=origin+p['start_offset'],stop_step=origin+p['stop_offset'])
            for p in intervals(actions,config,events=events)]


def intervals_for_timing(actions,origin,config,timing):
    """Use the same event evidence as the selected refresh decision.

    The motion shoulders/boundaries and thresholds stay unchanged. A new
    forecast only replaces the legacy gripper/turn anchors. In particular an
    empty new forecast must not silently fall back to expired-prefix events.
    """
    source=timing.get('phase_event_source')
    if source is None:
        return absolute_intervals(actions,origin,config)
    if source!='remaining_v1':
        raise ValueError('Unknown phase event source')
    proof=timing['critical_forecast'];result=proof['result'];at=proof['source_step']
    if (proof.get('version')!='remaining_v1' or result['current_step']!=at
            or not origin<=at<=origin+len(actions)):
        raise ValueError('Phase anchors require the active remaining-action forecast')
    anchors=[]
    for event in result['events']:
        target=event['target_step']
        if target!=at+event['offset'] or not at<=target<origin+len(actions):
            raise ValueError('Phase anchor outside its remaining-action source')
        anchors.append(dict(kind=event['kind'],offset=target-origin))
    return absolute_intervals(actions,origin,config,events=anchors)


def coverage(phases,*,step,now,latest_start,period,horizon,prediction_id):
    # Include acquisition/inference: a Fast observation taken during precision
    # manipulation may affect later actions even if its result arrives later.
    begin=step+math.floor(max(0.,latest_start-now)/period+1e-9)
    stop=begin+horizon
    overlaps=[p for p in phases if p['start_step']<stop and begin<p['stop_step']]
    return dict(prediction_id=prediction_id,start_step=begin,stop_step=stop,allowed=not overlaps,
                overlaps=overlaps,scope='forecast request through maximum action coverage; includes perception/inflight; future chunk unknown')


def full_response(phases,*,step,prediction_id,mode):
    """Only an active Fast chunk can create a one-Full response obligation."""
    if mode!='Fast':return None
    pending=[p for p in phases if p['stop_step']>step]
    if not pending:return None
    phase=min(pending,key=lambda p:(p['start_step'],p['stop_step']))
    return dict(prediction_id=prediction_id,start_step=phase['start_step'],stop_step=phase['stop_step'],
                target_step=max(step,phase['start_step']),state='active' if phase['start_step']<=step else 'upcoming')
