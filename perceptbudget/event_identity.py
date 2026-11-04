"""Check a repeated turn against the latest still-active turn response."""
import math

from perceptbudget.critical_wait import covered_by_phase


def review_turn(event, phases, chunks, config):
    """Same type/window is insufficient when the intended direction changed.

    Use decoded commands from the completed Full and the current prediction.
    The first unexecuted response command is used if its old target has passed.
    Another event type does not erase a turn; a newer turn response supersedes
    an older one for this overlap. Original phase ends are never extended.
    """
    phase=max((p for p in phases if covered_by_phase(event,p)),
        key=lambda p:p['completed_prediction_id'])
    reference=phase['completed_prediction_id']
    anchor=max(phase['target_step'],phase['completed_step'])

    def direction(pid,step):
        chunk=chunks[pid];offset=step-chunk['origin']
        window=chunk['phase_actions'][offset:offset+config.confirm_steps] if offset>=0 else []
        if len(window)!=config.confirm_steps:return None
        vector=[sum(a[k] for a in window)/len(window) for k in range(3)]
        return vector if math.sqrt(sum(v*v for v in vector))>=config.minimum_motion else None

    old=direction(reference,anchor);new=direction(event['prediction_id'],event['target_step'])
    angle=None
    if old is not None and new is not None:
        cosine=sum(a*b for a,b in zip(old,new))/math.sqrt(sum(a*a for a in old)*sum(b*b for b in new))
        angle=math.degrees(math.acos(max(-1.,min(1.,cosine))))
    same=angle is not None and angle+1e-9<config.turn_degrees
    return dict(kind=event['kind'],prediction_id=event['prediction_id'],target_step=event['target_step'],
        reference_prediction_id=reference,reference_action_step=anchor,
        reference_direction=old,new_direction=new,angle_degrees=angle,
        threshold_degrees=config.turn_degrees,same_event=same,
        reason='direction_unavailable' if angle is None else 'same_turn_direction' if same else 'new_turn_direction')


def review_ordinary_turn(event, *, actions, origin, source_step, history, phases, events, config):
    """A smooth isolated bend need not be an early-refresh deadline.

    Keep low-amplitude phases, gripper/slowdown neighbours and abrupt direction
    changes. Read actual controls before this forecast's remaining-action cut.
    The ordinary refresh cap and all supply bounds stay in the scheduler.
    """
    at=event['target_step'];n=config.confirm_steps
    def direction(steps):
        values=[]
        for step in steps:
            index=step-origin
            action=actions[index] if step>=source_step and 0<=index<len(actions) else history.get(step)
            if action is None:return None
            values.append(action)
        vector=[sum(a[k] for a in values)/len(values) for k in range(3)]
        return vector if math.sqrt(sum(v*v for v in vector))>=config.minimum_motion else None
    before=direction(range(at-n,at));after=direction(range(at,at+n));angle=None
    if before is not None and after is not None:
        cosine=sum(a*b for a,b in zip(before,after))/math.sqrt(sum(a*a for a in before)*sum(b*b for b in after))
        angle=math.degrees(math.acos(max(-1.,min(1.,cosine))))
    overlaps=[[p['start_step'],p['stop_step']] for p in phases if p['start_step']<=at<p['stop_step']]
    grip=min((abs(e['target_step']-at) for e in events if e['kind'].startswith('gripper_')),default=None)
    slow=min((abs(e['target_step']-at) for e in events if e['kind']=='slowdown'),default=None)
    reason=('protected_motion_phase' if overlaps else
        'near_gripper' if grip is not None and grip<=config.match_steps else
        'near_slowdown' if slow is not None and slow<=config.match_steps else
        'direction_unavailable' if angle is None else
        'abrupt_turn' if angle+1e-9>=config.turn_degrees else 'ordinary_bend')
    return dict(kind='turn',prediction_id=event['prediction_id'],target_step=at,source_step=source_step,
        before_direction=before,after_direction=after,local_angle_degrees=angle,
        protected_intervals=overlaps,nearest_gripper_steps=grip,nearest_slowdown_steps=slow,
        ordinary=reason=='ordinary_bend',reason=reason)
