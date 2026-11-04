"""Prototype critical-time estimate from unexecuted decoded LIBERO actions.

History must contain executed controls, never the new prediction's expired
prefix. XYZ-command amplitude is a motion proxy, not measured physical speed.
The scheduler can select this module through its optional version interface;
frequency selection remains outside this module.
"""
import math

from perceptbudget.action_events import EventConfig, _actions


def _norm(vector):
    return math.sqrt(sum(v*v for v in vector))


def _mean_direction(actions):
    if not actions:return None
    return tuple(sum(a[k] for a in actions)/len(actions) for k in range(3))


def _angle(a,b):
    return math.degrees(math.acos(max(-1.,min(1.,sum(x*y for x,y in zip(a,b))/(_norm(a)*_norm(b))))))


def forecast(remaining_actions, *, previous_actions=(), current_step, now,
             control_period, config=None):
    """Earliest supported command change and its predicted monotonic arrival.

    Gripper changes use the last executed command as their starting state.
    Slowdown compares sustained low commands with the preceding motion peak.
    Turns accumulate heading change, so gradual turns can also produce a cue.
    Motion cues can require longer support in the already predicted future.
    This confirms the same onset, rather than waiting for more controls to run.
    Near the end, retain the original confirmation when future support is unknown.
    """
    if type(current_step) is not int or current_step<0:
        raise ValueError('current_step must be a nonnegative integer')
    if not math.isfinite(now) or not math.isfinite(control_period) or control_period<=0:
        raise ValueError('Use finite current time and positive measured control period')
    cfg=config or EventConfig();n=cfg.confirm_steps
    motion_n=cfg.motion_confirm_steps or n
    future=_actions(remaining_actions);history=_actions(previous_actions)[-n:]
    events=[]

    def emit(kind,i,*,confirmation_steps=None,**evidence):
        count=n if confirmation_steps is None else confirmation_steps
        events.append(dict(kind=kind,offset=i,confirmed_offset=i+count-1,
            target_step=current_step+i,t_critical=now+i*control_period,**evidence))

    def grip(a):
        return 1 if a[6]>=cfg.gripper_threshold else -1 if a[6]<=-cfg.gripper_threshold else 0

    state=next((grip(a) for a in reversed(history) if grip(a)),0)
    for i in range(len(future)-n+1):
        value=grip(future[i])
        if value and value!=state and all(grip(a)==value for a in future[i:i+n]):
            if state:
                emit('gripper_close' if value>0 else 'gripper_open',i,
                     from_command=state,to_command=value)
            state=value

    amplitude=[_norm(a[:3]) for a in future]
    peak=max((_norm(a[:3]) for a in history),default=amplitude[0] if amplitude else 0.)
    moving=peak>=cfg.minimum_motion
    for i,value in enumerate(amplitude):
        if moving:
            window=amplitude[i:i+motion_n]
            if len(window)>=n and max(window)<=peak*cfg.slowdown_ratio:
                emit('slowdown',i,confirmation_steps=len(window),
                     reference_command_norm=peak,next_command_norm=value)
                moving=False
            else:peak=max(peak,value)
        elif value>=cfg.minimum_motion and value>peak*cfg.slowdown_ratio:
            peak=value;moving=True

    direction=_mean_direction(history)
    if direction is None or _norm(direction)<cfg.minimum_motion:
        direction=next((a[:3] for a in future if _norm(a[:3])>=cfg.minimum_motion),None)
    for i in range(len(future)-n+1):
        window=future[i:i+motion_n]
        if direction is None or not all(_norm(a[:3])>=cfg.minimum_motion for a in window):
            continue
        angles=[_angle(direction,a[:3]) for a in window]
        if all(angle+1e-9>=cfg.turn_degrees for angle in angles):
            emit('turn',i,confirmation_steps=len(window),angle_degrees=angles[0])
            updated=_mean_direction(window)
            if _norm(updated)>=cfg.minimum_motion:direction=updated

    priority={'gripper_close':0,'gripper_open':0,'slowdown':1,'turn':2}
    events.sort(key=lambda e:(e['offset'],priority[e['kind']]))
    earliest=events[0]['offset'] if events else None
    return dict(current_step=current_step,remaining_steps=len(future),
        critical_step=current_step+earliest if earliest is not None else None,
        t_critical=now+earliest*control_period if earliest is not None else None,
        event_kinds=sorted({e['kind'] for e in events if e['offset']==earliest}),
        events=events,control_period_seconds=control_period,
        motion_units='decoded XYZ command amplitude; not physical speed')
