"""One optional observation after an executed closing transition.

This detects issued gripper commands, not a successful grasp. It reads no
object/contact labels and does not change the actions or the RTC prefix.
"""
from perceptbudget.action_signal_experiment import grip


def confirmed_closure(rows, config):
    """Return the current confirmed open-to-close transition, if any."""
    n = config.confirm_steps
    end = len(rows)
    start = end
    while start and grip(rows[start-1]['action'], config.gripper_threshold) == 1:
        start -= 1
    if end-start < n or start < n:
        return None
    if any(grip(r['action'], config.gripper_threshold) != -1 for r in rows[start-n:start]):
        return None
    return dict(close_step=rows[start]['step'],
                confirmed_step=rows[start+n-1]['step']+1)


def saturated_lift_warning(rows, next_actions, config):
    """Near the measured gripper stop while planning lift: request feedback.

    This is a proprioception/action conflict, not a grasp-success detector.
    Opening and closing references come from this executed episode, without
    object poses, contact labels, task names or another perception model.
    """
    close=confirmed_closure(rows,config)
    n=config.confirm_steps
    if close is None or len(next_actions)<n:
        return None
    def width(row):
        state=row.get('after',{}).get('raw_state',[])
        return sum(abs(x) for x in state[6:8]) if len(state)>=8 else None
    opened=[width(r) for r in rows if r['step']<close['close_step']
            and grip(r['action'],config.gripper_threshold)==-1]
    opened=[v for v in opened if v is not None]
    recent=[width(r) for r in rows[-n:]]
    if not opened or max(opened)<=0. or any(v is None for v in recent):
        return None
    reference=max(opened)
    # Five percent of the actually observed open aperture, not object size.
    ratio=.05
    lifts=[a[2] for a in next_actions[:n]]
    if max(recent)>ratio*reference or min(lifts)<=config.minimum_motion:
        return None
    return dict(policy='saturated_gripper_lift_v1',close_step=close['close_step'],
        state_steps=[r['step'] for r in rows[-n:]],widths_m=recent,
        open_width_m=reference,stop_ratio=ratio,planned_lift_commands=lifts)


def request_cap(plan, *, rows, step, observation_step, inflight, config,
                full_seconds, previous=None, existing_wait_seconds=None):
    """Advance a distant observation once, leaving existing deadlines intact.

    An already planned observation within one hot Full service plus the usual
    margin is soon enough. Otherwise leave one normal control tick for device
    preparation and request feedback; the existing mode/frequency selector
    still owns inference. Do not turn a pre-close in-flight image into a
    post-close observation, and never overlap two requests.
    """
    close = confirmed_closure(rows, config)
    if close is None or observation_step >= close['confirmed_step'] or inflight:
        return None
    if previous and previous['close_step'] == close['close_step']:
        return previous
    now = plan['decision_monotonic']
    wait = max(0., plan['latest_start']-now)
    limit=full_seconds+plan['margin_seconds'] if existing_wait_seconds is None else existing_wait_seconds
    if wait <= limit:
        return None
    return dict(policy='post_close_observation_v1', **close,
                activation_step=step, activation_monotonic=now,
                observation_step=observation_step, request_step=step+1,
                ordinary_latest_start=plan['latest_start'],
                ordinary_wait_seconds=wait, full_seconds=full_seconds,
                existing_wait_threshold_seconds=limit,
                margin_seconds=plan['margin_seconds'])
