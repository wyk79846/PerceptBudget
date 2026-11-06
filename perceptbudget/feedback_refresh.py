"""Spend a new Full on new feedback, not every predictable action transition."""
import math


def review_cues(events, *, prediction_id, origin, source_step, period,
                response_seconds, margin_seconds, minimum_steps, actions,
                reference, config, policy='overlap_pacing_v1', full_approach_steps=None):
    """Freeze one feedback window per chunk, anchored to its observation.

    Ordinary cues inside the first useful request time plus a complete Full
    budget are executed from the fresh plan. A gripper transition newly absent
    from an overlapping older forecast can still ask for earlier feedback.
    Buffer/RTC admission is outside this cue filter and is never relaxed.
    """
    if policy not in ('overlap_pacing_v1','overlap_pacing_v2'):
        raise ValueError('Unknown feedback policy')
    if (type(minimum_steps) is not int or minimum_steps<1 or period<=0
            or response_seconds<=0 or margin_seconds<0):
        raise ValueError('Invalid feedback window')
    earliest=origin+minimum_steps
    end=earliest+math.ceil((response_seconds+margin_seconds)/period-1e-9)
    n=config.confirm_steps

    def direction(values, index):
        segment=values[index:index+n] if index>=0 else []
        if len(segment)!=n:return None
        vector=[sum(a[k] for a in segment)/n for k in range(3)]
        return vector if sum(x*x for x in vector)>=config.minimum_motion**2 else None

    first_slowdown=min((e['target_step'] for e in events if e['kind']=='slowdown'),default=None)
    reviews=[]
    for event in events:
        target=event['target_step'];kind=event['kind'];matched=None;angle=None
        covered=bool(reference and reference['origin']<=target
            and target+n<=reference['origin']+len(reference['actions']))
        if reference:
            candidates=[e for e in reference['events'] if e['kind']==kind
                and abs(e['target_step']-target)<=config.match_steps]
            matched=min(candidates,key=lambda e:abs(e['target_step']-target),default=None)
        consistent=matched is not None
        if consistent and kind=='turn':
            a=direction(reference['actions'],matched['target_step']-reference['origin'])
            b=direction(actions,target-origin)
            if a is not None and b is not None:
                cosine=sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
                angle=math.degrees(math.acos(max(-1.,min(1.,cosine))))
            consistent=angle is not None and angle+1e-9<config.turn_degrees
        new_gripper=kind.startswith('gripper_') and covered and matched is None
        keep_gripper=policy=='overlap_pacing_v2' and kind.startswith('gripper_')
        first_contact=policy=='overlap_pacing_v2' and prediction_id==0 and kind=='slowdown' and target==first_slowdown
        approach=kind=='turn' and full_approach_steps is not None and target in full_approach_steps
        reason=('approach_feedback' if approach else 'gripper_feedback' if keep_gripper else 'initial_slowdown_feedback' if first_contact
                else 'consistent_forecast' if consistent else 'new_gripper_transition' if new_gripper
                else 'fresh_plan_window' if target<end else 'new_feedback_deadline')
        reviews.append(dict(kind=kind,prediction_id=prediction_id,target_step=target,
            reference_target_step=matched['target_step'] if matched else None,
            reference_covers_target=covered,direction_difference_degrees=angle,
            suppress=reason in ('consistent_forecast','fresh_plan_window'),reason=reason))
    return dict(policy=policy,prediction_id=prediction_id,observation_step=origin,
        source_step=source_step,period_seconds=period,response_seconds=response_seconds,
        margin_seconds=margin_seconds,minimum_request_steps=minimum_steps,
        earliest_ordinary_request_step=earliest,fresh_plan_end_step=end,
        reference_prediction_id=reference['prediction_id'] if reference else None,events=reviews,
        **({'full_approach_steps':list(full_approach_steps)} if full_approach_steps is not None else {}))
