"""Optional local coverage: a timely Full closing event can cover its slowdown."""


def close_slowdowns(request,commit,events,covered,match_steps):
    timing=request.get('m3_timing') or {}
    target=timing.get('critical_step')
    if (request.get('m4_service_mode')!='Full' or request.get('observation_step',0)==0
            or target is None or 'gripper_close' not in timing.get('candidate_event_kinds',[])
            or timing.get('arrival_lateness_seconds',float('inf'))>0):
        return []
    if not any(e['kind']=='gripper_close' and abs(e['target_step']-target)<=match_steps for e in covered):
        return []
    closes=[e for e in events if e['kind']=='gripper_close'
        and commit['commit_step']<=e['target_step'] and abs(e['target_step']-target)<=match_steps]
    result=[]
    for event in events:
        if event['kind']!='slowdown' or event['target_step']<commit['commit_step']:continue
        close=next((e for e in closes if 0<event['target_step']-e['target_step']<=match_steps),None)
        if close is None or any(e['kind']=='slowdown' and e['target_step']==event['target_step'] for e in covered):continue
        result.append(dict(kind='slowdown',target_step=event['target_step'],close_target_step=close['target_step'],
            prediction_id=request['prediction_id'],commit_step=commit['commit_step'],
            gap_steps=event['target_step']-close['target_step'],rule='timely_full_close_then_slowdown'))
    return result
