from copy import deepcopy
from perceptbudget.m4_event_coverage import close_slowdowns


def example():
    q=dict(prediction_id=1,observation_step=26,m4_service_mode='Full',m3_timing=dict(
        critical_step=39,candidate_event_kinds=['gripper_close'],arrival_lateness_seconds=0.))
    events=[dict(kind='gripper_close',target_step=40),dict(kind='slowdown',target_step=43),
        dict(kind='slowdown',target_step=46),dict(kind='turn',target_step=42),dict(kind='gripper_open',target_step=44)]
    return q,{'commit_step':37},events,[dict(kind='gripper_close',target_step=39)]


def test_close_only_covers_near_future_slowdown_and_does_not_mutate_events():
    args=example();old=deepcopy(args)
    added=close_slowdowns(*args,5)
    assert [e['target_step'] for e in added]==[43]
    assert added[0]['gap_steps']==3 and args==old
    assert close_slowdowns(args[0],args[1],args[2],args[3]+added,5)==[]


def test_no_coverage_for_fast_late_full_or_past_close():
    for mode,late,step in [('Fast',0.,37),('Full',.01,37),('Full',0.,41)]:
        q,c,events,covered=example();q['m4_service_mode']=mode;q['m3_timing']['arrival_lateness_seconds']=late;c['commit_step']=step
        assert close_slowdowns(q,c,events,covered,5)==[]
    q,c,events,covered=example();q['m3_timing']['candidate_event_kinds']=['gripper_open']
    assert close_slowdowns(q,c,events,covered,5)==[]
