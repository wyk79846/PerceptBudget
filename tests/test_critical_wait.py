from perceptbudget.critical_wait import should_wait


def test_execute_available_prefix_then_wait_only_at_boundary():
    request = {'m3_timing': {'reason': 'action_event', 'critical_step': 15}}
    assert not should_wait(request, 10)
    assert not should_wait(request, 14)
    assert should_wait(request, 15)
    assert should_wait(request, 16)


def test_no_request_or_buffer_deadline_does_not_pause():
    assert not should_wait(None, 15)
    assert not should_wait({'m3_timing': {'reason': 'buffer', 'critical_step': 15}}, 15)
    assert not should_wait({'m3_timing': {'reason': 'action_event', 'critical_step': None}}, 15)


def test_completed_phase_suppresses_repeat_without_sliding_or_hiding_opposite_grip():
    from perceptbudget.critical_wait import PhaseWait
    from perceptbudget.action_events import EventConfig
    guard = PhaseWait()
    guard.complete(dict(phase=dict(start_step=41, stop_step=64), kinds=['gripper_close']))
    def request(step, kind):
        return dict(m3_timing=dict(reason='action_event', critical_step=step,
                                  candidate_event_kinds=[kind]))
    cfg = EventConfig()
    assert not guard.review(request(56, 'turn'), [], 49, cfg)['wait']
    assert guard.completed['stop_step'] == 64
    assert guard.review(request(64, 'turn'), [], 49, cfg)['wait']
    assert guard.review(request(56, 'gripper_open'), [], 49, cfg)['wait']


def test_completion_without_observed_phase_end_does_not_cover_whole_horizon():
    from perceptbudget.critical_wait import PhaseWait
    guard = PhaseWait()
    guard.complete(dict(phase=None, kinds=['turn']))
    assert guard.completed is None


def test_phase_event_coverage_keeps_opposite_grip_and_outside_events():
    from perceptbudget.critical_wait import covered_by_phase
    phase=dict(start_step=44,stop_step=80,kinds=['gripper_close'])
    assert covered_by_phase(dict(target_step=52,kind='turn'),phase)
    assert covered_by_phase(dict(target_step=60,kind='slowdown'),phase)
    assert not covered_by_phase(dict(target_step=74,kind='gripper_open'),phase)
    assert not covered_by_phase(dict(target_step=80,kind='turn'),phase)


def test_known_wait_limits_old_actions_to_those_that_will_actually_execute():
    from perceptbudget.critical_wait import rtc_prefix_plan
    request=dict(m3_timing=dict(reason='action_event',critical_step=46))
    plan=rtc_prefix_plan(request,38,10,dict(wait=True))
    assert plan['maximum_consumed_steps']==8
    assert plan['selected_delay_steps']==8
    # The inference still takes ten ticks of wall time; after eight physical
    # steps control pauses, so the last two old actions are not committed.
    assert plan['estimated_delay_steps']==10
    request['m3_timing']['critical_step']=38
    assert rtc_prefix_plan(request,38,10,dict(wait=True))['selected_delay_steps']==0


def test_phase_reopens_moved_anchor_but_keeps_nearby_and_other_phase_events():
    from perceptbudget.critical_wait import covered_by_phase
    phase=dict(start_step=49,stop_step=68,target_step=52,kinds=['gripper_close'])
    close=lambda target:dict(kind='gripper_close',target_step=target)
    assert covered_by_phase(close(58),phase)  # Frozen legacy behavior.
    assert covered_by_phase(close(57),phase,match_steps=5)
    assert not covered_by_phase(close(58),phase,match_steps=5)
    assert covered_by_phase(dict(kind='slowdown',target_step=60),phase,match_steps=5)
    assert not covered_by_phase(dict(kind='gripper_open',target_step=60),phase,match_steps=5)
    assert not covered_by_phase(close(68),phase,match_steps=5)


def test_phase_dedup_that_allows_control_to_continue_must_not_cap_rtc():
    from perceptbudget.critical_wait import PhaseWait,rtc_prefix_plan
    from perceptbudget.action_events import EventConfig
    guard=PhaseWait()
    guard.complete(dict(phase=dict(start_step=41,stop_step=64),kinds=['gripper_close']))
    request=dict(m3_timing=dict(reason='action_event',critical_step=56,candidate_event_kinds=['turn']))
    review=guard.review(request,[],49,EventConfig())
    assert not review['wait']
    assert rtc_prefix_plan(request,49,10,review)['selected_delay_steps']==10
    # An opposite gripper transition must stop even in that same phase.
    request['m3_timing']['candidate_event_kinds']=['gripper_open']
    review=guard.review(request,[],49,EventConfig())
    assert review['wait']
    assert rtc_prefix_plan(request,49,10,review)['selected_delay_steps']==7
