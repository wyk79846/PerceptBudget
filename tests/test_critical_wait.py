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
