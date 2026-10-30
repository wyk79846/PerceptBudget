from copy import deepcopy
import pytest

from perceptbudget.m4_acceptance import configure_m3, configure_m3_fast
from perceptbudget.closed_loop import DeadlineRuntime
from perceptbudget.deadline_controller import FullServiceTimes


def test_fast_setup_preserves_m3_deadline_and_device_rules():
    a,b={},{}
    sa,sb=dict(timing={}),dict(timing={})
    configure_m3(a,sa);configure_m3_fast(b,sb)
    assert sa==sb
    for field in ('m3_action_events_enabled','m3_dvac_enabled','m3_dvfs_policy',
                  'm3_scheduler_mode','m3_event_coverage_mode','m3_hold_high_clocks'):
        assert a[field]==b[field]
    assert b['m4_enabled'] is False and b['m3_fast_enabled'] is True
    assert b['m2_fast_no_state'] and b['m4_fast_prompt_capacity']==20
    assert 'm4_settings' not in b
    configure_m3(b,sb)
    assert b['m3_fast_enabled'] is False


def test_fast_completion_cannot_shrink_full_service_history():
    runtime=object.__new__(DeadlineRuntime)
    runtime.chunks={};runtime.dvac_enabled=False;runtime.action_events_enabled=False
    runtime.merge_gap=None;runtime.open_probe=None;runtime.event_coverage_mode='event_only'
    runtime.settings={'margin_updates':False};runtime.fast_selection=True
    runtime.times=FullServiceTimes({k:[.5] for k in ('hot','short','long')})
    request=dict(prediction_id=1,observation_step=10,observation_monotonic=1.,
        actions=[],m3_fast_choice={'mode':'Fast'},
        m3_timing=dict(deadline=3.,critical_step=None,reason='buffer',history='hot'))
    commit=dict(commit_monotonic=1.2,commit_step=14)
    before=deepcopy(runtime.times.samples)
    runtime.committed(request,commit)
    assert runtime.times.samples==before
    request['m3_fast_choice']['mode']='Full'
    runtime.committed(request,dict(commit,commit_monotonic=1.6))
    assert list(runtime.times.samples['hot'])==pytest.approx([.5,.6])
