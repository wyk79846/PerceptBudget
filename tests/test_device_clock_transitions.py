from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.device_clocks import cycle_trace, PROFILES
from perceptbudget.device_clock_transitions import transition_estimates, CycleLease, summarize_cycles


def test_bounded_schedule_changes_down_then_up_and_reference_is_noop():
    state = ['baseline']
    calls, events = [], []
    clock = [0.]
    def apply(profile):
        calls.append(profile)
        state[0] = profile
    def actual():
        gpu, emc, cpu = PROFILES[state[0]]
        return {'gpu/cur_freq': gpu, 'emc/rate': emc, 'policy/scaling_cur_freq': cpu}
    def sleep(seconds):
        clock[0] += seconds
    control = SimpleNamespace(apply=apply, actual=actual)
    with patch('time.monotonic', side_effect=lambda: clock[0]), patch('time.sleep', side_effect=sleep):
        cycle_trace(control, 'gpu_emc', events.append, cycles=2, phase_seconds=.01)
        assert calls == ['gpu_emc', 'baseline']*2
        calls.clear()
        cycle_trace(control, 'baseline', events.append, cycles=1, phase_seconds=.01)
    assert calls == []
    assert len([e for e in events if e['status'] == 'CYCLE_END']) == 3
    assert all(e['actual'] for e in events if e['status'] == 'PHASE')


def condition(label, profile, low=5., high=8., extra=.3):
    cycles = []
    for i in range(5):
        start = 6*i
        cycles.append(dict(low={'command_monotonic': start}, high={'command_monotonic': start+3},
            cycle_energy={'duration_seconds': 6, 'energy_j': low*3+high*3+extra},
            low_hold={'mean_power_w': low}, high_hold={'mean_power_w': high},
            down_observed_readback_ms=1., up_observed_readback_ms=2.))
    return dict(label=label, profile=profile, cycles=cycles)


def test_positive_cycle_energy_and_break_even():
    rows = [condition('pre', 'baseline', 8, 8, 0), condition('combo', 'gpu_emc'),
            condition('all', 'gpu_emc_cpu'), condition('post', 'baseline', 8, 8, 0)]
    result = transition_estimates(rows)
    assert result[1]['transition_excess_energy_j_estimate'] == pytest.approx(.3)
    assert result[1]['break_even_seconds'] == pytest.approx(.1)
    assert result[1]['hold_saving_w'] == 3


@pytest.mark.parametrize('extra', [0, -.3])
def test_nonpositive_signal_is_not_reported_as_break_even(extra):
    rows = [condition('pre', 'baseline', 8, 8, 0), condition('combo', 'gpu_emc', extra=extra),
            condition('all', 'gpu_emc_cpu', extra=extra), condition('post', 'baseline', 8, 8, 0)]
    result = transition_estimates(rows)
    assert result[1]['transition_energy_status'] == 'unresolved'
    assert result[1]['break_even_seconds'] is None


def test_baseline_drift_prevents_false_precision():
    rows = [condition('pre', 'baseline', 7, 7, 0), condition('combo', 'gpu_emc'),
            condition('all', 'gpu_emc_cpu'), condition('post', 'baseline', 9, 9, 0)]
    result = transition_estimates(rows)
    assert result[1]['transition_energy_status'] == 'unresolved'


def test_cycle_lease_only_allows_fixed_root_schedule(tmp_path):
    lease = CycleLease('gpu_emc', tmp_path, 'test')
    command = lease.command()
    assert command[-3:] == ['cycles', '--profile', 'gpu_emc']
    assert lease.duration == 40
    assert '--seconds' not in command and '--command' not in command


def test_cycle_window_summary_requires_frequency_and_power_coverage():
    from perceptbudget.power import PowerReport
    events = []
    for i in range(5):
        start = 6*i
        for phase, profile, offset in (('low', 'gpu_emc', 0), ('high', 'baseline', 3)):
            gpu, emc, cpu = PROFILES[profile]
            events.append(dict(status='PHASE', cycle=i, phase=phase,
                               command_monotonic=start+offset,
                               observed_match_monotonic=start+offset+.001))
            events.append(dict(status='SAMPLE', monotonic=start+offset+1,
                               actual={'gpu/cur_freq': gpu, 'emc/rate': emc,
                                       'cpu/scaling_cur_freq': cpu}))
        events.append(dict(status='CYCLE_END', cycle=i, monotonic=start+6))
    def report(a, b):
        return PowerReport(8000*(b-a), b-a, 1, 0, 8000*(b-a))
    monitor = SimpleNamespace(integrate_report=report)
    rows = summarize_cycles(events, monitor, 'gpu_emc')
    assert len(rows) == 5 and rows[0]['down_observed_readback_ms'] == 1
    assert rows[0]['cycle_energy']['energy_j'] == 48
    with pytest.raises(RuntimeError, match='frequency'):
        summarize_cycles([e for e in events if e['status'] != 'SAMPLE'], monitor, 'gpu_emc')
    partial = SimpleNamespace(integrate_report=lambda a, b: PowerReport(1, .1, .1, 0, None))
    with pytest.raises(RuntimeError, match='coverage'):
        summarize_cycles(events, partial, 'gpu_emc')


def test_transition_dry_run_has_zero_timed_model_requests(capsys):
    from scripts.profile_clock_screen import main
    import json
    with patch('sys.argv', ['probe', '--transitions', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as run:
        assert main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record['requests_per_profile'] == 0 and record['cycles_per_condition'] == 5
    assert set(record['profiles']) == {'baseline', 'gpu_emc', 'gpu_emc_cpu'}
    run.assert_not_called()
