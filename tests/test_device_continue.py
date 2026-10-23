from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import pytest

from perceptbudget.device_continue import control_window, ContinueLease


def test_window_executes_only_first_five_unique_committed_indices():
    actions = np.arange(50*7, dtype=np.float32).reshape(50, 7)
    executed = []
    def step(action):
        executed.append(action.copy())
        return {'step': len(executed)}, 0., False, {}
    env = SimpleNamespace(step=step, check_success=lambda: False)
    lease = SimpleNamespace(ensure=lambda *args: None)
    ticks = []
    with patch('perceptbudget.device_continue.time.sleep'):
        result, frames = control_window(env, {'step': 0}, actions, lease,
            lambda e, o: o.copy(), lambda o: o['step'], ticks.append, np=np)
    assert len(executed) == result['executed_steps'] == 5
    np.testing.assert_array_equal(executed, actions[:5])
    assert [r['source_action_index'] for r in ticks] == [0, 1, 2, 3, 4]
    assert not any(r['inference_active'] for r in ticks)
    assert frames == [0, 1, 2, 3, 4, 5]
    assert ticks[-1]['queue_after_step_actions'] == 0


def test_terminal_stops_without_padding_or_restarting_chunk():
    env = SimpleNamespace(step=lambda action: ({}, 0., True, {}), check_success=lambda: False)
    result, _ = control_window(env, {}, np.zeros((50, 7)), SimpleNamespace(ensure=lambda *a: None),
        lambda *a: {}, lambda o: 0, lambda r: None, np=np)
    assert result['executed_steps'] == 1


def test_short_or_nonfinite_prefix_rejected_before_environment_step():
    env = SimpleNamespace(step=lambda a: pytest.fail('Should not execute'))
    for actions in (np.zeros((4, 7)), np.full((5, 7), np.nan)):
        with pytest.raises(ValueError):
            control_window(env, {}, actions, None, None, None, None, np=np)


def test_continue_lease_is_short_and_has_dense_actual_clock_samples(tmp_path):
    lease = ContinueLease('balanced', tmp_path, 'one')
    command = lease.command()
    assert command[command.index('--seconds')+1] == '2'
    assert command[-2:] == ['--trace-interval', '.02']
    with pytest.raises(ValueError):
        ContinueLease('cpu1190', tmp_path, 'bad')


def test_continue_dry_run_has_one_model_no_long_open_loop_claim(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--continue-control', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as run:
        assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['model_loads'] == 1 and result['max_control_steps'] == 75
    assert result['inference_during_continue'] is False
    assert result['steady_hold_claim'] is False
    run.assert_not_called()


def test_camera_comparison_plan_is_resident_and_paired(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--continue-camera-compare', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as run:
        assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['model_loads'] == 1 and result['windows'] == 30
    assert result['max_control_steps'] == 150 and result['step_period_seconds'] == .07
    assert result['fresh_images_after_both'] and not result['production_baseline_changed']
    run.assert_not_called()


def test_control_without_frames_does_not_access_camera_observation():
    env = SimpleNamespace(step=lambda a: ({'state': 1}, 0., False, {}), check_success=lambda: False)
    with patch('perceptbudget.device_continue.time.sleep'):
        result, frames = control_window(env, {'state': 0}, np.zeros((50, 7)),
            SimpleNamespace(ensure=lambda *a: None), lambda e,o: o.copy(),
            lambda o: pytest.fail('No image access allowed'), lambda row: None,
            np=np, period=.07, capture_frames=False)
    assert result['executed_steps'] == 5 and frames == []
