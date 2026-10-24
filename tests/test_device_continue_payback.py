from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np
import pytest

from perceptbudget.device_continue_payback import execute_handoff


def run_fake_handoff(latency=.2):
    now = [0.]
    executed = []
    raw = np.repeat(np.arange(50, dtype=np.float32)[:, None], 32, axis=1)
    old = raw[:, :7].copy()
    class Env:
        def step(self, action):
            executed.append(float(action[0]))
            now[0] += .01
            return {}, 0., False, {}
        def check_success(self):
            return False
    class Adapter:
        def predict(self, snapshot, prompt, previous, *, delay, executed, request):
            assert executed == 4 and previous.shape == (46, 32)
            assert previous[0, 0] == 4 and delay == 10
            request.update(actions_raw=(raw+100).tolist(), rtc_guidance={'variable_length_fallback': False})
            return old+100
    class Future:
        def __init__(self, fn):
            self.fn = fn
            self.ready = now[0]+latency
            self.cached = None
        def done(self):
            return now[0]+1e-10 >= self.ready
        def result(self):
            if self.cached is None:
                self.cached = self.fn()
            return self.cached
    lease = SimpleNamespace(ensure=lambda *args: None,
        receive=lambda *args: {'down': {'command_monotonic': 0.}, 'up': {'command_monotonic': .2}})
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    obs = {'agentview_image': frame, 'robot0_eye_in_hand_image': frame}
    snapshot = {'raw_state': np.zeros(8, dtype=np.float32)}
    from contextlib import nullcontext
    with patch('perceptbudget.device_continue_payback.time.monotonic', side_effect=lambda: now[0]), \
         patch('perceptbudget.device_continue_payback.time.sleep', side_effect=lambda t: now.__setitem__(0, now[0]+t)), \
         patch('perceptbudget.device_continue_payback.suspend_cameras', side_effect=lambda env: nullcontext()), \
         patch('perceptbudget.device_continue_payback.fresh_observation', return_value=(obs, {'start_monotonic': .2})), \
         patch('scripts.eval_q._make_snapshot', return_value=snapshot):
        result = execute_handoff(Env(), obs, old, {'actions_raw': raw.tolist()},
            observation_step=4, adapter=Adapter(), prompt='test', stats={}, torch=None, np=np,
            executor=SimpleNamespace(submit=lambda fn: Future(fn)), lease=lease, index=0, delay_estimate=10)
    return result, executed


def test_actual_consumer_crop_executes_old_actions_during_prediction_then_new_tail():
    result, actions = run_fake_handoff()
    assert result['committed']['step'] == 8
    assert result['committed']['expired_offsets'] == 4
    assert result['committed']['old_buffer_actions_at_commit'] == 42
    assert actions[:8] == list(range(8))
    assert actions[8:] == list(range(104, 109))
    assert result['new_actions_executed'] == 5 and result['underflows'] == 0
    assert [r['source_offset'] for r in result['control_rows'][8:]] == list(range(4, 9))


def test_real_delay_exhaustion_is_failure_not_pause_or_zero_action_extension():
    with pytest.raises(RuntimeError, match='exhausted'):
        run_fake_handoff(latency=3.)


def test_continue_cli_dry_plan_one_resident_model_no_sudo(capsys):
    from scripts.profile_idle_payback import main
    with patch('sys.argv', ['probe', '--continue-control', '--dry-run']), patch('subprocess.run') as command:
        assert main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data['measured_requests'] == 20 and data['model_loads'] == 1
    assert data['refresh_observation_steps'] == [4, 20]
    command.assert_not_called()


def test_no_write_preflight_is_explicit_and_not_static_mode(capsys):
    from scripts.profile_idle_payback import main
    with patch('sys.argv', ['probe', '--continue-control', '--preflight', '--dry-run']):
        assert main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data['no_clock_writes'] and data['measured_requests'] == 4
    with patch('sys.argv', ['probe', '--preflight', '--dry-run']):
        with pytest.raises(SystemExit):
            main()
