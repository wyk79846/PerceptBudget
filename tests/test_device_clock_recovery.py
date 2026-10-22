import json
import os
import queue
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.device_clocks import recovery_trace, PROFILES
from perceptbudget.device_clock_recovery import RecoveryLease, PLAN, screen


@pytest.mark.parametrize('serve_profile', ['baseline', 'balanced', 'gpu_emc'])
def test_fixed_root_handshake_runs_five_requests_without_arbitrary_commands(serve_profile):
    read, write = os.pipe()
    events, calls = [], []
    current = [serve_profile]
    def apply(profile):
        calls.append(profile)
        current[0] = profile
    def actual():
        gpu, emc, cpu = PROFILES[current[0]]
        return {'gpu/cur_freq': gpu, 'emc/rate': emc, 'cpu/scaling_cur_freq': cpu}
    def emit(event):
        events.append(event)
        if event['status'] == 'SERVE':
            os.write(write, b'DONE\n')
        if event['status'] == 'REQUEST_ACK' and event['request_index'] < 4:
            os.write(write, b'RUN\n')
    try:
        os.write(write, b'RUN\n')
        recovery_trace(SimpleNamespace(apply=apply, actual=actual), 'gpu_emc', 0., emit, read,
                       serve_profile=serve_profile)
    finally:
        os.close(read)
        os.close(write)
    assert calls == ([] if serve_profile == 'gpu_emc' else ['gpu_emc', serve_profile]*5)
    assert len([e for e in events if e['status'] == 'SERVE']) == 5
    assert current[0] == serve_profile


def test_root_protocol_rejects_unexpected_tokens():
    read, write = os.pipe()
    try:
        os.write(write, b'EXEC\n')
        with pytest.raises(ValueError, match='protocol'):
            recovery_trace(SimpleNamespace(actual=lambda: {}), 'baseline', 0., lambda e: None, read)
    finally:
        os.close(read)
        os.close(write)


def test_recovery_pipe_eof_causes_failure_not_infinite_wait():
    read, write = os.pipe()
    os.close(write)
    try:
        with pytest.raises(RuntimeError, match='closed'):
            recovery_trace(SimpleNamespace(actual=lambda: {}), 'baseline', 0., lambda e: None, read)
    finally:
        os.close(read)


def test_recovery_lease_checks_response_index(tmp_path):
    lease = RecoveryLease('gpu_emc', .2, tmp_path, 'test')
    assert lease.command()[-4:] == ['--profile', 'gpu_emc', '--idle-seconds', '0.2']
    lease.lines.put(dict(status='SERVE', request_index=2))
    with pytest.raises(RuntimeError, match='index'):
        lease.receive('SERVE', 1)
    with pytest.raises(ValueError):
        lease.send('shell')
    with pytest.raises(ValueError):
        RecoveryLease('cpu1190', 1, tmp_path, 'test')


def test_recovery_dry_plan_loads_once_and_runs_thirty_requests(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--recovery', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as run:
        assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['total_requests'] == 30 and result['model_loads'] == 1
    assert result['N'] == 5 and result['live_observation'] is False
    run.assert_not_called()


@pytest.mark.parametrize('matrix', [False, True])
def test_resident_screen_preserves_seed_pairing_and_records_whole_recovery(tmp_path, matrix):
    import numpy as np
    from perceptbudget.power import PowerReport
    calls, seeds = [], []
    def predict(*args, **kwargs):
        calls.append(kwargs)
        return np.zeros((50, 32), dtype=np.float32)
    class FakeLease:
        events = []
        def __init__(self, *args, **kwargs):
            self.serve_profile = kwargs.get('serve_profile', 'baseline')
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def send(self, token):
            assert token in ('RUN', 'DONE')
        def receive(self, status, index):
            now = time.monotonic()
            return dict(request_index=index, serve_profile=self.serve_profile,
                        down={'command_monotonic': now-.003,
                        'observed_match_monotonic': now-.002},
                        up={'command_monotonic': now-.001, 'observed_match_monotonic': now-.0005})
    class Monitor:
        readings = []
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def sample(self):
            pass
        def integrate_report(self, a, b):
            return PowerReport(1000, b-a, 1, 0, 1000)
    with patch('perceptbudget.device_clock_recovery.RecoveryLease', FakeLease), \
         patch('perceptbudget.device_clock_recovery.PowerMonitor', Monitor), \
         patch('perceptbudget.device_clock_recovery.frequency_check', return_value={'matches': True}), \
         patch('perceptbudget.device_clock_recovery.inventory', return_value={'thermal': []}):
        result = screen(SimpleNamespace(predict=predict), {}, 'test', np.zeros((45, 32)),
                        SimpleNamespace(manual_seed=seeds.append), tmp_path, {'model_loads': 1}, matrix=matrix)
    groups = 9 if matrix else len(PLAN)
    count = groups*5
    assert result['status'] == 'completed' and len(result['conditions']) == groups
    assert len(calls) == len(seeds) == count
    assert all(seeds[i:i+5] == seeds[:5] for i in range(0, count, 5))
    assert all(c['restored'] and c['max_abs_action_difference_vs_hot_pre'] == 0 for c in result['conditions'])
    assert all(r['recovery_compute_ms'] >= r['predict_ms'] for c in result['conditions'] for r in c['requests'])
    assert len(list(tmp_path.glob('*_actions.npz'))) == groups
    if matrix:
        assert len((tmp_path/'full_cost_table.csv').read_text().splitlines()) == 10


def test_full_matrix_covers_three_service_profiles_and_three_idle_states(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--matrix', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as run:
        assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['total_requests'] == 45 and result['full_cells'] == 9
    pairs = {(item[2], item[3]) for item in result['plan']}
    assert len(pairs) == 9
    assert {profile for _, profile in pairs} == {'baseline', 'balanced', 'gpu_emc'}
    assert all(item[1] == 'gpu_emc' for item in result['plan'] if item[2] > 0)
    run.assert_not_called()


def test_matrix_root_command_whitelists_service_profile(tmp_path):
    lease = RecoveryLease('gpu_emc', .2, tmp_path, 'test', serve_profile='balanced')
    assert lease.command()[-2:] == ['--serve-profile', 'balanced']
    with pytest.raises(ValueError):
        RecoveryLease('gpu_emc', .2, tmp_path, 'test', serve_profile='cpu1190')


def test_matrix_preflight_stops_before_any_model_on_clock_failure(tmp_path):
    from scripts.profile_clock_screen import clock_preflight
    summary = {}
    def run(command, **kwargs):
        kwargs['stdout'].write('RESTORED\n')
        return SimpleNamespace(returncode=1 if 'balanced' in command else 0)
    with patch('scripts.profile_clock_screen.subprocess.run', side_effect=run) as call:
        with pytest.raises(RuntimeError, match='NO MODEL LOADED'):
            clock_preflight(tmp_path, ('baseline', 'balanced', 'gpu_emc'), summary)
    assert call.call_count == 2
    assert summary['clock_preflight'][-1]['restored'] is True
    assert not (tmp_path/'gpu_emc_preflight.log').exists()
