import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.device_clock_screen import Lease, energy_window
from perceptbudget.power import PowerReport


def test_energy_window_requires_full_coverage():
    partial = PowerReport(1000, .5, .5, 0, None)
    out = energy_window(SimpleNamespace(integrate_report=lambda a, b: partial), 1, 2)
    assert out['energy_j'] is None and out['mean_power_w'] is None
    complete = PowerReport(10000, 1, 1, 0, 10000)
    out = energy_window(SimpleNamespace(integrate_report=lambda a, b: complete), 1, 2)
    assert out['energy_j'] == out['mean_power_w'] == 10


def test_lease_rejects_non_whitelisted_profiles(tmp_path):
    with pytest.raises(ValueError):
        Lease('arbitrary', tmp_path)


def test_actual_frequency_mismatch_not_hidden():
    from perceptbudget.device_clock_screen import frequency_check
    event = dict(status='SAMPLE', monotonic=2, actual={
        'gpu/cur_freq': 816000000, 'emc/rate': 3199000000,
        'policy0/scaling_cur_freq': 1497600})
    assert frequency_check([event], 'gpu816', 1, 3)['matches']
    assert not frequency_check([event], 'baseline', 1, 3)['matches']
    assert not frequency_check([event], 'gpu816', 3, 4)['matches']


def test_lease_requires_successful_restore(tmp_path):
    lease = Lease('gpu816', tmp_path)
    lease.proc = SimpleNamespace(wait=lambda **kwargs: 0)
    lease.thread = SimpleNamespace(join=lambda **kwargs: None)
    with pytest.raises(RuntimeError, match='restoration'):
        lease.finish()
    lease.events.append({'message': 'RESTORED'})
    lease.finish()
    lease.proc = SimpleNamespace(wait=lambda **kwargs: 1)
    with pytest.raises(RuntimeError):
        lease.finish()


def test_root_helper_has_fixed_args_and_no_model_command(tmp_path):
    import io
    from perceptbudget.device_clock_screen import LEASE_SECONDS
    ready = json.dumps(dict(status='READY', monotonic=10))+'\nRESTORED\n'
    child = SimpleNamespace(stdout=io.StringIO(ready), wait=lambda **kw: 0)
    with patch('perceptbudget.device_clock_screen.subprocess.Popen', return_value=child) as spawn:
        with Lease('baseline', tmp_path) as lease:
            assert lease.deadline == 10+LEASE_SECONDS
    command = spawn.call_args.args[0]
    assert command[:3] == ['sudo', '-n', '/usr/bin/python3']
    assert command[-1] == '--trace'
    assert '--command' not in command and '--run-dir' not in command
    assert (tmp_path/'baseline_clocks.jsonl').exists()


def test_lease_refuses_measurement_beyond_deadline(tmp_path):
    lease = Lease('baseline', tmp_path)
    lease.proc = SimpleNamespace(poll=lambda: None)
    lease.deadline = 5
    with patch('perceptbudget.device_clock_screen.time.monotonic', return_value=4):
        lease.ensure()
        with pytest.raises(RuntimeError):
            lease.ensure(2)


def test_memory_monitor_rejects_before_spawn():
    from scripts.profile_clock_screen import monitored_worker
    with patch('scripts.guarded_run.available_ram_bytes', return_value=0), \
         patch('scripts.profile_clock_screen.subprocess.Popen') as spawn:
        with pytest.raises(RuntimeError, match='768MiB'):
            monitored_worker(['unused'], None)
        spawn.assert_not_called()


def test_dry_plan_has_one_model_twenty_requests_without_sudo(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as launch:
        assert main() == 0
    record = json.loads(capsys.readouterr().out)
    assert record['model_loads'] == 1 and record['requests_per_profile'] == 5
    assert len(record['profiles']) == 4 and record['N'] == 5
    launch.assert_not_called()


def test_complete_screen_is_four_groups_of_five_with_same_seed_list(tmp_path):
    import numpy as np
    from perceptbudget.device_clock_screen import screen, SCREEN_PROFILES
    calls = []
    seeds = []
    def predict(*args, **kwargs):
        calls.append(kwargs)
        return np.zeros((50, 32), dtype=np.float32)
    class FakeLease:
        def __init__(self, profile, directory):
            self.ready = {'profile': profile}
            self.events = []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def ensure(self, *args):
            pass
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
        def integrate_report(self, start, stop):
            return PowerReport(1, stop-start, 1, 0, 1)
    with patch('perceptbudget.device_clock_screen.Lease', FakeLease), \
         patch('perceptbudget.device_clock_screen.PowerMonitor', Monitor), \
         patch('perceptbudget.device_clock_screen.inventory', return_value={'thermal': []}), \
         patch('perceptbudget.device_clock_screen.time.sleep'):
        result = screen(SimpleNamespace(predict=predict), {}, 'test', np.zeros((45, 32)),
                        SimpleNamespace(manual_seed=seeds.append), tmp_path, {'model_loads': 1})
    assert result['status'] == 'completed'
    assert [r['profile'] for r in result['conditions']] == list(SCREEN_PROFILES)
    assert len(calls) == 4*7 and len(seeds) == 20
    assert seeds[:5] == seeds[5:10] == seeds[10:15] == seeds[15:]
    assert all(r['restored'] and len(r['requests']) == 5 for r in result['conditions'])
    assert len(list(tmp_path.glob('*_actions.npz'))) == 4
