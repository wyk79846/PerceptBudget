import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.device_clocks import recovery_trace, PROFILES
from perceptbudget.device_idle_payback import DELAYS, PairedIdleLease, plan, summarize


def test_paired_protocol_uses_command_origin_and_restores_service_profile():
    read, write = os.pipe()
    events = []
    current = ['baseline']
    def apply(profile):
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
        recovery_trace(SimpleNamespace(apply=apply, actual=actual), 'gpu_emc', .05,
                       emit, read, paired_idle=True)
    finally:
        os.close(read)
        os.close(write)
    rows = [e for e in events if e['status'] == 'SERVE']
    assert len(rows) == 5 and current[0] == 'baseline'
    for row in rows:
        assert row['requested_serve_monotonic'] == row['down']['command_monotonic']+.05
        assert row['up']['command_monotonic'] >= row['requested_serve_monotonic']


def test_pair_summary_preserves_negative_savings_and_readiness_penalty():
    rows = []
    for delay, profile in plan():
        for index in range(5):
            low = profile == 'gpu_emc'
            rows.append(dict(idle_seconds=delay, profile=profile, index=index,
                matched_cycle={'energy_j': 11 if low else 10},
                early_cycle={'energy_j': 8 if low else 9},
                actions_ready_from_origin_ms=520 if low else 500))
    for row in summarize(rows):
        assert row['mean_saved_j'] == -1 and row['positive_pairs'] == 0
        assert row['mean_ready_extra_ms'] == 20
        assert all(p['early_window_energy_saved_j'] == 1 for p in row['pairs'])


def test_root_command_is_bounded_and_old_whitelist_unchanged(tmp_path):
    lease = PairedIdleLease('gpu_emc', 2., tmp_path, 'pair')
    assert lease.command()[-5:] == ['paired-idle', '--profile', 'gpu_emc', '--idle-seconds', '2.0']
    with pytest.raises(ValueError):
        PairedIdleLease('cpu1190', .2, tmp_path, 'bad')
    with pytest.raises(ValueError):
        recovery_trace(None, 'gpu_emc', 2., None, None)


def test_dry_run_no_model_or_sudo(capsys):
    from scripts.profile_idle_payback import main
    with patch('sys.argv', ['probe', '--dry-run']), patch('subprocess.run') as command:
        assert main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data['measured_requests'] == 40 and data['model_loads'] == 1
    assert {p[0] for p in data['plan']} == set(DELAYS)
    command.assert_not_called()
