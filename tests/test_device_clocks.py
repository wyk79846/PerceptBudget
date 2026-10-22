from pathlib import Path
from unittest.mock import patch

import pytest

from perceptbudget.device_clocks import ClockControl, PROFILES


@pytest.fixture
def control(tmp_path):
    data = {
        'class/devfreq/17000000.gpu/min_freq': '1173000000',
        'class/devfreq/17000000.gpu/max_freq': '1173000000',
        'class/devfreq/17000000.gpu/cur_freq': '1173000000',
        'class/devfreq/17000000.gpu/available_frequencies': '816000000 1020000000 1173000000',
        'kernel/debug/bpmp/debug/clk/emc/rate': '3199000000',
        'kernel/debug/bpmp/debug/clk/emc/min_rate': '204000000',
        'kernel/debug/bpmp/debug/clk/emc/max_rate': '3199000000',
        'kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked': '1',
        'kernel/debug/bpmp/debug/clk/emc/dvfs_table': 'vdd_core 2133000000 1\nvdd_core 2750000000 1\nvdd_core 3199000000 1',
    }
    for policy in ('policy0', 'policy4'):
        prefix = 'devices/system/cpu/cpufreq/'+policy+'/'
        for name in ('scaling_min_freq', 'scaling_max_freq', 'scaling_cur_freq'):
            data[prefix+name] = '1497600'
        data[prefix+'scaling_available_frequencies'] = '1190400 1497600'
    for name, value in data.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    instance = ClockControl(tmp_path)
    writes = []
    def write(path, value):
        assert path.exists()
        writes.append((path.name, value))
        path.write_text(str(value))  # regular files emulate sysfs replacement
    with patch.object(instance, 'write', side_effect=write):
        yield instance, writes


@pytest.mark.parametrize('profile', PROFILES)
def test_apply_and_restore_all_profiles(control, profile):
    c, writes = control
    saved = c.snapshot()
    c.apply(profile)
    gpu, emc, cpu = PROFILES[profile]
    assert c.read(c.gpu/'min_freq') == gpu
    assert c.read(c.emc/'rate') == emc
    assert all(c.read(p/'scaling_max_freq') == cpu for p in c.cpus)
    c.restore(saved)
    assert c.snapshot() == saved
    assert all(name not in ('governor', 'scaling_governor', 'state') for name, _ in writes)


def test_unsupported_emc_rejected_before_any_write(control):
    c, writes = control
    (c.emc/'dvfs_table').write_text('vdd_core 3199000000 1')
    with pytest.raises(ValueError):
        c.apply('emc2133')
    assert writes == []


def test_restore_rejects_arbitrary_nodes(control):
    c, writes = control
    saved = c.snapshot()
    saved['../../arbitrary'] = 1
    with pytest.raises(ValueError):
        c.restore(saved)
    assert writes == []


def test_restore_attempts_remaining_domains_on_failure(control):
    c, _ = control
    saved = c.snapshot()
    c.apply('cpu1190')
    real_write = c.write
    def fail_emc(path, value):
        if path == c.emc/'rate':
            raise OSError('injected EMC failure')
        real_write(path, value)
    with patch.object(c, 'write', side_effect=fail_emc), pytest.raises(RuntimeError):
        c.restore(saved)
    assert all(c.read(p/'scaling_min_freq') == 1497600 for p in c.cpus)


def test_cli_has_no_arbitrary_path_or_frequency_options():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['/usr/bin/python3', str(root/'scripts/control_device_clocks.py'),
                             'lease', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--seconds' in result.stdout and '--profile' in result.stdout
    assert '--sys-root' not in result.stdout and '--command' not in result.stdout
