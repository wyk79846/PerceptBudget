"""M2 discovery never writes clock controls or hides missing permissions."""
from pathlib import Path
from unittest.mock import patch

from perceptbudget.device_control import inventory, read_node
from perceptbudget.device_profile import discovery, inspect_fast_interface, sensor_steps


def test_missing_and_nonwritable_interfaces_are_explicit(tmp_path):
    node = tmp_path / 'missing'
    report = read_node(node)
    assert report['value'] is None and report['error']
    assert report['writable'] is False
    assert not node.exists()


def test_inventory_keeps_separate_cpu_domains_and_no_hardware_writes(tmp_path):
    for policy, cpus in (('policy0', '0 1 2 3'), ('policy4', '4 5 6 7')):
        path = tmp_path / 'devices/system/cpu/cpufreq' / policy
        path.mkdir(parents=True)
        (path / 'related_cpus').write_text(cpus)
    with patch('perceptbudget.device_control.subprocess.run', side_effect=PermissionError('denied')):
        result = inventory(tmp_path)
    assert result['domains']['cpu_policy0']['related_cpus']['value'] == '0 1 2 3'
    assert result['domains']['cpu_policy4']['related_cpus']['value'] == '4 5 6 7'
    assert result['domains']['emc']['rate']['error']
    assert result['hardware_writes'] is False
    assert result['power_mode']['error'] == 'denied'


def test_current_source_interface_is_not_claimed_as_live_fast_validation():
    import json
    import pytest
    root = Path(__file__).resolve().parents[1]
    upstream = Path(json.loads((root / 'config.json').read_text())['flashrt_root'])
    if not upstream.is_dir():
        pytest.skip('Local pinned FlashRT source not installed')
    report = inspect_fast_interface(upstream)
    assert len(report['source']) == 2
    assert all(r['sha256'] and r['relevant_lines'] for r in report['source'])
    assert report['flashrt_modified'] is False
    assert report['upstream_num_views_interface'] is True
    assert 'no live Fast' in report['scope']


def test_sensor_rejects_bad_windows_before_cuda():
    import pytest
    with pytest.raises(ValueError):
        sensor_steps(block_seconds=.1)


def test_discovery_has_no_service_measurements():
    with patch('perceptbudget.device_profile.inventory', return_value={'hardware_writes': False}), \
         patch('perceptbudget.device_profile.inspect_fast_interface', return_value={'scope': 'source only'}):
        result = discovery({'flashrt_root': '/unused', 'baseline_id': 'frozen'})
    assert result['full_service_measurements_started'] is False
    assert result['clock_settings_changed'] is False
