"""Lower-idle probe uses bounded existing restoration without changing defaults."""
import sys
from unittest.mock import patch

import pytest

from scripts import probe_lower_idle as probe
from perceptbudget import device_clocks


def test_helper_only_extends_diagnostic_process(monkeypatch):
    original = dict(device_clocks.PROFILES)
    monkeypatch.setattr(device_clocks, 'PROFILES', dict(original))
    monkeypatch.setattr(device_clocks, 'MATRIX_PROFILES', device_clocks.MATRIX_PROFILES)
    monkeypatch.setattr(sys, 'argv', ['probe', '--clock-helper', 'lease',
                                    '--profile', 'idle408', '--seconds', '1'])
    with patch.object(probe.os, 'geteuid', return_value=0), \
         patch('scripts.control_device_clocks.main') as control:
        probe.clock_helper()
        control.assert_called_once_with()
    assert '--clock-helper' not in sys.argv
    assert device_clocks.PROFILES['idle408'] == (408000000, 1600000000, 1497600)
    assert all(device_clocks.PROFILES[k] == v for k,v in original.items())
    assert 'idle306' in device_clocks.MATRIX_PROFILES


def test_helper_refuses_unprivileged_execution():
    with patch.object(probe.os, 'geteuid', return_value=1001):
        with pytest.raises(RuntimeError, match='requires sudo'):
            probe.clock_helper()


def test_root_command_is_only_known_diagnostic_entry():
    command = probe.helper_command('lease', '--profile', 'idle306', '--seconds', '1')
    assert command[:3] == ['sudo','-n','/usr/bin/python3']
    assert command[4:] == ['--clock-helper','lease','--profile','idle306','--seconds','1']
