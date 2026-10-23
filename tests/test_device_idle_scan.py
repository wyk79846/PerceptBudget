import json
from unittest.mock import patch
import pytest
from perceptbudget.device_idle_scan import scan_plan,describe,DELAYS


def test_scan_balances_durations_per_block_and_reproduces_randomization():
    plan=scan_plan()
    assert plan==scan_plan() and len(plan)==21
    for block in range(3):
        values=[d for b,d in plan if b==block]
        assert sorted(values)==list(DELAYS)
    assert [d for b,d in plan if b==0]!=[d for b,d in plan if b==1]
    with pytest.raises(ValueError):scan_plan(2)


def test_summary_uses_actual_idle_and_preserves_unresolved_short_power():
    rows=[dict(requested_idle_seconds=0.,actual_idle_seconds=.006,recovery_ms=490.,idle_tail_mean_power_w=None),
          dict(requested_idle_seconds=0.,actual_idle_seconds=.008,recovery_ms=510.,idle_tail_mean_power_w=None)]
    summary=describe(rows)[0]
    assert summary['actual_idle_median_ms']==7.
    assert summary['recovery_median_ms']==500.
    assert summary['idle_tail_mean_power_w'] is None
    assert 'no state boundary' in summary['interpretation']


def test_scan_cli_never_requires_sudo_for_dry_plan(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv',['probe','--idle-state-scan','--dry-run']),patch('scripts.profile_clock_screen.subprocess.run') as call:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['state_classification'] is None
    assert not plan['frequency_switch_during_scan'] and plan['model_loads']==1
    assert plan['recovery_requests']==21 and plan['preparation_requests']==21
    call.assert_not_called()


def test_missing_power_remains_fatal_for_calibration_but_not_gpu_timing_only():
    from perceptbudget.device_idle_scan import validate_recovery_power
    window={'energy_j':None,'power_coverage':{'coverage_fraction':.4}}
    with pytest.raises(RuntimeError,match='coverage incomplete'):
        validate_recovery_power(window)
    assert validate_recovery_power(window,startup_timing=True) is False
    assert window['energy_j'] is None and window['power_coverage']['coverage_fraction']==.4
    assert validate_recovery_power({'energy_j':1.},startup_timing=True)
