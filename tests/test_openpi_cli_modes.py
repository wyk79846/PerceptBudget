"""The old experimental CLI is retired; use the frozen baseline entry."""
import subprocess
import sys
from pathlib import Path

def test_internal_helpers_cannot_silently_run_as_old_cli():
    root=Path(__file__).resolve().parents[1]
    for script in ('eval_rtc_openpi.py', 'eval_q.py'):
        result=subprocess.run([sys.executable, 'scripts/'+script, '--horizon', '20'],
                              cwd=root, capture_output=True, text=True)
        assert result.returncode == 2
        assert 'run_baseline.py' in result.stderr


def test_h50_source_horizon_transfer_requires_explicit_opt_in():
    from types import SimpleNamespace
    from scripts.eval_rtc_openpi import horizon_transfer_allowed
    args=SimpleNamespace(actual_delay=True,experimental_varlen=False,horizon=50,
        allow_horizon_transfer=False,author_order_reference=True,source_step_prefixes=[566,567])
    assert not horizon_transfer_allowed(args,25)
    args.allow_horizon_transfer=True
    assert horizon_transfer_allowed(args,25)
    assert not horizon_transfer_allowed(args,20)
    args.horizon=20
    assert horizon_transfer_allowed(args,25)
    args.allow_horizon_transfer=False
    assert not horizon_transfer_allowed(args,25)
    args.allow_horizon_transfer=True
    args.actual_delay=False
    assert not horizon_transfer_allowed(args,25)
