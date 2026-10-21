"""Configuration, startup and import checks for the single retained baseline."""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from perceptbudget import baseline
from scripts import diagnose_rtc_handoff as diag

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'config.json').read_text())


def test_current_config_and_command_are_the_measured_baseline():
    c = config()
    baseline.validate(c)
    cmd = baseline.command(c, ROOT, suite='libero_goal', task_id=9, q=40)
    with patch('sys.argv', ['diagnostic', *cmd[3:], '--worker', 'actual_history',
                            '--parent', '/tmp/no-gpu-baseline-check', '--task-id', '9']), \
         patch.object(diag, 'worker', return_value=0) as worker:
        assert diag.main() == 0
    worker.assert_called_once_with(Path(c['baseline_reference_run']), 'actual_history',
        Path('/tmp/no-gpu-baseline-check'), 9, 5., 40, 50, 'h50',
        exact_flash=True, fused_norm=True, gemm_policy='autotune', suite='libero_goal',
        calibrate_control_period=True)


def test_old_reference_provides_evidence_not_runtime_defaults():
    old = {'H': 20, 'N': 10, 'state_enabled': False,
           'model_path': '/old/model.safetensors', 'proof': 'keep',
           'screen': {'task_ids': [2]}}
    result = baseline.apply_to_reference(old, config())
    assert result['H'] == 50 and result['N'] == 5
    assert result['state_enabled'] is True
    assert result['model_path'] == config()['model_path']
    assert result['proof'] == 'keep'
    assert old['H'] == 20


@pytest.mark.parametrize('key,value', [
    ('H', 20), ('N', 10), ('state_enabled', False), ('rtc_exact_flash_attention', False),
    ('rtc_fused_norm', False), ('native_gemm_policy', 'heuristic-first'),
    ('action_handoff', 'latency_limited_progress'), ('rtc_fixed_delay_estimate', 5),
])
def test_old_or_alternative_baselines_rejected(key, value):
    c = copy.deepcopy(config())
    c[key] = value
    with pytest.raises(ValueError):
        baseline.validate(c)


def test_all_local_python_imports_resolve_after_cleanup():
    for folder in ('perceptbudget', 'scripts', 'tests'):
        for path in (ROOT / folder).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith(('scripts.', 'perceptbudget.')):
                        assert (ROOT / (node.module.replace('.', '/') + '.py')).is_file(), (path, node.module)
                    elif node.module in ('scripts', 'perceptbudget'):
                        for name in node.names:
                            assert (ROOT / node.module / (name.name + '.py')).is_file(), (path, name.name)


def test_diagnostic_cannot_reenable_old_model_or_solver():
    for flags in (['--horizon', '20'], ['--checkpoint-profile', 'old'],
                  ['--suffix-only-vjp'], ['--gemm-policy', 'heuristic-first']):
        with patch('sys.argv', ['diagnostic', '--reference-run', '/tmp/reference', *flags]), \
             pytest.raises(SystemExit) as err:
            diag.main()
        assert err.value.code == 2
