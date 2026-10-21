"""Retired experiments must not remain dependencies of RTC."""
import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_has_no_retired_module_imports():
    retired = {'perceptbudget.prefix_model', 'perceptbudget.prefix_timing',
               'perceptbudget.prefix_fusion', 'perceptbudget.seam_rollout',
               'perceptbudget.seam_guidance', 'scripts.eval_seam_sync',
               'scripts.eval_seam_async'}
    for folder in ('scripts', 'perceptbudget'):
        for path in (ROOT / folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert node.module not in retired, path
                    assert all(f'{node.module}.{a.name}' not in retired
                               for a in node.names), path
                elif isinstance(node, ast.Import):
                    assert all(a.name not in retired for a in node.names), path


def test_task_loader_and_rtc_loop_independent_of_retired_adapters():
    from scripts import libero_tasks, probe_delay_buffer as probe
    assert callable(libero_tasks.get_tasks)
    assert callable(libero_tasks.make_env)
    assert libero_tasks.MAX_STEPS_BY_SUITE['libero_10'] == 520
    assert 'prefix_adapter' not in inspect.signature(probe.run_episode).parameters
    assert 'seam_adapter' not in inspect.signature(probe.run_episode).parameters
    for name in ('advance_prefix_due', 'prefix_hold_due', 'commit_prefix_hold'):
        assert not hasattr(probe, name)
