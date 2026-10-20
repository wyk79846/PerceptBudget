"""CPU preparation tests; no model, simulator, power hardware, or GPU loaded."""
from concurrent.futures import Future
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from perceptbudget.power import PowerReading
from scripts import probe_m1_h50_power as probe


def reference(tmp_path):
    cfg = dict(H=50, N=5, precision='bfloat16', state_enabled=True,
        checkpoint_profile=probe.PROFILE, model_id=probe.PROFILE,
        state_prompt_capacity=200, checkpoint_provenance_revision=probe.REVISION,
        rtc_exact_flash_attention=True, rtc_fused_norm=True, native_gemm_policy='autotune',
        rtc_beta=5, diagnostic_condition='actual_history', execution_mode='diagnostic_actual_history',
        rtc_author_order=True, rtc_expert='original_openpi_source', rtc_action_state_dtype='float32',
        rtc_paper_schedule=True, release_unused_native_expert=True,
        source_step_prefixes=[665], screen=dict(task_ids=[0], seed=7, init_state_index=0), rtc_suite='libero_spatial')
    summary = dict(status='completed', diagnostic_condition='actual_history', episode=dict(status='timeout'))
    (tmp_path/'run.json').write_text(json.dumps(dict(config_snapshot=cfg)))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    return cfg


def test_reference_uses_worker_config_without_global_defaults(tmp_path):
    cfg = reference(tmp_path)
    config = probe.reference_config(tmp_path)
    assert config['H'] == 50 and config['N'] == 5
    assert config['native_gemm_policy'] == 'autotune'
    assert config['execution_mode'] == 'm1_h50_power_calibration'
    assert json.loads((tmp_path/'run.json').read_text())['config_snapshot'] == cfg


@pytest.mark.parametrize('key,value', [
    ('H', 20), ('N', 10), ('state_enabled', False), ('precision', 'float16'),
    ('checkpoint_profile', 'old'), ('checkpoint_provenance_revision', 'wrong'),
    ('rtc_exact_flash_attention', False), ('rtc_fused_norm', False),
    ('native_gemm_policy', 'heuristic-first'), ('rtc_beta', 2),
    ('diagnostic_condition', 'sync_no_rtc'), ('rtc_fixed_delay_estimate', 5),
    ('rtc_suffix_only_vjp', True), ('release_unused_native_expert', False),
    ('rtc_author_order', False), ('rtc_action_state_dtype', 'bfloat16'),
    ('source_step_prefixes', []),
])
def test_reference_rejects_wrong_conditions(tmp_path, key, value):
    cfg = reference(tmp_path)
    cfg[key] = value
    (tmp_path/'run.json').write_text(json.dumps(dict(config_snapshot=cfg)))
    with pytest.raises(ValueError):
        probe.reference_config(tmp_path)


def test_invalid_worker_rejected(tmp_path):
    reference(tmp_path)
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='invalid')))
    with pytest.raises(ValueError):
        probe.reference_config(tmp_path)


def test_full_guided_predict_with_fixed_old_target():
    previous = np.zeros((45, 32), dtype=np.float32)
    snapshot = dict(state=np.zeros(32))
    def predict(snap, prompt, old, *, delay, executed, request):
        assert snap is snapshot and old is previous
        assert prompt == 'task' and delay == executed == 5
        request.update(actions_raw=np.zeros((50, 32)).tolist(),
            inference_started_monotonic=1., inference_ended_monotonic=2.,
            rtc_guidance=dict(guidance_enabled=True, executed_steps=5, delay_estimate_steps=5,
                variable_length_fallback=False, fused_norm=True,
                attention_backend='experimental_flash_exact', vjp_norms=[1.]*5))
    adapter = SimpleNamespace(predict=Mock(side_effect=predict))
    result = probe.guided_request(adapter, snapshot, 'task', previous, 1)
    adapter.predict.assert_called_once()
    assert 'actions_raw' not in result
    assert result['rtc_guidance']['guidance_enabled']


@pytest.mark.parametrize('key,value', [('guidance_enabled', False),
    ('variable_length_fallback', True), ('fused_norm', False), ('vjp_norms', [float('nan')])])
def test_bad_guidance_fails(key, value):
    def predict(*args, request, **kwargs):
        g = dict(guidance_enabled=True, executed_steps=5, delay_estimate_steps=5,
                 variable_length_fallback=False, fused_norm=True,
                 attention_backend='experimental_flash_exact', vjp_norms=[1.]*5)
        g[key] = value
        request.update(rtc_guidance=g, actions_raw=np.zeros((50, 32)).tolist())
    with pytest.raises(RuntimeError):
        probe.guided_request(SimpleNamespace(predict=predict), {}, 'task', np.zeros((45, 32)), 0)


def test_idle_has_control_and_no_inference():
    clock = SimpleNamespace(now=0.)
    def tick():
        clock.now += .5
    executor = Mock()
    block = probe.measure_block('idle', 0, executor, Mock(), tick, clock=lambda: clock.now)
    assert block['env_steps'] == 16 and block['stop_monotonic'] == 8.
    assert block['drain_seconds'] == 0
    executor.submit.assert_not_called()


def test_busy_drains_before_next_idle_and_keeps_control_running():
    clock = SimpleNamespace(now=0.)
    pending = Future()
    def tick():
        clock.now += 1.
        if clock.now == 10.:
            pending.set_result(None)
    executor = Mock()
    executor.submit.return_value = pending
    block = probe.measure_block('busy', 1, executor, Mock(), tick, clock=lambda: clock.now)
    assert block['stop_monotonic'] == 8.
    assert block['drain_start_monotonic'] == 8.
    assert block['drain_stop_monotonic'] == 10.
    assert block['drain_seconds'] == 2.
    assert block['drain_env_steps'] == 2
    executor.submit.assert_called_once()
    idle = probe.measure_block('idle', 2, executor, Mock(), tick, clock=lambda: clock.now)
    assert idle['start_monotonic'] == 10.


def test_busy_repeats_without_overlapping_requests():
    clock = SimpleNamespace(now=0.)
    pending = Future()
    stop = Mock()
    stop.is_set.side_effect = [False, False, False, True]
    predict = Mock()
    executor = Mock()
    def submit(fn):
        fn()
        return pending
    executor.submit.side_effect = submit
    def tick():
        clock.now += 1.
        if clock.now == 8.:
            pending.set_result(None)
    with patch.object(probe.threading, 'Event', return_value=stop):
        block = probe.measure_block('busy', 1, executor, predict, tick, clock=lambda: clock.now)
    assert executor.submit.call_count == 1
    assert predict.call_count == 3
    stop.set.assert_called_once()
    assert block['stop_monotonic'] == 8. and block['drain_seconds'] == 0.


def test_inference_exception_propagates():
    future = Future()
    future.set_exception(RuntimeError('gate failed'))
    executor = Mock()
    executor.submit.return_value = future
    clock = SimpleNamespace(now=0.)
    def tick():
        clock.now += 1.
    with pytest.raises(RuntimeError, match='gate failed'):
        probe.measure_block('busy', 1, executor, Mock(), tick, clock=lambda: clock.now)


def test_reused_schedule_and_power_summary():
    assert probe.block_schedule() == ['idle', 'busy']*3+['idle']
    readings = [PowerReading(i/10, 10000.) for i in range(181)]
    result = probe.measurement_summary(readings, [
        dict(mode='busy', start_monotonic=0., stop_monotonic=8., drain_stop_monotonic=10.),
        dict(mode='idle', start_monotonic=10., stop_monotonic=18.)])
    assert result['P_on_w'] == pytest.approx(10.)
    assert result['P_idle_w'] == pytest.approx(10.)
    assert result['grouped']['busy']['duration_seconds'] == pytest.approx(6.5)


def test_cli_requires_reference_and_wraps_standard_guard(tmp_path):
    with pytest.raises(SystemExit):
        probe.main([])
    reference(tmp_path)
    with patch.object(probe.subprocess, 'call', return_value=0) as call:
        assert probe.main(['--reference-run', str(tmp_path)]) == 0
    argv = call.call_args.args[0]
    assert argv[1].endswith('/scripts/guarded_run.py')
    assert '--guarded-worker' in argv
