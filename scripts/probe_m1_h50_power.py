"""Prepared M1 power probe. Run only after the main 300 episodes finish.

CLI automatically wraps the worker in guarded_run.py (768 MiB RAM floor).
No power mode, clocks, or frozen runtime files are changed.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.power_windows import block_schedule, measurement_summary
from scripts import eval_q as base

PROFILE = 'lerobot_h50_quantiles_v044'
REVISION = 'b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd'


def reference_config(reference):
    config = json.loads((reference/'run.json').read_text())['config_snapshot']
    result = json.loads((reference/'summary.json').read_text())
    expected = dict(H=50, N=5, precision='bfloat16', state_enabled=True,
        checkpoint_profile=PROFILE, model_id=PROFILE, state_prompt_capacity=200,
        checkpoint_provenance_revision=REVISION, rtc_exact_flash_attention=True,
        rtc_fused_norm=True, native_gemm_policy='autotune', rtc_beta=5,
        rtc_author_order=True, rtc_expert='original_openpi_source', rtc_action_state_dtype='float32',
        diagnostic_condition='actual_history', execution_mode='diagnostic_actual_history',
        rtc_paper_schedule=True, release_unused_native_expert=True)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('reference config mismatch: '+key)
    for key in ('rtc_suffix_only_vjp', 'experimental_original_varlen', 'reuse_control_forward', 'original_cuda_graph'):
        if config.get(key):
            raise ValueError('unsupported reference backend: '+key)
    if config.get('controlled_delay_steps') is not None or config.get('rtc_fixed_delay_estimate') is not None:
        raise ValueError('reference must be real history-based async')
    if (result.get('status') != 'completed' or
            result.get('diagnostic_condition') != 'actual_history' or
            result.get('episode', {}).get('status') not in ('success', 'timeout', 'failure')):
        raise ValueError('reference must be a completed valid asynchronous worker')
    if len(config['screen']['task_ids']) != 1:
        raise ValueError('reference must identify one task')
    if config['screen'].get('seed') != 7 or config['screen'].get('init_state_index') != 0:
        raise ValueError('reference initial conditions mismatch')
    if not config.get('source_step_prefixes'):
        raise ValueError('reference requires source step bank')
    config = copy.deepcopy(config)
    config.update(execution_mode='m1_h50_power_calibration', rtc_audit_inputs=False,
                  rtc_audit_attention_same_input=False, rtc_audit_fused_norm=False)
    return config


def guided_request(adapter, snapshot, prompt, previous, index):
    import numpy as np
    request = dict(block_index=index, submitted_monotonic=time.monotonic())
    adapter.predict(snapshot, prompt, previous, delay=5, executed=5, request=request)
    request['ready_monotonic'] = time.monotonic()
    g = request['rtc_guidance']
    if (g.get('guidance_enabled') is not True or g['executed_steps'] != 5
            or g['delay_estimate_steps'] != 5 or g.get('variable_length_fallback')
            or not g.get('fused_norm') or g.get('attention_backend') != 'experimental_flash_exact'
            or np.asarray(request['actions_raw']).shape != (50, 32)
            or len(g['vjp_norms']) != 5
            or not np.isfinite(request['actions_raw']).all()
            or not np.isfinite(g['vjp_norms']).all()):
        raise RuntimeError('guided request backend/finite-output gate failed')
    # Keep request timing/solver metadata, without duplicating large action arrays.
    request.pop('actions_raw')
    return request


def measure_block(mode, index, executor, predict, tick, *, clock=time.monotonic, seconds=8.):
    block = dict(mode=mode, index=index, start_monotonic=clock(), env_steps=0, drain_env_steps=0)
    future = None
    stop = threading.Event()
    def busy_loop():
        while not stop.is_set():
            predict(index)
    try:
        if mode == 'busy':
            future = executor.submit(busy_loop)
        while clock()-block['start_monotonic'] < seconds:
            if future is not None and future.done():
                future.result()
                raise RuntimeError('continuous inference stopped unexpectedly')
            tick()
            block['env_steps'] += 1
    finally:
        block['stop_monotonic'] = clock()
        stop.set()
        # No new inference after this boundary; render/control continue during drain.
        try:
            while future is not None and not future.done():
                tick()
                block['drain_env_steps'] += 1
            if future is not None:
                future.result()
        finally:
            block['drain_stop_monotonic'] = clock()
    block['drain_start_monotonic'] = block['stop_monotonic']
    block['drain_seconds'] = block['drain_stop_monotonic']-block['stop_monotonic']
    block['control_hz'] = block['env_steps']/(block['stop_monotonic']-block['start_monotonic'])
    return block


def run(root, config):
    import numpy as np
    from dataclasses import asdict
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.power import PowerMonitor
    from perceptbudget.rtc_memory import memory_record, release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from scripts.probe_delay_buffer import device_state
    env = adapter = monitor = None
    summary = dict(status='loading', blocks=[], requests=[], config=config,
        device_before=device_state(), power_rail='INA3221 VDD_IN module input',
        scope='fixed snapshot; full context plus guided RTC VJP; resident model/env; zero seven-dimensional control',
        warmup_excluded=True, requested_block_seconds=8., poll_interval_seconds=.02,
        delay_steps=5, executed_steps=5, target_shape=[45, 32])
    def save():
        base._write_json_atomic(root/'summary.json', summary)
    def mem(stage):
        with (root/'memory.jsonl').open('a') as stream:
            stream.write(json.dumps(memory_record(torch, stage))+'\n')
    save()
    try:
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8')
        base._patch_egl_cleanup()
        torch, summary['cuda'] = base._check_cuda_and_native(config)
        from flash_rt.subgraphs.pi05.context_action import enable
        torch.use_deterministic_algorithms(True)
        base._seed_process(7, np, torch)
        task = base._load_tasks(config['screen']['task_ids'], [config['rtc_suite']])[0]
        started = time.monotonic()
        model = base._load_and_check_model(config, torch)
        summary['load_seconds'] = time.monotonic()-started
        enable(model)
        mem('model_loaded')
        _, cls = base._libero_api()
        env = base._make_env(cls, dict(task, max_steps=10000))
        env._pb_lock_fixtures = True
        obs, _ = base._prepare_env_episode(env, task, np)
        stats = load_libero_norm_stats(config['norm_stats_path'])
        snapshot = base._make_snapshot(obs, stats, True)
        base._infer_actions(model, snapshot, task['instruction'], 50, {})
        summary['native_expert_release'] = release_unused_native_expert(model, torch)
        adapter = OriginalOpenPiRTCAdapter(model, torch, config)
        gate = json.loads((Path(config['verified_reference_run'])/'summary.json').read_text())
        if not gate.get('passed') or not gate.get('weights_frozen') or not gate.get('author_order') or adapter.expert.provenance != gate['original_source_hashes']:
            raise RuntimeError('frozen author source gate mismatch')
        initial = {}
        adapter.bootstrap(snapshot, task['instruction'], initial)
        previous = np.asarray(initial['actions_raw'], dtype=np.float32)[5:].copy()
        if previous.shape != (45, 32) or not np.isfinite(previous).all():
            raise RuntimeError('invalid bootstrap target')
        valid = initial['rtc_guidance']['valid_prefix_length']
        # Fixed snapshot needs exactly its effective prefix length, no padded KV.
        summary['graph_audit'] = adapter.prepare_step_bank([valid])
        summary['effective_step_bank_lengths'] = [valid]
        if not summary['graph_audit'].get('passed'):
            raise RuntimeError('step bank gate failed')
        summary['warm_request'] = guided_request(adapter, snapshot, task['instruction'], previous, -1)
        summary.update(load_and_warm_seconds=time.monotonic()-started,
                       valid_prefix_length=valid, original_source_hashes=adapter.expert.provenance)
        mem('ready_for_power_windows')
        np.savez_compressed(root/'fixed_inputs.npz', previous_raw=previous,
                            model_state=snapshot['state'], bootstrap_raw=np.asarray(initial['actions_raw']))
        action = np.zeros(7, dtype=np.float32)
        def tick():
            start = time.monotonic()
            env.step(action)  # LIBERO observation rendering is part of every control step.
            time.sleep(max(0., .05-(time.monotonic()-start)))
        def predict(index):
            request = guided_request(adapter, snapshot, task['instruction'], previous, index)
            summary['requests'].append(request)
        monitor = PowerMonitor(poll_interval=.02)
        monitor.start()
        summary['status'] = 'running'
        with ThreadPoolExecutor(max_workers=1) as executor:
            for index, mode in enumerate(block_schedule()):
                block = measure_block(mode, index, executor, predict, tick)
                monitor.sample()  # Bracket drain end for integration before persisting.
                block['device'] = device_state()
                block['drain_power'] = asdict(monitor.integrate_report(
                    block['drain_start_monotonic'], block['drain_stop_monotonic']))
                summary['blocks'].append(block)
                save()
        monitor.stop()
        summary.update(measurement_summary(monitor.readings, summary['blocks']),
                       status='completed', device_after=device_state(),
                       limitations=['Fixed scene is not task success or asynchronous rollout power.',
                           '20 ms polling does not establish sensor update frequency.',
                           'Middle-window means exclude transitions; drains are separate from idle.',
                           'No device power mode or frequency changes.'])
    except BaseException as exc:
        summary.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if monitor is not None:
            monitor.stop()
            summary['raw_readings'] = [dict(timestamp_monotonic=r.timestamp, power_mw=r.power_mw,
                                            error=r.error) for r in monitor.readings]
        save()
        if adapter is not None:
            adapter.release_graph()
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True,
                        help='completed async worker from this M1 batch; not a parent run')
    parser.add_argument('--guarded-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = reference_config(args.reference_run)
    if not args.guarded_worker:
        print('Use only after main 300 episodes finish; launching standard 768 MiB guard. No power changes.', flush=True)
        return subprocess.call([sys.executable, str(ROOT/'scripts/guarded_run.py'), '--',
            sys.executable, str(Path(__file__).resolve()), '--reference-run', str(args.reference_run.resolve()), '--guarded-worker'])
    parent = Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    if not any(Path(os.fsdecode(arg)).name == 'guarded_run.py' for arg in parent if arg):
        parser.error('worker requires standard guarded_run.py parent (768 MiB)')
    from scripts.run_m1_h50 import provenance
    from perceptbudget.experiment import create_run
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = create_run(config, 'M1', 'h50-power', repo_root=ROOT)
        record = provenance(args.reference_run)
        import hashlib
        record['source_hashes']['scripts/probe_m1_h50_power.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        record['source_hashes']['perceptbudget/power_windows.py'] = hashlib.sha256((ROOT/'perceptbudget/power_windows.py').read_bytes()).hexdigest()
        record['norm_stats_sha256'] = hashlib.sha256(Path(config['norm_stats_path']).read_bytes()).hexdigest()
        record['reference_summary'] = json.loads((args.reference_run/'summary.json').read_text())
        base._write_json_atomic(root/'provenance.json', record)
        print('run_dir='+str(root), flush=True)
        run(root, config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
