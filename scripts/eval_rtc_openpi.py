"""Correctness-first original OpenPI RTC: fixture gate, then one async task.

Use --verify-only before --verified-run <gate>. No simulation optimization,
model training, custom backward or full-VLM duplicate is involved.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base


def verify(config, run):
    import numpy as np
    import torch
    from perceptbudget.openpi_reference import OriginalOpenPiExpert, sample_original_rtc
    reference = Path(config['original_reference_fixture'])
    reference_record = json.loads((reference/'summary.json').read_text())
    if reference_record.get('variant') != 'original' or not reference_record.get('author_euler_order'):
        raise ValueError('require independent original expert with author Euler operation order')
    prefix = Path(config['output_root'])/'M1/20261019T054917.649092Z-capacity-numeric-diagnosis-5f903047/capacity80a'
    expert = OriginalOpenPiExpert(config, 20, 5)
    trace = np.load(prefix/'trace.npz')
    def bits(value):
        return torch.from_numpy(value.copy()).to('cuda').view(torch.bfloat16)
    keys = torch.stack([bits(trace[f'21_encoder_layer{i:02d}_key']) for i in range(18)])
    values = torch.stack([bits(trace[f'21_encoder_layer{i:02d}_value']) for i in range(18)])
    expert.refresh_context(keys, values, keys.shape[1])
    saved = np.load(reference/'reference.npz')
    rows = []
    for step in range(5):
        x = torch.as_tensor(saved[f'input{step}'], device='cuda').requires_grad_(True)
        error = torch.as_tensor(saved[f'error{step}'], device='cuda')
        v = expert.velocity(x, step)
        correction, = torch.autograd.grad(x-(1-step/5)*v, x, grad_outputs=error)
        row = dict(step=step)
        for name, tensor in (('velocity', v), ('vjp', correction)):
            row[name+'_max_abs'] = float(np.max(np.abs(tensor.detach().cpu().numpy()-saved[f'{name}{step}'])))
        rows.append(row)
    noise = torch.as_tensor(trace['32_fixed_noise'], device='cuda')
    raw, _ = sample_original_rtc(expert, noise, noise[5:].clone(), 5, 5)
    raw_error = float(np.max(np.abs(raw.cpu().numpy()-saved['raw'])))
    frozen = all(not p.requires_grad and p.grad is None for p in expert.suffix.parameters())
    passed = frozen and raw_error < 1e-4 and all(max(r['velocity_max_abs'], r['vjp_max_abs']) < 1e-4 for r in rows)
    return dict(status='completed' if passed else 'failed_gate', passed=passed,
                reference=str(reference), raw_max_abs=raw_error, per_step=rows, weights_frozen=frozen,
                original_source_hashes=expert.provenance,
                scope='original OpenPI suffix against independent original-source fixture; encoder KV shared, not full model proof')


def rollout(config, run, gate, task_id):
    import numpy as np
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from scripts import probe_delay_buffer as probe
    from scripts.probe_rtc_step import timed
    base._patch_egl_cleanup()
    torch, _ = base._check_cuda_and_native(config)
    torch.use_deterministic_algorithms(True)
    from flash_rt.subgraphs.pi05.context_action import enable
    task = base._load_tasks([task_id], ['libero_spatial'])[0]
    stats = load_libero_norm_stats(config['norm_stats_path'])
    model = base._load_and_check_model(config, torch)
    enable(model)
    _, cls = base._libero_api()
    env = base._make_env(cls, task)
    env._pb_lock_fixtures = True
    try:
        obs, _ = base._prepare_env_episode(env, task, np)
        snapshot = base._make_snapshot(obs, stats, True)
        # Warm native context only; this native action is never executed.
        base._infer_actions(model, snapshot, task['instruction'], 20, {})
        adapter = OriginalOpenPiRTCAdapter(model, torch, config)
        if adapter.expert.provenance != gate['original_source_hashes']:
            raise RuntimeError('original sources changed since verification')
        provenance = adapter.expert.provenance
        initial = {}
        adapter.bootstrap(snapshot, task['instruction'], initial)
        previous = np.asarray(initial['actions_raw'], dtype=np.float32)[5:]
        def infer():
            adapter.predict(snapshot, task['instruction'], previous, delay=5, executed=5, request={})
        timing = timed(torch, infer, count=3)
        initial_delay = math.ceil(timing['median_ms']/50)
        config['rtc_initial_delay_steps'] = initial_delay
        record = json.loads((run/'run.json').read_text())
        record['config_snapshot'] = config
        base._write_json_atomic(run/'run.json', record)
        result = dict(status='preflight_completed', task_id=task_id, timing=timing,
                      initial_delay_steps=initial_delay, original_source_hashes=provenance,
                      simulation_optimization=False, scope='actual-delay async, no fixed simulation wait; actual Hz measured')
        base._write_json_atomic(run/'summary.json', result)
        print('preflight='+json.dumps(result), flush=True)
        if 5+initial_delay > 20:
            adapter.release_graph()
            result.update(status='infeasible_before_episode', reason='measured full latency exceeds available H-s old tail; no clipping or horizon change')
            return result
        directory = run/'rtc'
        directory.mkdir()
        with (directory/'trace.jsonl').open('w') as stream:
            episode = probe.run_episode(env, task, 5, model, stats, config, np, torch,
                directory, stream, rtc_adapter=adapter, condition='original_openpi_rtc',
                refresh_policy='latest_due', pending_arm_gain=1.)
        requests = json.loads((directory/'q5_requests.json').read_text())
        by_id = {r['prediction_id']: r for r in requests}
        audited = 0
        for line in (directory/'trace.jsonl').open():
            row = json.loads(line)
            if row['event'] != 'control_step':
                continue
            assert row['applied_arm_gain'] == 1
            source = row['source']
            if source is not None:
                assert source['origin_step']+source['offset'] == row['step']
                assert np.array_equal(np.asarray(row['action'], dtype=np.float32),
                    np.asarray(by_id[source['prediction_id']]['actions'][source['offset']], dtype=np.float32))
                audited += 1
        assert all('controlled_wait_seconds' not in r for r in requests)
        assert all(r['rtc_guidance']['expert_backend']=='original_openpi_source' for r in requests)
        times = [(r['inference_ended_monotonic']-r['inference_started_monotonic'])*1000 for r in requests[1:]]
        result.update(status='completed', episode=episode, executed_actions_audited=audited,
                      steady_inference_median_ms=float(np.median(times)) if times else None)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verified-run', type=Path)
    parser.add_argument('--reference-fixture', type=Path)
    parser.add_argument('--task-id', type=int, choices=range(10), default=0)
    args = parser.parse_args()
    if not args.verify_only and args.verified_run is None:
        parser.error('require --verify-only or a passed --verified-run')
    if args.verify_only and args.reference_fixture is None:
        parser.error('--verify-only requires --reference-fixture (author Euler order)')
    config = load_config(repo_root=ROOT)
    config.update(H=20, N=5, state_enabled=True, state_prompt_capacity=80, candidate_q=[5],
        native_gemm_policy='heuristic-first', execution_mode='original_openpi_rtc_actual_delay',
        rtc_paper_schedule=True, rtc_delay_buffer_size=10, reuse_control_forward=False,
        rtc_wait_for_delay_feasibility=False, release_rtc_graph_before_artifacts=True,
        disk_backed_frames=True, pending_arm_gain=1.0,
        rtc_expert='original_openpi_source', rtc_action_state_dtype='float32')
    config.pop('controlled_delay_steps', None)
    if args.reference_fixture:
        config['original_reference_fixture'] = str(args.reference_fixture.resolve())
    config['screen'].update(seed=7, init_state_index=0, suites=['libero_spatial'], task_ids=[args.task_id])
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'openpi-rtc-gate' if args.verify_only else 'openpi-rtc-actual', repo_root=ROOT)
        print('run_dir='+str(run), flush=True)
        for name in ('scripts/eval_rtc_openpi.py', 'scripts/probe_delay_buffer.py',
                     'perceptbudget/openpi_reference.py', 'perceptbudget/rtc_openpi.py'):
            dest = run/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dest)
        result = dict(status='running')
        try:
            if args.verify_only:
                result = verify(config, run)
            else:
                gate = json.loads((args.verified_run/'summary.json').read_text())
                if not gate.get('passed'):
                    raise RuntimeError('reference gate not passed')
                gate_config = json.loads((args.verified_run/'run.json').read_text())['config_snapshot']
                for key in ('H','N','model_path','openpi_root','checkpoint_path'):
                    if config.get(key) != gate_config.get(key):
                        raise RuntimeError('reference gate config mismatch: '+key)
                config['verified_reference_run'] = str(args.verified_run.resolve())
                result = rollout(config, run, gate, args.task_id)
            print(json.dumps(result), flush=True)
        except BaseException as exc:
            result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json', result)
            base._save_run_status(run, result['status'])


if __name__ == '__main__':
    main()
