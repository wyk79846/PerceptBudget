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
import time

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
    q=config.get('candidate_q',[5])[0]
    import numpy as np
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from scripts import probe_delay_buffer as probe
    from perceptbudget.measurement import timed
    base._patch_egl_cleanup()
    torch, _ = base._check_cuda_and_native(config)
    from perceptbudget.rtc_memory import memory_record,release_unused_native_expert
    def mem(stage):
        record=memory_record(torch,stage)
        with (run/'memory.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('memory='+json.dumps(record),flush=True)
    mem('before_model_load')
    torch.use_deterministic_algorithms(True)
    from flash_rt.subgraphs.pi05.context_action import enable
    if config.get('rtc_suite','libero_spatial') == 'libero_10':
        from scripts.libero_tasks import get_tasks
        task=get_tasks([task_id])[0]
    else:
        task = base._load_tasks([task_id], [config.get('rtc_suite','libero_spatial')])[0]
    stats = load_libero_norm_stats(config['norm_stats_path'])
    model = base._load_and_check_model(config, torch)
    mem('after_model_load')
    enable(model)
    _, cls = base._libero_api()
    env = base._make_env(cls, task)
    mem('after_environment_creation')
    env._pb_lock_fixtures = True
    try:
        obs, _ = base._prepare_env_episode(env, task, np)
        if config.get('rtc_calibrate_control_period'):
            # Same zero-action warmup rule for every q; reset before model inputs.
            samples=[]
            from scripts.probe_delay_buffer import snapshot_scene
            for _ in range(12):
                started=time.monotonic()
                snapshot_scene(env,obs)
                np.concatenate((obs['agentview_image'],obs['robot0_eye_in_hand_image']),axis=1)
                obs,_,_,_=env.step(np.zeros(7,dtype=np.float32))
                snapshot_scene(env,obs)
                env.check_success()
                elapsed=time.monotonic()-started
                time.sleep(max(0.,.05-elapsed))
                samples.append(time.monotonic()-started)
            config['rtc_initial_control_period_seconds']=float(np.median(samples))
            config['rtc_initial_control_period_source']='same-environment preflight 12 zero-action ticks; reset before episode'
            base._write_json_atomic(run/'control_period_calibration.json',dict(
                samples_seconds=samples,median_seconds=config['rtc_initial_control_period_seconds'],
                suite=task['suite'],task_id=task_id,rule=config['rtc_initial_control_period_source']))
            obs,_=base._prepare_env_episode(env,task,np)
        snapshot = base._make_snapshot(obs, stats, True)
        # Warm native context only; this native action is never executed.
        base._infer_actions(model, snapshot, task['instruction'], config['H'], {})
        mem('after_native_warmup')
        if config.get('release_unused_native_expert'):
            cleanup=release_unused_native_expert(model,torch)
            base._write_json_atomic(run/'native_expert_release.json',cleanup)
            print('native_expert_release='+json.dumps(cleanup),flush=True)
            mem('after_unused_native_expert_release')
        adapter = OriginalOpenPiRTCAdapter(model, torch, config)
        mem('after_original_expert_load')
        if adapter.expert.provenance != gate['original_source_hashes']:
            raise RuntimeError('original sources changed since verification')
        provenance = adapter.expert.provenance
        initial = {}
        adapter.bootstrap(snapshot, task['instruction'], initial)
        previous = np.asarray(initial['actions_raw'], dtype=np.float32)[5:]
        graph_audit = adapter.prepare_graph(previous) if config.get('original_cuda_graph') else None
        if config.get('source_step_prefixes'):
            graph_audit=adapter.prepare_step_bank(config['source_step_prefixes'])
        mem('after_rtc_graph_prepare')
        def infer():
            adapter.predict(snapshot, task['instruction'], previous, delay=5, executed=5, request={})
        timing = timed(torch, infer, count=3)
        initial_delay = (config['rtc_fixed_delay_estimate'] if config.get('rtc_fixed_delay_estimate') is not None else
                         5 if config.get('controlled_delay_steps') == 5 else
                         math.ceil(timing['median_ms']/1000/config.get('rtc_initial_control_period_seconds',.05)))
        config['rtc_initial_delay_steps'] = initial_delay
        record = json.loads((run/'run.json').read_text())
        record['config_snapshot'] = config
        base._write_json_atomic(run/'run.json', record)
        result = dict(status='preflight_completed', task_id=task_id, timing=timing,
                      suite=task['suite'],task_name=task['task_name'],max_steps=task['max_steps'],
                      initial_delay_steps=initial_delay, original_source_hashes=provenance,
                      graph_audit=graph_audit,
                      simulation_optimization=False,
                      experimental_attention_not_original_parity=bool(config.get('experimental_original_varlen') or config.get('rtc_exact_flash_attention') or config.get('rtc_suffix_only_vjp')),
                      attention_backend=('experimental_eager_suffix_vjp' if config.get('rtc_suffix_only_vjp') else
                                         'experimental_flash_exact' if config.get('rtc_exact_flash_attention') else 'reference_default'),
                      attention_same_input_audit=getattr(adapter,'attention_same_input_audit',None),
                      norm_same_input_audit=getattr(adapter,'norm_same_input_audit',None),
                      norm_fusion=getattr(adapter,'norm_metadata',None),
                      scope=('fixed-observation full-request timing; no async episode' if config.get('original_timing_only')
                             else 'fixed five simulated delay steps; pauses simulation if inference late; not real-time async' if config.get('controlled_delay_steps') == 5
                             else 'actual-delay async, no fixed simulation wait; actual Hz measured'))
        base._write_json_atomic(run/'summary.json', result)
        print('preflight='+json.dumps(result), flush=True)
        if config.get('original_timing_only'):
            if config.get('original_state_switches'):
                from perceptbudget.openpi_reference import sample_original_rtc
                switches=[]
                for state in (np.zeros_like(snapshot['state']),np.full_like(snapshot['state'],-.99),
                              np.full_like(snapshot['state'],.99),snapshot['state']):
                    changed=dict(snapshot,state=state)
                    rng=torch.cuda.get_rng_state()
                    request={}
                    adapter.predict(changed,task['instruction'],previous,delay=5,executed=5,request=request)
                    guidance=request['rtc_guidance']
                    torch.cuda.set_rng_state(rng)
                    noise=torch.randn(config['H'],32,device='cuda',dtype=torch.float32)
                    ref,norms=sample_original_rtc(adapter.expert,noise,
                        torch.as_tensor(previous,device='cuda'),5,5)
                    error=float(np.abs(ref.cpu().numpy()-np.asarray(request['actions_raw'])).max())
                    norm_error=float(np.max(np.abs(np.asarray(norms)-np.asarray(guidance['vjp_norms']))))
                    row=dict(valid_prefix_length=guidance['valid_prefix_length'],raw_max_abs=error,
                        vjp_norm_max_abs=norm_error,solver_backend=guidance['solver_backend'],
                        context_graph_rebuilt=guidance['context_graph_rebuilt'],
                        context_prepare_seconds=guidance['context_prepare_seconds'],
                        full_request_ms=(request['inference_ended_monotonic']-request['inference_started_monotonic'])*1000)
                    switches.append(row)
                    result['state_switches']=switches
                    base._write_json_atomic(run/'summary.json',result)
                    if (not np.isfinite(error) or not np.isfinite(norm_error) or max(error,norm_error)>=1e-4
                            or guidance['context_graph_rebuilt'] or guidance['context_prepare_seconds']):
                        raise RuntimeError('state switch parity or no-runtime-capture gate failed')
                    print('state_switch='+json.dumps(row),flush=True)
            adapter.release_graph()
            result.update(status='completed', timing_only=True, episode_started=False,
                          note='same observation/full context + original guided expert; no task success rate')
            return result
        if q+initial_delay > config['H']:
            adapter.release_graph()
            result.update(status='infeasible_before_episode', reason='measured full latency exceeds available H-s old tail; no clipping or horizon change')
            return result
        directory = run/'rtc'
        directory.mkdir()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        mem('before_episode_reset')
        with (directory/'trace.jsonl').open('w') as stream:
            episode = probe.run_episode(env, task, q, model, stats, config, np, torch,
                directory, stream, rtc_adapter=adapter, condition='original_openpi_rtc',
                refresh_policy='latest_due', pending_arm_gain=1.)
        requests = json.loads((directory/f'q{q}_requests.json').read_text())
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
        if config.get('controlled_delay_steps') is None:
            assert all('controlled_wait_seconds' not in r for r in requests)
        assert all(r['rtc_guidance']['expert_backend']=='original_openpi_source' for r in requests)
        times = [(r['inference_ended_monotonic']-r['inference_started_monotonic'])*1000 for r in requests[1:]]
        result.update(status='completed', episode=episode, executed_actions_audited=audited,
                      steady_inference_median_ms=float(np.median(times)) if times else None)
        return result
    finally:
        env.close()


def horizon_transfer_allowed(args,reference_h):
    legacy=(args.actual_delay and args.experimental_varlen and args.horizon in (25,50) and reference_h==20)
    source=(args.allow_horizon_transfer and args.actual_delay and args.author_order_reference
            and args.source_step_prefixes and args.horizon in (20,50) and reference_h==25)
    return bool(legacy or source)


if __name__ == "__main__":
    print("This is an internal helper; use scripts/run_baseline.py", file=sys.stderr)
    raise SystemExit(2)
