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
    q=config.get('candidate_q',[5])[0]
    import numpy as np
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from scripts import probe_delay_buffer as probe
    from scripts.probe_rtc_step import timed
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
        from scripts.eval_seam_sync import get_tasks
        task=get_tasks([task_id])[0]
    else:
        task = base._load_tasks([task_id], ['libero_spatial'])[0]
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
                         5 if config.get('controlled_delay_steps') == 5 else math.ceil(timing['median_ms']/50))
        config['rtc_initial_delay_steps'] = initial_delay
        record = json.loads((run/'run.json').read_text())
        record['config_snapshot'] = config
        base._write_json_atomic(run/'run.json', record)
        result = dict(status='preflight_completed', task_id=task_id, timing=timing,
                      suite=task['suite'],task_name=task['task_name'],max_steps=task['max_steps'],
                      initial_delay_steps=initial_delay, original_source_hashes=provenance,
                      graph_audit=graph_audit,
                      simulation_optimization=False,
                      experimental_attention_not_original_parity=config.get('experimental_original_varlen',False),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verified-run', type=Path)
    parser.add_argument('--reference-fixture', type=Path)
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--cuda-graph', action='store_true',help='CUDA graph for explicit timing or quality/async experiments')
    parser.add_argument('--state-switches',action='store_true',help='Validate changed state inputs, no task episode')
    parser.add_argument('--controlled-delay-five',action='store_true',help='Quality-only test with simulation waiting at d=5')
    parser.add_argument('--experimental-varlen',action='store_true',help='Attention-only candidate; not independently original-parity')
    parser.add_argument('--actual-delay',action='store_true',help='Explicit actual-delay async experiment; never pause simulation for inference')
    parser.add_argument('--fixed-delay-estimate',type=int,help='Actual async only: fixed guidance estimate; crop still uses actual elapsed control steps')
    parser.add_argument('--allow-horizon-transfer',action='store_true',help='Explicit H25 gate provenance transfer to H20/H50 source-step actual experiment; live graph parity still required')
    parser.add_argument('--author-order-reference',action='store_true',help='Original attention/expert and author-source order; controlled audit or explicit source-step async')
    parser.add_argument('--source-step-prefixes',nargs='+',type=int,help='Opt-in original single-step graphs for up to eight exact lengths; miss falls back slow')
    parser.add_argument('--source-step-profile',type=Path,help='Read this task prefix lengths from a matched slow-batch summary; unseen lengths use slow fallback')
    parser.add_argument('--source-packed-conditioning',action='store_true',help='Opt-in exact fixed-time results with source-step graph bank')
    parser.add_argument('--release-unused-native-expert',action='store_true',help='One-way context-only FlashRT in this process; free duplicate BF16 expert')
    parser.add_argument('--task-id', type=int, choices=range(10), default=0)
    parser.add_argument('--suite',choices=('libero_spatial','libero_10'),default='libero_spatial')
    parser.add_argument('--horizon',type=int,choices=(20,25,50),default=20,
                        help='Experimental prediction horizon; does not change q=5 or N=5')
    args = parser.parse_args()
    if args.fixed_delay_estimate is not None and (not args.actual_delay or not args.author_order_reference
            or not 0<=args.fixed_delay_estimate<args.horizon):
        parser.error('--fixed-delay-estimate requires author-order actual-delay and 0 <= estimate < horizon')
    step_profile=None
    if args.source_step_profile:
        if args.source_step_prefixes:
            parser.error('choose source-step-profile or source-step-prefixes, not both')
        from perceptbudget.rtc_source_step_graph import lengths_from_reference_profile
        try:
            args.source_step_prefixes,step_profile=lengths_from_reference_profile(
                args.source_step_profile,task_id=args.task_id,suite=args.suite,horizon=args.horizon)
        except (OSError,ValueError,KeyError) as exc:
            parser.error(str(exc))
    if args.source_packed_conditioning and not args.source_step_prefixes:
        parser.error('--source-packed-conditioning requires source-step-prefixes or profile')
    if args.allow_horizon_transfer and not (args.actual_delay and args.author_order_reference
            and args.source_step_prefixes and args.horizon in (20,50)):
        parser.error('--allow-horizon-transfer requires H20/H50 source-step author-order actual-delay')
    if args.source_step_prefixes and (not args.author_order_reference or not (args.controlled_delay_five or args.actual_delay)
            or len(set(args.source_step_prefixes))>8 or any(not 1<=n<=592 for n in args.source_step_prefixes)):
        parser.error('--source-step-prefixes requires author-order-reference, controlled-delay-five or actual-delay, and at most eight lengths')
    if args.author_order_reference and (not (args.controlled_delay_five or (args.actual_delay and args.source_step_prefixes))
            or args.cuda_graph or args.experimental_varlen or args.verify_only):
        parser.error('--author-order-reference requires controlled-delay-five or source-step actual-delay; forbids legacy cuda-graph/varlen/verify')
    if args.horizon != 20 and not (args.actual_delay or args.author_order_reference):
        parser.error('non-default horizon requires explicit --actual-delay experiment')
    if args.actual_delay and (args.controlled_delay_five or args.timing_only or args.verify_only):
        parser.error('--actual-delay cannot combine with controlled delay, timing-only or verify-only')
    if args.actual_delay and not ((args.experimental_varlen and args.cuda_graph)
            or (args.author_order_reference and args.source_step_prefixes)):
        parser.error('--actual-delay requires source-step author-order-reference or legacy experimental-varlen cuda-graph')
    if args.experimental_varlen and not (args.controlled_delay_five or args.actual_delay):
        parser.error('--experimental-varlen requires explicit controlled or actual-delay experiment')
    if args.cuda_graph and (not (args.timing_only or args.controlled_delay_five or args.actual_delay) or args.verify_only):
        parser.error('--cuda-graph requires timing-only, controlled-delay-five or actual-delay')
    if args.state_switches and not (args.cuda_graph and args.timing_only):
        parser.error('--state-switches requires --cuda-graph --timing-only')
    if not args.verify_only and args.verified_run is None:
        parser.error('require --verify-only or a passed --verified-run')
    if args.verify_only and args.reference_fixture is None:
        parser.error('--verify-only requires --reference-fixture (author Euler order)')
    config = load_config(repo_root=ROOT)
    config.update(H=args.horizon, N=5, state_enabled=True, state_prompt_capacity=80, candidate_q=[5],
        native_gemm_policy='heuristic-first', execution_mode='original_openpi_rtc_actual_delay',
        rtc_paper_schedule=True, rtc_delay_buffer_size=10, reuse_control_forward=False,
        rtc_wait_for_delay_feasibility=False, release_rtc_graph_before_artifacts=True,
        disk_backed_frames=True, pending_arm_gain=1.0,
        rtc_expert='original_openpi_source', rtc_action_state_dtype='float32')
    config.update(original_cuda_graph=args.cuda_graph,original_timing_only=args.timing_only,
                  original_state_switches=args.state_switches)
    config.pop('controlled_delay_steps', None)
    if args.controlled_delay_five:
        config.update(controlled_delay_steps=5,rtc_paper_schedule=False,
                      execution_mode='original_expert_controlled_delay_quality_only')
    config['experimental_original_varlen']=args.experimental_varlen
    config.update(rtc_author_order=args.author_order_reference,rtc_audit_inputs=args.author_order_reference)
    config['source_step_prefixes']=args.source_step_prefixes
    config['source_step_profile']=step_profile
    config['source_packed_conditioning']=args.source_packed_conditioning
    config['release_unused_native_expert']=args.release_unused_native_expert
    config['explicit_actual_delay_experiment']=args.actual_delay
    config['rtc_fixed_delay_estimate']=args.fixed_delay_estimate
    if args.reference_fixture:
        config['original_reference_fixture'] = str(args.reference_fixture.resolve())
    config['rtc_suite']=args.suite
    config['screen'].update(seed=7, init_state_index=0, suites=[args.suite], task_ids=[args.task_id])
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'openpi-rtc-gate' if args.verify_only else 'openpi-rtc-actual', repo_root=ROOT)
        print('run_dir='+str(run), flush=True)
        for name in ('scripts/eval_rtc_openpi.py', 'scripts/probe_delay_buffer.py',
                     'perceptbudget/openpi_reference.py', 'perceptbudget/rtc_openpi.py',
                     'perceptbudget/openpi_graph.py','perceptbudget/openpi_varlen.py',
                     'perceptbudget/rtc_expert.py','perceptbudget/rtc_memory.py',
                     'perceptbudget/rtc_schedule.py','perceptbudget/rtc_author_source.py',
                     'scripts/eval_seam_sync.py','perceptbudget/rtc_source_step_graph.py',
                     'perceptbudget/rtc_condition_cache.py'):
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
                if args.author_order_reference:
                    import hashlib
                    source=Path(config['output_root'])/'rtc-author-reference/src/model.py'
                    if not gate.get('author_order') or not gate.get('weights_frozen') or gate.get('device')!='cuda':
                        raise RuntimeError('author-order reference requires passed frozen-original GPU source-order gate')
                    if hashlib.sha256(source.read_bytes()).hexdigest()!=gate.get('author_source_sha256'):
                        raise RuntimeError('author source changed since source-order gate')
                gate_config = json.loads((args.verified_run/'run.json').read_text())['config_snapshot']
                for key in ('N','model_path','openpi_root','checkpoint_path'):
                    if config.get(key) != gate_config.get(key):
                        raise RuntimeError('reference gate config mismatch: '+key)
                if config['H'] != gate_config.get('H'):
                    if not horizon_transfer_allowed(args,gate_config.get('H')):
                        raise RuntimeError('reference gate config mismatch: H')
                    config['reference_horizon_transfer']=dict(reference_H=gate_config.get('H'),experiment_H=config['H'],
                        limitation='source/checkpoint provenance transfer; live exact-shape graph/eager parity required for source-step path; not independent author-body horizon validation')
                config['verified_reference_run'] = str(args.verified_run.resolve())
                result = rollout(config, run, gate, args.task_id)
            print(json.dumps(result), flush=True)
            return 0 if result['status'] == 'completed' else 2
        except BaseException as exc:
            result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json', result)
            base._save_run_status(run, result['status'])


if __name__ == '__main__':
    raise SystemExit(main())
