"""Stage-level timing of the production H50 RTC request path.

Wraps live objects only; no production source or algorithm change.
Fixed observation, fixed seed noise; inputs identical across iterations.
The wrapped sequence is exactly OriginalOpenPiRTCAdapter._predict's order.
"""
import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run

DEFAULT_REFERENCE = Path(
    '/home/wyk/workspace/任务产出/perceptbudget/M1/'
    '20261020T080811.461979Z-rtc-diag-actual_history-f7c12c37')


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def summarize(values):
    if not values:
        return dict(count=0)
    return dict(count=len(values), median=sorted(values)[len(values)//2],
                p95=percentile(values, .95), max=max(values), min=min(values))


def aggregate(rows):
    """Median/P95 per stage over steady rows; pure function for tests."""
    steady = [r for r in rows if not r.get('warmup')]
    if not steady:
        return dict(iterations=0)
    out = dict(iterations=len(steady),
               total_ms=summarize([r['total_ms'] for r in steady]))
    for field in ('walls', 'gaps', 'gpu'):
        stages = {}
        for row in steady:
            for name, value in row.get(field, {}).items():
                values = value if isinstance(value, list) else [value]
                stages.setdefault(name, []).extend(values)
        out[field] = {name: summarize(values) for name, values in sorted(stages.items())}
    return out


class Recorder:
    """Record wall time and optional CUDA events around wrapped bound methods."""

    def __init__(self):
        self.entries = []

    def clear(self):
        self.entries = []

    def wrap(self, obj, attr, name, *, events=False):
        import torch
        original = getattr(obj, attr)
        rec = self

        def wrapper(*args, **kwargs):
            started = time.monotonic()
            if events:
                before = torch.cuda.Event(enable_timing=True)
                before.record()
            result = original(*args, **kwargs)
            if events:
                after = torch.cuda.Event(enable_timing=True)
                after.record()
            ended = time.monotonic()
            rec.entries.append(dict(name=name, started=started, ended=ended,
                                    event=(before, after) if events else None))
            return result

        setattr(obj, attr, wrapper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--delay', type=int, default=12)
    parser.add_argument('--executed', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--eager-reps', type=int, default=3)
    parser.add_argument('--profile-expert', type=int, default=0,
                        help='Profile N step-graph replays with torch.profiler after the steady loop')
    parser.add_argument('--gemm-policy', choices=('heuristic-first', 'autotune'), default=None,
                        help='Override native GEMM policy for this measurement run only')
    args = parser.parse_args()
    if args.iterations < 5 or args.warmup < 1 or not 0 < args.delay <= 45 \
            or not 0 <= args.executed <= 50 or args.executed + 5 > 50:
        parser.error('invalid iteration/guidance parameters')

    reference = json.loads((args.reference_run / 'run.json').read_text())['config_snapshot']
    gate = json.loads((args.reference_run / 'summary.json').read_text())
    config = dict(reference)
    if args.gemm_policy:
        config['native_gemm_policy'] = args.gemm_policy
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                      PYOPENGL_PLATFORM='egl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (Path(config['output_root']) / '.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'rtc-stage-profile-h50', repo_root=ROOT)
        print('run_dir=' + str(run), flush=True)
        for name in ('scripts/probe_rtc_stage_profile.py',
                     'perceptbudget/rtc_openpi.py',
                     'perceptbudget/openpi_reference.py',
                     'perceptbudget/rtc_source_step_graph.py',
                     'perceptbudget/rtc_condition_cache.py',
                     'perceptbudget/h50_checkpoint.py',
                     'perceptbudget/libero_input.py'):
            dest = run / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        result = dict(status='running', reference_run=str(args.reference_run),
                      iterations=args.iterations, warmup=args.warmup,
                      delay=args.delay, executed=args.executed, seed=args.seed)
        try:
            profile = run_once(config, gate, run, args)
            result.update(profile)
            result['status'] = 'completed'
        except BaseException as exc:
            result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            from scripts.eval_q import _write_json_atomic, _save_run_status
            _write_json_atomic(run / 'summary.json', result)
            _save_run_status(run, result['status'])
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ('rows', 'aggregate')}, ensure_ascii=False), flush=True)
        return 0


def run_once(config, gate, run, args):
    import numpy as np
    from scripts import eval_q as base
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.rtc_memory import memory_record, release_unused_native_expert

    base._patch_egl_cleanup()
    torch, _ = base._check_cuda_and_native(config)
    torch.use_deterministic_algorithms(True)
    from flash_rt.subgraphs.pi05.context_action import enable

    def mem(stage):
        record = memory_record(torch, stage)
        with (run / 'memory.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print('memory=' + json.dumps(record), flush=True)

    mem('before_model_load')
    task = base._load_tasks([config['screen']['task_ids'][0]], ['libero_spatial'])[0]
    stats = load_libero_norm_stats(config['norm_stats_path'])
    model = base._load_and_check_model(config, torch)
    enable(model)
    mem('after_model_load')
    _, cls = base._libero_api()
    env = base._make_env(cls, task)
    env._pb_lock_fixtures = True
    obs, _ = base._prepare_env_episode(env, task, np)
    snapshot = base._make_snapshot(obs, stats, True)
    base._infer_actions(model, snapshot, task['instruction'], config['H'], {})
    mem('after_native_warmup')
    if config.get('release_unused_native_expert'):
        release_unused_native_expert(model, torch)
        mem('after_unused_native_expert_release')
    adapter = OriginalOpenPiRTCAdapter(model, torch, config)
    if adapter.expert.provenance != gate['original_source_hashes']:
        raise RuntimeError('original sources differ from reference run')
    mem('after_original_expert_load')

    torch.cuda.manual_seed(args.seed)
    initial = {}
    adapter.bootstrap(snapshot, task['instruction'], initial)
    initial_raw = np.asarray(initial['actions_raw'], dtype=np.float32)
    tail = config['H'] - args.executed
    if tail <= 0 or args.executed + args.delay > config['H']:
        raise ValueError('guided request exceeds horizon overlap')
    previous = initial_raw[config['H'] - tail:]
    graph_audit = adapter.prepare_step_bank(config['source_step_prefixes'])
    mem('after_rtc_graph_prepare')

    recorder = Recorder()
    pipe, pl = adapter.pipe, adapter.pipeline
    recorder.wrap(adapter.model, 'set_prompt', 'set_prompt')
    recorder.wrap(pipe, '_fill_img_buf', 'fill_img')
    recorder.wrap(pipe, '_copy_tensor_to_pipeline_buf_stream', 'img_upload')
    recorder.wrap(pl._context_graph, 'replay', 'context_replay', events=True)
    recorder.wrap(adapter.expert, 'refresh_context', 'refresh_context', events=True)
    for valid, (bound, step_graph) in adapter.step_bank.entries.items():
        recorder.wrap(bound, 'refresh_context', 'refresh_context', events=True)
        recorder.wrap(step_graph.graph, 'replay', 'step_replay', events=True)
        if step_graph.condition_cache is not None:
            recorder.wrap(step_graph.condition_cache, 'select_step', 'cond_select',
                          events=True)

    rows = []
    first_raw = None
    instruction = task['instruction']
    steady_raw_saved = False
    for iteration in range(args.warmup + args.iterations):
        torch.cuda.manual_seed(args.seed)
        recorder.clear()
        request = {}
        started = time.monotonic()
        adapter.predict(snapshot, instruction, previous,
                        delay=args.delay, executed=args.executed, request=request)
        ended = time.monotonic()
        torch.cuda.synchronize()
        walls, gpu = {}, {}
        for entry in recorder.entries:
            walls.setdefault(entry['name'], []).append((entry['ended'] - entry['started']) * 1000)
            if entry['event'] is not None:
                gpu.setdefault(entry['name'], []).append(entry['event'][0].elapsed_time(entry['event'][1]))
        ordered = sorted(recorder.entries, key=lambda e: e['started'])
        gaps = {}
        if ordered:
            gaps['pre'] = (ordered[0]['started'] - started) * 1000
            for before, after in zip(ordered, ordered[1:]):
                gaps.setdefault('gap_after_' + before['name'], []).append(
                    (after['started'] - before['ended']) * 1000)
            gaps['tail'] = (ended - ordered[-1]['ended']) * 1000
        raw = np.asarray(request['actions_raw'], dtype=np.float32)
        if first_raw is None:
            first_raw = raw
        diff = float(np.abs(raw - first_raw).max())
        steady = iteration >= args.warmup
        if steady and not steady_raw_saved:
            # One canonical raw/norms pair saved for cross-version parity checks.
            (run / 'steady_reference.json').write_text(json.dumps(dict(
                actions_raw=raw.tolist(),
                vjp_norms=request['rtc_guidance']['vjp_norms'],
                valid=request['rtc_guidance']['valid_prefix_length'],
                delay=args.delay, executed=args.executed, seed=args.seed,
                solver_inputs=request.get('rtc_solver_inputs'))))
            steady_raw_saved = True
        rows.append(dict(iteration=iteration, warmup=iteration < args.warmup,
                         total_ms=(ended - started) * 1000,
                         valid=request['rtc_guidance']['valid_prefix_length'],
                         backend=request['rtc_guidance']['solver_backend'],
                         raw_max_diff_vs_first=diff,
                         walls={k: v if len(v) > 1 else v[0] for k, v in walls.items()},
                         gpu={k: v if len(v) > 1 else v[0] for k, v in gpu.items()},
                         gaps={k: v if isinstance(v, list) and len(v) > 1
                               else v[0] if isinstance(v, list) else v for k, v in gaps.items()}))
        print('iter=%d total=%.2fms valid=%d diff=%.3g' % (
            iteration, rows[-1]['total_ms'], rows[-1]['valid'], diff), flush=True)

    eager = []
    for rep in range(args.eager_reps):
        torch.cuda.manual_seed(args.seed)
        request = {}
        started = time.monotonic()
        adapter.bootstrap(snapshot, instruction, request)
        ended = time.monotonic()
        eager.append((ended - started) * 1000)
        print('eager_rep=%d total=%.2fms' % (rep, eager[-1]), flush=True)

    mem('after_steady_loop')
    aggregate_out = aggregate(rows)
    # Forward-only vs full-step split: run the same velocity forward without
    # grad to see how much of the 5x56ms is backward/VJP. Uses the same bound
    # expert/KV as the step graph; pure measurement, no production change.
    split = {}
    valid = rows[-1]['valid']
    bound, step_graph = adapter.step_bank.select(valid)
    torch.cuda.manual_seed(args.seed)
    noise = torch.randn(config['H'], 32, device='cuda', dtype=torch.float32)
    previous_t = torch.as_tensor(previous, device='cuda', dtype=torch.float32)
    for name, count in (('full_step_replay', 10), ('forward_only', 10)):
        times = []
        for rep in range(count):
            torch.cuda.synchronize()
            started = time.monotonic()
            if name == 'full_step_replay':
                step_graph.replay(noise, previous_t, args.delay, args.executed)
                torch.cuda.synchronize()
            else:
                with torch.no_grad():
                    x = noise
                    for step in range(config['N']):
                        x = x - bound.velocity(x, step) / config['N']
                torch.cuda.synchronize()
            times.append((time.monotonic() - started) * 1000)
        split[name] = summarize(times[2:])
        print('%s median=%.2fms' % (name, split[name]['median']), flush=True)
    # Whole-denoising graph candidate: one capture for all 5 steps, then
    # parity-check against the step-graph path on the exact same inputs.
    whole_parity = {}
    try:
        started = time.monotonic()
        step_graph.capture_full()
        capture_seconds = time.monotonic() - started
        times = []
        raw_whole = None
        for rep in range(10):
            torch.cuda.synchronize()
            started = time.monotonic()
            raw_whole, norms_whole = step_graph.replay_full(
                noise, previous_t, args.delay, args.executed)
            torch.cuda.synchronize()
            times.append((time.monotonic() - started) * 1000)
        split['whole_denoise_replay'] = summarize(times[2:])
        raw_step, norms_step = step_graph.replay(noise, previous_t, args.delay, args.executed)
        whole_parity = dict(
            capture_seconds=capture_seconds,
            raw_max_abs=float((raw_whole - raw_step).abs().max()),
            vjp_norm_max_abs=float((norms_whole - norms_step).abs().max()),
            limitation='same step kernels, Python inter-step loop removed; parity vs step graph on identical inputs')
        print('whole_denoise_replay median=%.2fms raw_diff=%g norm_diff=%g' % (
            split['whole_denoise_replay']['median'],
            whole_parity['raw_max_abs'], whole_parity['vjp_norm_max_abs']), flush=True)
    except BaseException as exc:
        whole_parity = dict(error=f'{type(exc).__name__}: {exc}')
        print('whole graph failed: %r' % exc, flush=True)
    # Forward-only whole graph: how much of the 5 steps is backward/VJP?
    try:
        fwd_graph = torch.cuda.CUDAGraph()
        fwd_x = torch.zeros_like(noise)
        steps = config['N']

        def fwd_run():
            with torch.no_grad():
                x = fwd_x
                for step in range(steps):
                    x = x - bound.velocity(x, step) / steps
                return x.detach()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fwd_x.copy_(noise); fwd_run()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(fwd_graph, stream=side):
            fwd_out = fwd_run()
        torch.cuda.synchronize()
        times = []
        for rep in range(10):
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.no_grad():
                fwd_x.copy_(noise)
            fwd_graph.replay()
            torch.cuda.synchronize()
            times.append((time.monotonic() - started) * 1000)
        split['forward_only_graph'] = summarize(times[2:])
        # Parity: same arithmetic as bootstrap (no guidance) path.
        eager_x = noise.clone()
        with torch.no_grad():
            for step in range(steps):
                eager_x = eager_x - bound.velocity(eager_x, step) / steps
        fwd_parity = float((fwd_out - eager_x).abs().max())
        print('forward_only_graph median=%.2fms parity=%g' % (
            split['forward_only_graph']['median'], fwd_parity), flush=True)
        whole_parity['forward_only_parity'] = fwd_parity
    except BaseException as exc:
        whole_parity['forward_only_error'] = f'{type(exc).__name__}: {exc}'
        print('forward-only graph failed: %r' % exc, flush=True)
    expert_profile = {}
    if args.profile_expert > 0:
        import torch.profiler
        valid = rows[-1]['valid']
        _, step_graph = adapter.step_bank.select(valid)
        torch.cuda.manual_seed(args.seed)
        noise = torch.randn(config['H'], 32, device='cuda', dtype=torch.float32)
        previous_t = torch.as_tensor(previous, device='cuda', dtype=torch.float32)
        # Warm the profiled path so compilation/cudnn caches are not attributed.
        step_graph.replay(noise, previous_t, args.delay, args.executed)
        activities = [torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as prof:
            for _ in range(args.profile_expert):
                step_graph.replay(noise, previous_t, args.delay, args.executed)
        table = prof.key_averages().table(sort_by='cuda_time_total', row_limit=25)
        (run / 'expert_profile.txt').write_text(table)
        expert_profile = dict(
            note='torch.profiler over step_graph.replay only; see expert_profile.txt',
            replays=args.profile_expert)
        print(table, flush=True)
    summary = dict(
        scope='fixed observation, no simulation running; wrapped production path only',
        total_ms=aggregate_out['total_ms'],
        stages=aggregate_out,
        eager_bootstrap_ms=eager,
        expert_split=split,
        expert_profile=expert_profile,
        determinism=dict(raw_max_abs_across_iterations=max(
            r['raw_max_diff_vs_first'] for r in rows if not r['warmup']),
            seed=args.seed,
            note='identical seed per iteration; nonzero diff means nondeterminism'),
        valid_lengths=sorted({r['valid'] for r in rows}),
        backend=rows[-1]['backend'],
        graph_hit=all(r['backend'] == 'original_source_step_graph' for r in rows
                      if not r['warmup']),
        graph_audit=dict(prepare_seconds=graph_audit['prepare_seconds'],
                         lengths=graph_audit['lengths'],
                         audits=[{k: a[k] for k in ('valid', 'raw_max_abs',
                                                     'vjp_norm_max_abs', 'passed')}
                                 for a in graph_audit['audits']]),
        memory=dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    reserved_mib=torch.cuda.memory_reserved() / 2**20),
    )
    (run / 'stage_rows.json').write_text(json.dumps(rows, indent=1))
    adapter.release_graph()
    del env, model, adapter
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return dict(summary=summary, aggregate=aggregate_out)


if __name__ == '__main__':
    raise SystemExit(main())
