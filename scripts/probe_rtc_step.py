#!/usr/bin/env python3
"""Opt-in FlashRT extension: single-step genuine RTC VJP capture gate."""

import ctypes
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import load_libero_norm_stats


def timed(torch, fn, count=5):
    values = []
    for _ in range(count):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return {'trials_ms': values, 'median_ms': statistics.median(values)}


def main():
    config = load_config(repo_root=ROOT)
    root = create_run(config, 'M1', 'rtc-step-graph', repo_root=ROOT)
    print('run_dir='+str(root), flush=True)
    env = None
    result = {'status': 'loading', 'scope': 'one denoise step only, no task trial',
              'backend': 'opt-in FlashRT VJP graph, project differentiable expert; no native backward kernels yet'}
    for repo, name in [(ROOT, 'scripts/probe_rtc_step.py'), (ROOT, 'perceptbudget/rtc_expert.py'),
                       (Path(config['flashrt_root']), 'flash_rt/subgraphs/pi05/rtc_action_vjp.py')]:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, target)
    result['flashrt_revision'] = subprocess.check_output(
        ['git', '-C', config['flashrt_root'], 'rev-parse', 'HEAD'], text=True).strip()
    try:
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        torch, _ = base._check_cuda_and_native(config)
        import numpy as np
        from perceptbudget.rtc_expert import DifferentiableExpert
        from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedActionVjp
        stats = load_libero_norm_stats(config['norm_stats_path'])
        model = base._load_and_check_model(config, torch)
        task = base._load_tasks([0], config['screen']['suites'])[0]
        _, env_class = base._libero_api()
        env = base._make_env(env_class, task)
        obs, _ = base._prepare_env_episode(env, task, np)
        snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
        base._infer_actions(model, snapshot, task['instruction'], config['H'], {})
        pipe, pl = model._pipe, model._pipe.pipeline
        expert = DifferentiableExpert(pipe).snapshot_context()
        torch.manual_seed(7)
        noise = torch.randn(config['H'], 32, device='cuda', dtype=torch.bfloat16).float()
        target = pipe._noise_out.detach().float().clone()
        mask = torch.zeros(config['H'], 1, device='cuda')
        mask[:7] = 1
        def eager():
            with torch.enable_grad():
                x = noise.detach().clone().requires_grad_(True)
                delta = expert.step_delta(x, 0).float()
                estimate = x+config['N']*delta
                vjp, = torch.autograd.grad(estimate, x,
                    grad_outputs=((target-estimate).detach()*mask).detach())
                return tuple(v.detach() for v in (delta, estimate, vjp, x+delta+.5*vjp))
        for _ in range(2):
            reference = eager()
        torch.cuda.synchronize()
        result['eager_forward_vjp'] = timed(torch, eager)
        start = time.monotonic()
        captured = CapturedActionVjp(expert.step_delta, noise, step=0, num_steps=config['N'])
        result['capture_seconds'] = time.monotonic()-start
        got = tuple(v.detach().clone() for v in captured.replay(noise, target, mask))
        torch.cuda.synchronize()
        errors = [float((a-b).abs().max()) for a, b in zip(got, reference)]
        result['eager_vs_graph_max_errors_delta_estimate_vjp_update'] = errors
        result['graph_matches_eager'] = all(value <= 1e-5 for value in errors)
        result['captured_forward_vjp'] = timed(torch, lambda: captured.replay(noise, target, mask))
        # Test changing graph inputs; no fixed target baked into the capture.
        changed = captured.replay(noise, target+.5, mask)[2].detach().clone()
        result['runtime_target_changes_vjp'] = bool((changed-got[2]).abs().max() > 0)
        zero = captured.replay(noise, target, torch.zeros_like(mask))[2].detach().clone()
        result['zero_mask_vjp_max'] = float(zero.abs().max())
        # The native forward remains the original kernels. Temporarily run
        # only step0 eagerly; restore the count immediately (no source edit).
        old_n = pl.num_steps
        try:
            pl.num_steps = 1
            with torch.cuda.stream(pipe._graph_torch_stream):
                stream = pipe._graph_torch_stream.cuda_stream
                pipe._copy_tensor_to_pipeline_buf_stream(noise.to(torch.bfloat16), pl.input_noise_buf, stream)
                pl.transformer_decoder(stream=stream)
                pipe._cudart.cudaMemcpyAsync(ctypes.c_void_p(pipe._noise_out.data_ptr()),
                    pl.input_noise_buf.ptr, pipe._noise_out.numel()*2, 3, stream)
            pipe._graph_torch_stream.synchronize()
        finally:
            pl.num_steps = old_n
        native = pipe._noise_out.detach().float().clone()
        updated = (noise+got[0]).to(torch.bfloat16).float()
        result['single_step_native_forward_max_abs'] = float((updated-native).abs().max())
        result['weights_frozen'] = all(not w.requires_grad for w in expert.weights.values())
        result['weights_grads_none'] = all(w.grad is None for w in expert.weights.values())
        result.update(status='completed', torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        np.savez_compressed(root/'step.npz', noise=noise.cpu().numpy(), target=target.cpu().numpy(),
                            native=native.cpu().numpy(), eager_vjp=reference[2].cpu().numpy(),
                            graph_vjp=got[2].cpu().numpy())
    except BaseException as error:
        result.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        base._write_json_atomic(root/'summary.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
