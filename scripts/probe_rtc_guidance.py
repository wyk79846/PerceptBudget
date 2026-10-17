#!/usr/bin/env python3
"""Single-scene genuine RTC VJP feasibility gate; never a success-rate trial."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import load_libero_norm_stats


def main():
    config = load_config(repo_root=ROOT)
    run = create_run(config, 'M1', 'rtc-vjp-gate', repo_root=ROOT)
    for name in ('scripts/probe_rtc_guidance.py', 'perceptbudget/rtc_expert.py',
                 'perceptbudget/rtc_guidance.py'):
        target = run/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    print('run_dir='+str(run), flush=True)
    sources = [Path(config['flashrt_root'])/name for name in (
        'flash_rt/frontends/torch/pi05_rtx.py', 'flash_rt/models/pi05/pipeline_rtx.py',
        'flash_rt/subgraphs/pi05/context_action.py')]
    hashes_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {'status': 'loading', 'scope': 'RTC VJP gate only; no task rollout',
              'checkpoint_backend': 'FlashRT context + project-side differentiable expert',
              'source_hashes_before': hashes_before}
    env = None
    try:
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        torch, _ = base._check_cuda_and_native(config)
        import numpy as np
        from flash_rt.subgraphs.pi05.context_action import enable
        from perceptbudget.rtc_expert import DifferentiableExpert
        from perceptbudget.rtc_guidance import guided_sample
        stats = load_libero_norm_stats(config['norm_stats_path'])
        model = base._load_and_check_model(config, torch)
        enable(model)
        task = base._load_tasks([0], config['screen']['suites'])[0]
        _, env_class = base._libero_api()
        env = base._make_env(env_class, task)
        observation, _ = base._prepare_env_episode(env, task, np)
        snapshot = base._make_snapshot(observation, stats, config['state_enabled'])
        base._infer_actions(model, snapshot, task['instruction'], config['H'], {})
        pipe, pl = model._pipe, model._pipe.pipeline
        expert = DifferentiableExpert(pipe)
        expert.snapshot_context()
        result['expert_weights_frozen'] = all(not w.requires_grad for w in expert.weights.values())
        result['expert_weights_shared_storage'] = all(
            w.data_ptr() == pipe._ckpt_bf16['decoder_'+name].data_ptr()
            for name, w in expert.weights.items())
        torch.cuda.synchronize()
        torch.manual_seed(7)
        noise = torch.randn(config['H'], 32, device='cuda', dtype=torch.bfloat16)
        # Replay existing decode-only graph on the exact same noise/context.
        start = time.monotonic()
        with torch.cuda.stream(pipe._graph_torch_stream):
            pipe._copy_tensor_to_pipeline_buf_stream(noise, pl.input_noise_buf,
                                                      pipe._graph_torch_stream.cuda_stream)
            pl._decoder_only_graph.replay(pl._graph_stream)
            pipe._cudart.cudaMemcpyAsync(ctypes.c_void_p(pipe._noise_out.data_ptr()),
                pl.input_noise_buf.ptr, pipe._noise_out.numel()*2, 3,
                pipe._graph_torch_stream.cuda_stream)
        pipe._graph_torch_stream.synchronize()
        reference = pipe._noise_out.detach().float().clone()
        result['flashrt_decode_seconds'] = time.monotonic()-start
        # Same N, style table, initial noise, and BF16 residual rounding.
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.no_grad():
            x = noise
            for step in range(config['N']):
                delta = expert.step_delta(x, step)
                x = (x.float()+delta.float()).to(torch.bfloat16)
        torch.cuda.synchronize()
        diff = x.float()-reference
        result.update(expert_decode_seconds=time.monotonic()-start,
                      forward_max_abs=float(diff.abs().max()),
                      forward_rmse=float(diff.square().mean().sqrt()),
                      forward_relative_rmse=float(diff.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-9)),
                      forward_arm7_relative_rmse=float(diff[:, :7].square().mean().sqrt()/reference[:, :7].square().mean().sqrt().clamp_min(1e-9)),
                      forward_finite=bool(torch.isfinite(x).all()))
        # This threshold is a numerical gate, not a task-quality guarantee.
        result['forward_gate'] = (result['forward_finite'] and result['forward_relative_rmse'] <= .02
                                  and result['forward_arm7_relative_rmse'] <= .02)
        np.savez_compressed(run/'forward.npz', noise=noise.float().cpu().numpy(),
                            reference=reference.cpu().numpy(), expert=x.float().cpu().numpy())
        base._write_json_atomic(run/'summary.json', result)
        if not result['forward_gate']:
            result['status'] = 'forward_mismatch_no_rollout'
            return
        start = time.monotonic()
        guided, records = guided_sample(expert.step_delta, noise, reference[8:].detach(),
                                         delay=7, executed=8, num_steps=config['N'], beta=5.)
        torch.cuda.synchronize()
        result.update(status='vjp_probe_completed', rtc_expert_seconds=time.monotonic()-start,
                      rtc_finite=bool(torch.isfinite(guided).all()),
                      per_step_vjp=records, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      note='Fixed same-scene synthetic previous chunk; feasibility only, not a valid async task result')
        np.savez_compressed(run/'raw.npz', noise=noise.float().cpu().numpy(),
                            reference=reference.cpu().numpy(), expert=x.float().cpu().numpy(),
                            guided=guided.cpu().numpy())
    except BaseException as error:
        result.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['source_files_unchanged'] = all(
            hashlib.sha256(p.read_bytes()).hexdigest() == hashes_before[str(p)] for p in sources)
        base._write_json_atomic(run/'summary.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
