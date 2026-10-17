#!/usr/bin/env python3
"""Compare math and Flash SDPA inside the opt-in full RTC CUDA graph.

Uses one LIBERO observation and a synthetic d=7/s=8 old-action target. This is
a numerical/performance gate only; it does not run a task or change defaults.
"""

import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from scripts.probe_rtc_step import timed
from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import load_libero_norm_stats


def _relative_rmse(actual, reference):
    import torch

    actual = actual.float()
    reference = reference.float()
    return float((actual - reference).square().mean().sqrt()
                 / reference.square().mean().sqrt().clamp_min(1e-12))


def main():
    config = load_config(repo_root=ROOT)
    root = create_run(config, 'M1', 'rtc-sdpa-backend', repo_root=ROOT)
    print('run_dir=' + str(root), flush=True)
    result = {
        'status': 'loading',
        'scope': 'math vs Flash SDPA full RTC graph, one fixed observation; no task rollout',
        'defaults_changed': False,
        'attention_backends': ['math', 'flash'],
        'H': config['H'],
        'N': config['N'],
        'delay_test': 7,
        'executed_test': 8,
    }
    env = None
    flashrt_root = Path(config['flashrt_root'])
    for repo, name in [
        (ROOT, 'scripts/probe_rtc_attention_backend.py'),
        (ROOT, 'scripts/probe_rtc_chunk.py'),
        (ROOT, 'scripts/probe_rtc_step.py'),
        (ROOT, 'perceptbudget/rtc_expert.py'),
        (ROOT, 'perceptbudget/rtc_guidance.py'),
        (flashrt_root, 'flash_rt/subgraphs/pi05/rtc_action_vjp.py'),
    ]:
        target = root / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    result['flashrt_revision'] = subprocess.check_output(
        ['git', '-C', str(flashrt_root), 'rev-parse', 'HEAD'], text=True
    ).strip()

    def save():
        base._write_json_atomic(root / 'summary.json', result)

    try:
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        torch, _ = base._check_cuda_and_native(config)
        import numpy as np
        from flash_rt.subgraphs.pi05.context_action import enable
        from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedRtcChunk
        from perceptbudget.rtc_expert import DifferentiableExpert
        from perceptbudget.rtc_guidance import soft_mask

        stats = load_libero_norm_stats(config['norm_stats_path'])
        model = base._load_and_check_model(config, torch)
        enable(model)
        task = base._load_tasks([0], config['screen']['suites'])[0]
        _, env_class = base._libero_api()
        env = base._make_env(env_class, task)
        obs, _ = base._prepare_env_episode(env, task, np)
        snapshot = base._make_snapshot(obs, stats, config['state_enabled'])

        def vanilla_decode():
            return base._infer_actions(model, snapshot, task['instruction'], config['H'], {})

        vanilla_decode()
        result['flashrt_vanilla_full_inference'] = timed(torch, vanilla_decode, count=3)
        pipe, pl = model._pipe, model._pipe.pipeline
        expert = DifferentiableExpert(pipe, attention_backend='math').snapshot_context()

        torch.manual_seed(7)
        noise = torch.randn(config['H'], 32, device='cuda', dtype=torch.bfloat16).float()
        previous = pipe._noise_out[8:].detach().float().clone()
        target = torch.zeros_like(noise)
        target[:len(previous)] = previous
        mask = torch.tensor(soft_mask(config['H'], 7, 8), device='cuda')[:, None]

        references = {}
        for name in ('math', 'flash'):
            torch.cuda.reset_peak_memory_stats()
            expert.set_attention_backend(name)
            result['status'] = 'capturing_' + name
            save()
            start = time.monotonic()
            captured = CapturedRtcChunk(expert.step_delta, noise,
                                        num_steps=config['N'])
            entry = {'capture_seconds': time.monotonic() - start}

            outputs = tuple(value.detach().clone()
                            for value in captured.replay(noise, target, mask))
            torch.cuda.synchronize()
            entry['finite_outputs'] = all(bool(torch.isfinite(value).all())
                                          for value in outputs)
            entry['expert_graph_warm'] = timed(
                torch, lambda: captured.replay(noise, target, mask), count=5)

            def fresh_context_and_rtc():
                with torch.cuda.stream(pipe._graph_torch_stream):
                    model.set_prompt(task['instruction'], state=snapshot.get('state'))
                    pipe._fill_img_buf({'images': snapshot['images']})
                    pipe._copy_tensor_to_pipeline_buf_stream(
                        pipe._img_buf, pl.input_images_buf,
                        pipe._graph_torch_stream.cuda_stream)
                    pl._context_graph.replay(pl._graph_stream)
                pipe._graph_torch_stream.synchronize()
                prefix = (int(pl.attn.dec_devpos.item()) if pl._fixed_shape
                          else int(pl.encoder_seq_len))
                if prefix != expert.prefix_length:
                    raise RuntimeError('context length changed; graph recapture required')
                with torch.no_grad():
                    expert._context['key'].copy_(pl.attn.enc_K[:, :prefix])
                    expert._context['value'].copy_(pl.attn.enc_V[:, :prefix])
                return captured.replay(noise, target, mask)

            entry['fresh_context_and_rtc'] = timed(
                torch, fresh_context_and_rtc, count=5)
            refreshed = tuple(value.detach().clone()
                              for value in fresh_context_and_rtc())
            entry['same_scene_refresh_max_abs'] = float(
                (refreshed[0] - outputs[0]).abs().max())
            entry['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            references[name] = tuple(value.detach().cpu() for value in outputs)
            result[name] = entry
            save()

            del captured
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        math_raw, math_vjp_norms, math_error_norms = references['math']
        flash_raw, flash_vjp_norms, flash_error_norms = references['flash']
        result['flash_vs_math'] = {
            'raw_max_abs': float((flash_raw - math_raw).abs().max()),
            'raw_relative_rmse': _relative_rmse(flash_raw, math_raw),
            'vjp_norms_max_abs': float((flash_vjp_norms - math_vjp_norms).abs().max()),
            'weighted_error_norms_max_abs': float(
                (flash_error_norms - math_error_norms).abs().max()),
            'all_finite': bool(torch.isfinite(flash_raw).all()
                               and torch.isfinite(flash_vjp_norms).all()),
        }
        result['weights_frozen'] = all(
            not weight.requires_grad and weight.grad is None
            for weight in expert.weights.values())
        result['status'] = 'completed'
        np.savez_compressed(
            root / 'comparison.npz', math_raw=math_raw.numpy(),
            flash_raw=flash_raw.numpy(), math_vjp_norms=math_vjp_norms.numpy(),
            flash_vjp_norms=flash_vjp_norms.numpy(),
            math_error_norms=math_error_norms.numpy(),
            flash_error_norms=flash_error_norms.numpy())
    except BaseException as error:
        result.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
