#!/usr/bin/env python3
"""Full N-step RTC graph timing gate, fixed scene; no task rollout."""

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


def main():
    config = load_config(repo_root=ROOT)
    root = create_run(config, 'M1', 'rtc-full-graph', repo_root=ROOT)
    print('run_dir='+str(root), flush=True)
    result = {'status': 'loading', 'scope': 'full N-step generation gate, no task rollout',
              'backend': 'captured differentiable expert, not native FlashRT backward kernels',
              'H': config['H'], 'N': config['N'], 'delay_test': 7, 'executed_test': 8}
    env = None
    sources = [(ROOT, name) for name in ('scripts/probe_rtc_chunk.py', 'scripts/probe_rtc_step.py',
               'perceptbudget/rtc_expert.py', 'perceptbudget/rtc_guidance.py')]
    sources += [(Path(config['flashrt_root']), 'flash_rt/subgraphs/pi05/rtc_action_vjp.py')]
    for repo, name in sources:
        target_path = root/'source'/name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, target_path)
    result['flashrt_revision'] = subprocess.check_output(
        ['git', '-C', config['flashrt_root'], 'rev-parse', 'HEAD'], text=True).strip()
    def save():
        base._write_json_atomic(root/'summary.json', result)
    try:
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        torch, _ = base._check_cuda_and_native(config)
        import numpy as np
        from flash_rt.subgraphs.pi05.context_action import enable
        from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedRtcChunk
        from perceptbudget.rtc_expert import DifferentiableExpert
        from perceptbudget.rtc_guidance import guided_sample, soft_mask
        stats = load_libero_norm_stats(config['norm_stats_path'])
        model = base._load_and_check_model(config, torch)
        enable(model)
        task = base._load_tasks([0], config['screen']['suites'])[0]
        _, env_class = base._libero_api()
        env = base._make_env(env_class, task)
        obs, _ = base._prepare_env_episode(env, task, np)
        snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
        def baseline():
            return base._infer_actions(model, snapshot, task['instruction'], config['H'], {})
        baseline()
        result['flashrt_normal_full_inference'] = timed(torch, baseline, count=3)
        pipe, pl = model._pipe, model._pipe.pipeline
        expert = DifferentiableExpert(pipe).snapshot_context()
        torch.manual_seed(7)
        noise = torch.randn(config['H'], 32, device='cuda', dtype=torch.bfloat16).float()
        previous = pipe._noise_out[8:].detach().float().clone()
        target = torch.zeros_like(noise)
        target[:len(previous)] = previous
        mask = torch.tensor(soft_mask(config['H'], 7, 8), device='cuda')[:, None]
        result['status'] = 'eager_reference'
        save()
        start = time.monotonic()
        reference, records = guided_sample(expert.step_delta, noise, previous,
            delay=7, executed=8, num_steps=config['N'], beta=5.)
        torch.cuda.synchronize()
        result['eager_full_rtc_seconds'] = time.monotonic()-start
        result['status'] = 'capturing_full_graph'
        save()
        start = time.monotonic()
        captured = CapturedRtcChunk(expert.step_delta, noise, num_steps=config['N'])
        result['capture_seconds'] = time.monotonic()-start
        output, norms, errors = (v.detach().clone() for v in captured.replay(noise, target, mask))
        torch.cuda.synchronize()
        result['raw_max_abs_difference_vs_eager'] = float((output-reference).abs().max())
        result['vjp_norm_max_difference_vs_eager'] = float((norms-torch.tensor(
            [r['vjp_norm'] for r in records], device='cuda')).abs().max())
        result['all_steps_vjp_norms'] = norms.cpu().tolist()
        result['numerical_gate'] = (bool(torch.isfinite(output).all())
            and result['raw_max_abs_difference_vs_eager'] <= 1e-5
            and result['vjp_norm_max_difference_vs_eager'] <= 1e-5)
        if not result['numerical_gate']:
            result['status'] = 'numerical_gate_failed_no_rollout'
            return
        captured.replay(noise, target, mask)
        torch.cuda.synchronize()
        result['full_rtc_expert_warm'] = timed(torch, lambda: captured.replay(noise, target, mask))
        result['status'] = 'measuring_fresh_context'
        save()
        # Same shape and observation, but actually recompute vision/language
        # every trial and copy KV into graph-owned stable addresses. Never
        # replace the context dict while a captured graph points to its tensors.
        def fresh_context():
            with torch.cuda.stream(pipe._graph_torch_stream):
                model.set_prompt(task['instruction'], state=snapshot.get('state'))
                pipe._fill_img_buf({'images': snapshot['images']})
                pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf, pl.input_images_buf,
                                                          pipe._graph_torch_stream.cuda_stream)
                pl._context_graph.replay(pl._graph_stream)
            pipe._graph_torch_stream.synchronize()
            prefix = int(pl.attn.dec_devpos.item()) if pl._fixed_shape else int(pl.encoder_seq_len)
            if prefix != expert.prefix_length:
                raise RuntimeError('context length changed: recapture required')
            with torch.no_grad():
                expert._context['key'].copy_(pl.attn.enc_K[:, :prefix])
                expert._context['value'].copy_(pl.attn.enc_V[:, :prefix])
            return captured.replay(noise, target, mask)
        fresh_context()
        torch.cuda.synchronize()
        result['full_context_and_rtc_warm'] = timed(torch, fresh_context)
        after = captured.replay(noise, target, mask)[0].detach().clone()
        result['fresh_context_same_scene_output_max_diff'] = float((after-output).abs().max())
        result['weights_frozen'] = all(not w.requires_grad and w.grad is None for w in expert.weights.values())
        result['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        duration = result['full_context_and_rtc_warm']['median_ms']/1000
        result['nominal_delay_steps_at_20hz'] = int(np.ceil(duration/.05))
        result['nominal_H_seconds'] = config['H']*.05
        result['fits_whole_H_window'] = duration <= config['H']*.05
        result['fits_RTC_d_le_s_le_H_minus_d'] = 2*result['nominal_delay_steps_at_20hz'] <= config['H']
        result['status'] = 'completed'
        np.savez_compressed(root/'full.npz', noise=noise.cpu().numpy(), target=target.cpu().numpy(),
            mask=mask.cpu().numpy(), eager=reference.cpu().numpy(), graph=output.cpu().numpy(),
            vjp_norms=norms.cpu().numpy(), error_norms=errors.cpu().numpy())
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
