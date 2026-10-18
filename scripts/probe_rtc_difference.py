#!/usr/bin/env python3
"""Read-only model diagnostic: isolate numerical RTC solver differences.

Reconstruct the first guided request of the saved task0 experiment. No full
rollout, no training, no changes to FlashRT or production solver defaults.
"""
import fcntl
import gc
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
from scripts import eval_seam_sync as sync

OLD_RUN = Path('/home/wyk/workspace/任务产出/perceptbudget/M1/20261018T042446.603163Z-rtc-controlled-d5-q10-460587e9')
NEW_RUN = Path('/home/wyk/workspace/任务产出/perceptbudget/M1/20261018T051321.472044Z-rtc-controlled-d5-q10-4d326938')


def solve(expert, noise, target, mask, torch, *, expression='combined',
          scale_mode='tensor', functional=False):
    x = noise.detach().float()
    beta = torch.tensor(5., device=x.device)
    trace = []
    for k in range(10):
        tau, t = k/10, 1-k/10
        def clean(a):
            d = expert.step_delta(a, k).float()
            return a+(t*10)*d if expression == 'combined' else a-t*(-10*d)
        if functional:
            with torch.no_grad():
                estimate = clean(x)
                error = ((target-estimate)*mask).detach()
                delta = expert.step_delta(x, k).float()
            with torch.enable_grad():
                _, vjp = torch.autograd.functional.vjp(clean, x, v=error)
        else:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                delta = expert.step_delta(x, k).float()
                estimate = x+(t*10)*delta if expression == 'combined' else x-t*(-10*delta)
                error = ((target-estimate).detach()*mask).detach()
                vjp, = torch.autograd.grad(estimate, x, grad_outputs=error)
        if scale_mode == 'tensor':
            coeff = beta if not k else beta.clamp(max=(tau*tau+(1-tau)**2)/(tau*(1-tau)))
        else:
            coeff = 5. if not k else min(5., (t*t+(1-t)**2)/(t*(1-t)))
        with torch.no_grad():
            nxt = (x+delta+coeff/10*vjp).detach()
        trace.append({name: value.detach().cpu().numpy().copy() for name, value in
                      [('input', x), ('delta', delta), ('estimate', estimate),
                       ('error', error), ('vjp', vjp), ('output', nxt)]})
        x = nxt
    return x.detach(), trace


def compare(a, b, np):
    rows = []
    for k, (left, right) in enumerate(zip(a, b)):
        rows.append({'step': k, **{field: float(np.max(np.abs(left[field]-right[field])))
                                   for field in left}})
    return rows


def main():
    config = load_config(repo_root=ROOT)
    config.update(H=20, N=10, state_enabled=False)
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = create_run(config, 'M1', 'rtc-difference-isolation', repo_root=ROOT)
        print('run_dir='+str(root), flush=True)
        for name in ('scripts/probe_rtc_difference.py', 'perceptbudget/rtc_expert.py',
                     'perceptbudget/rtc_guidance.py'):
            dest = root/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dest)
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
        base._patch_egl_cleanup()
        torch, _ = base._check_cuda_and_native(config)
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats, raw_libero_state
        from perceptbudget.rtc_expert import DifferentiableExpert
        from perceptbudget.rtc_guidance import soft_mask, paper_reference
        from flash_rt.subgraphs.pi05.context_action import enable
        from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedRtcChunk
        result = {'status': 'running', 'comparisons': {}}
        env = None
        try:
            stats = load_libero_norm_stats(config['norm_stats_path'])
            model = base._load_and_check_model(config, torch)
            enable(model)
            task = sync.get_tasks([0])[0]
            _, env_class = base._libero_api()
            env = sync.make_env(env_class, task)
            obs, _ = base._prepare_env_episode(env, task, np)
            base._infer_actions(model, base._make_snapshot(obs, stats, False), task['instruction'], 20, {})
            # Match the formal episode reset/RNG; warmup has no task actions.
            obs, _ = base._prepare_env_episode(env, task, np)
            base._seed_process(7, np, torch)
            actions = base._infer_actions(model, base._make_snapshot(obs, stats, False), task['instruction'], 20, {})
            pipe, pl = model._pipe, model._pipe.pipeline
            old = pipe._noise_out.float().detach().clone()
            saved_old = json.loads((OLD_RUN/'rtc/q10_requests.json').read_text())
            saved_new = json.loads((NEW_RUN/'rtc/q10_requests.json').read_text())
            result['bootstrap_max_difference'] = float(np.max(np.abs(old.cpu().numpy()-np.asarray(saved_old[0]['actions_raw']))))
            if result['bootstrap_max_difference'] != 0:
                raise RuntimeError('bootstrap does not match saved episode')
            for action in actions[:10]:
                obs, _, _, _ = env.step(action)
            result['request_state_max_difference'] = float(np.max(np.abs(raw_libero_state(obs)-np.asarray(saved_old[1]['observation_raw_state']))))
            if result['request_state_max_difference'] != 0:
                raise RuntimeError('first guided observation does not match saved episode')
            snapshot = base._make_snapshot(obs, stats, False)
            with torch.cuda.stream(pipe._graph_torch_stream):
                model.set_prompt(task['instruction'], state=None)
                pipe._fill_img_buf({'images': snapshot['images']})
                pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf, pl.input_images_buf,
                                                          pipe._graph_torch_stream.cuda_stream)
                pl._context_graph.replay(pl._graph_stream)
            pipe._graph_torch_stream.synchronize()
            expert = DifferentiableExpert(pipe).snapshot_context(fixed_prefix_capacity=True)
            noise = torch.empty((20, 32), device='cuda', dtype=torch.bfloat16).normal_().float()
            previous = old[10:]
            target = torch.zeros_like(noise)
            target[:10].copy_(previous)
            mask_old = torch.tensor(soft_mask(20, 5, 10), device='cuda')[:, None]
            i = torch.arange(20, device='cuda', dtype=torch.float32)
            c = (10-i)/6
            mask_new = torch.where(i < 5, torch.ones_like(i), torch.where(
                i < 10, c*torch.expm1(c)/(math.e-1), torch.zeros_like(i)))[:, None]
            result['mask_max_difference'] = float((mask_old-mask_new).abs().max())
            graph = CapturedRtcChunk(expert.step_delta, noise, num_steps=10)
            captured = graph.replay(noise, target, mask_old)[0].detach().cpu().numpy().copy()
            torch.cuda.synchronize()
            del graph
            gc.collect()
            torch.cuda.empty_cache()
            variants = [
                ('old_eager', mask_old, {}),
                ('mask_only', mask_new, {}),
                ('expression_only', mask_old, {'expression': 'separate'}),
                ('scale_only', mask_old, {'scale_mode': 'float'}),
                ('functional_only', mask_old, {'functional': True}),
                ('paper_eager', mask_new, {'expression': 'separate', 'scale_mode': 'float', 'functional': True}),
            ]
            traces, outputs = {}, {}
            for name, mask, options in variants:
                output, trace = solve(expert, noise, target, mask, torch, **options)
                outputs[name], traces[name] = output.cpu().numpy().copy(), trace
                np.savez_compressed(root/(name+'.npz'), **{
                    f'step{k}_{field}': value for k, row in enumerate(trace)
                    for field, value in row.items()})
                result['comparisons'][name] = {
                    'max_difference_from_captured': float(np.max(np.abs(outputs[name]-captured))),
                    'max_difference_from_old_saved': float(np.max(np.abs(outputs[name]-np.asarray(saved_old[1]['actions_raw'])))),
                    'max_difference_from_reference_saved': float(np.max(np.abs(outputs[name]-np.asarray(saved_new[1]['actions_raw'])))),
                }
                base._write_json_atomic(root/'summary.json', result)
                print(name+': '+json.dumps(result['comparisons'][name]), flush=True)
                del output
                gc.collect()
                torch.cuda.empty_cache()
            for name in traces:
                result['comparisons'][name]['step_differences_from_old_eager'] = compare(traces['old_eager'], traces[name], np)
            # Verify diagnostic replicas really correspond to both production implementations.
            reference, _ = paper_reference(expert.step_delta, noise, previous, delay=5, executed=10)
            result['paper_replica_max_difference'] = float(np.max(np.abs(reference.cpu().numpy()-outputs['paper_eager'])))
            np.savez_compressed(root/'inputs.npz', noise=noise.cpu().numpy(), target=target.cpu().numpy(),
                                old_mask=mask_old.cpu().numpy(), reference_mask=mask_new.cpu().numpy(), captured=captured)
            result['status'] = 'completed'
        except BaseException as exc:
            result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(root/'summary.json', result)
            base._save_run_status(root, result['status'])
            if env is not None:
                env.close()


if __name__ == '__main__':
    main()
