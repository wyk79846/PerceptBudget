"""Real model single-FFN native forward/input-VJP gate. No task rollout."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from scripts.probe_rtc_step import timed
from perceptbudget.experiment import create_run, load_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fused-backward',action='store_true')
    parser.add_argument('--full-rtc-gate',action='store_true')
    parser.add_argument('--fused-attention-ops',action='store_true')
    args=parser.parse_args()
    if args.fused_attention_ops and not (args.fused_backward and args.full_rtc_gate):
        parser.error('--fused-attention-ops requires --fused-backward --full-rtc-gate')
    config = load_config(repo_root=ROOT)
    config.update(H=20, N=5, state_enabled=True)
    config['screen'].update(seed=7, init_state_index=0, suites=['libero_spatial'])
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'rtc-native-ffn-gate', repo_root=ROOT)
        print('run_dir='+str(run), flush=True)
        os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                          PYOPENGL_PLATFORM='egl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        base._patch_egl_cleanup()
        torch, _ = base._check_cuda_and_native(config)
        torch.use_deterministic_algorithms(True)
        if args.fused_backward:
            from perceptbudget.rtc_fused_backward import load
            load(config['output_root'])
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats
        from perceptbudget.rtc_expert import DifferentiableExpert
        from perceptbudget.rtc_native_probe import ffn
        from flash_rt.subgraphs.pi05.context_action import enable
        model = base._load_and_check_model(config, torch)
        enable(model)
        task = base._load_tasks([0], ['libero_spatial'])[0]
        _, env_class = base._libero_api()
        env = base._make_env(env_class, task)
        result = {'status':'running', 'scope':'single FFN; no rollout or full RTC replacement',
                  'fused_backward':args.fused_backward,'fused_attention_ops':args.fused_attention_ops}
        try:
            obs, _ = base._prepare_env_episode(env, task, np)
            stats = load_libero_norm_stats(config['norm_stats_path'])
            base._infer_actions(model, base._make_snapshot(obs, stats, True), task['instruction'],20,{})
            expert = DifferentiableExpert(model._pipe,attention_backend='flash-varlen').snapshot_context(
                fixed_prefix_capacity=True)
            pl = model._pipe.pipeline
            torch.manual_seed(7)
            x = torch.randn(20,1024,device='cuda',dtype=torch.bfloat16).requires_grad_()
            dy = torch.randn_like(x)
            ones = torch.ones(1024,device='cuda',dtype=torch.bfloat16)
            result['cases'] = []
            for step, layer in ((0,0),(2,8),(4,17)):
                style = expert._context['ffn'][step,layer].contiguous()
                weights = [expert.weights[k][layer] for k in ('ffn_gate_w','ffn_up_w','ffn_down_w')]
                def execute(native, include_residual=True):
                    y = ffn(x,style,weights,native=native,gemm=pl.gemm,fvk=pl.fvk,ones=ones,
                            include_residual=include_residual, fused_backward=args.fused_backward)
                    dx, = torch.autograd.grad(y,x,grad_outputs=dy)
                    return y, dx
                reference = tuple(t.detach().clone() for t in execute(False))
                got = tuple(t.detach().clone() for t in execute(True))
                torch.cuda.synchronize()
                row = {'step':step,'layer':layer}
                for name,a,b in zip(('output','input_gradient'),reference,got):
                    row[name] = {'max_abs':float((a.float()-b.float()).abs().max()),
                        'relative_l2':float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12)),
                        'finite':bool(torch.isfinite(b).all())}
                row['gate_passed'] = all(row[k]['finite'] and row[k]['relative_l2']<0.02
                                         for k in ('output','input_gradient'))
                # Residual identity can hide relative branch/VJP errors.
                branch_reference = tuple(t.detach().clone() for t in execute(False, False))
                branch_native = tuple(t.detach().clone() for t in execute(True, False))
                for name,a,b in zip(('branch_output','branch_input_gradient'),branch_reference,branch_native):
                    row[name] = {'max_abs':float((a.float()-b.float()).abs().max()),
                        'relative_l2':float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12)),
                        'finite':bool(torch.isfinite(b).all())}
                    row['gate_passed'] &= row[name]['finite'] and row[name]['relative_l2']<0.02
                if row['gate_passed']:
                    for native,name in ((False,'torch'),(True,'native')):
                        side = torch.cuda.Stream()
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):
                            # Create the leaf and its grad seed on the capture
                            # stream too: no legacy-stream AccumulateGrad edge.
                            x = x.detach().clone().requires_grad_()
                            dy = dy.detach().clone()
                            for _ in range(3): execute(native)
                        torch.cuda.current_stream().wait_stream(side)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=side):
                            outputs = execute(native)
                        row[name+'_graph_forward_vjp'] = timed(torch,graph.replay)
                        del graph,outputs
                result['cases'].append(row)
                base._write_json_atomic(run/'summary.json',result)
                print(json.dumps(row),flush=True)
            result['status'] = 'completed' if all(r['gate_passed'] for r in result['cases']) else 'numeric_gate_failed'
            result['weights_frozen'] = all(not w.requires_grad and w.grad is None for w in expert.weights.values())
            if args.full_rtc_gate and result['status']=='completed':
                import gc
                from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedRtcChunk
                from perceptbudget.rtc_guidance import soft_mask_tensor, paper_reference
                noise=torch.randn(20,32,device='cuda',dtype=torch.bfloat16).float()
                previous=model._pipe._noise_out[5:].detach().float().clone()
                target=torch.zeros_like(noise)
                target[:15].copy_(previous)
                mask=soft_mask_tensor(20,5,5,device=noise.device)
                full={}
                def fresh_context():
                    pipe=model._pipe
                    with torch.cuda.stream(pipe._graph_torch_stream):
                        pl._context_graph.replay(pl._graph_stream)
                    pipe._graph_torch_stream.synchronize()
                    expert.refresh_context(pl.attn.enc_K[:,:expert.context_capacity],
                        pl.attn.enc_V[:,:expert.context_capacity],int(pl.attn.dec_devpos.item()))
                for name in ('baseline','fused'):
                    if args.fused_attention_ops:
                        expert.enable_native_fused_ffn(config['output_root'])
                        if name=='fused': expert.enable_fused_attention_ops(config['output_root'])
                    elif name=='fused': expert.enable_native_fused_ffn(config['output_root'])
                    captured=CapturedRtcChunk(expert.step_delta,noise,num_steps=5)
                    got=tuple(v.detach().clone() for v in captured.replay(noise,target,mask))
                    torch.cuda.synchronize()
                    if name=='baseline': baseline=got
                    else: fused=got
                    full[name+'_expert']=timed(torch,lambda: captured.replay(noise,target,mask))
                    def all_stages():
                        fresh_context()
                        return captured.replay(noise,target,mask)
                    full[name+'_context_and_expert']=timed(torch,all_stages)
                    del captured
                    gc.collect()
                    torch.cuda.empty_cache()
                for key,a,b in zip(('raw','vjp_norm','error_norm'),baseline,fused):
                    full[key+'_max_abs_vs_baseline']=float((a-b).abs().max())
                    full[key+'_relative_l2_vs_baseline']=float((a-b).norm()/a.norm().clamp_min(1e-12))
                reference, norms=paper_reference(expert.step_delta,noise,previous,
                    delay=5,executed=5,num_steps=5)
                full['same_backend_reference_raw_max_abs']=float((reference-fused[0]).abs().max())
                full['same_backend_reference_vjp_max_abs']=float((torch.tensor(norms,device=noise.device)-fused[1]).abs().max())
                full['reference_gate_passed']=all(full[k]<=1e-4 for k in (
                    'same_backend_reference_raw_max_abs','same_backend_reference_vjp_max_abs'))
                full['baseline_equivalence_gate_passed']=all(full[k]<=1e-4 for k in (
                    'raw_max_abs_vs_baseline','vjp_norm_max_abs_vs_baseline'))
                result['full_rtc']=full
                result['note']='Fixed-input full RTC gate, no rollout; backend change is not bitwise equivalence.'
                print('full_rtc='+json.dumps(full),flush=True)
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json',result)
            env.close()
            print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
