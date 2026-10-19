"""Independent OpenPI expert from original unfused checkpoint and source.

Reuses recorded prefix KV to isolate expert correctness, not encoder parity.
No FlashRT runtime or project expert is imported by this reference process.
"""
import ast
import argparse
import fcntl
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-reference', type=Path)
    parser.add_argument('--author-euler', action='store_true', help='Preserve author x+dt*(v+guidance) evaluation order')
    parser.add_argument('--reference-variant', choices=('original','folded-output','bf16-input','bf16-time','split-input-bias'), default='original')
    args = parser.parse_args()
    config = load_config(repo_root=ROOT)
    source = Path(config['output_root'])/'M1/20261019T054917.649092Z-capacity-numeric-diagnosis-5f903047/capacity80a'
    config.update(H=20, N=5, reference_trace=str(source), reference_variant=args.reference_variant,
                  author_euler_order=args.author_euler)
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'openpi-expert-independent', repo_root=ROOT)
        shutil.copy2(__file__, run/'audit_openpi_expert.py')
        print('run_dir='+str(run), flush=True)
        if args.project_reference:
            return project_check(config, source, run, args.project_reference)
        import torch
        import numpy as np
        from safetensors import safe_open
        from transformers import GemmaConfig
        from transformers.cache_utils import DynamicCache
        source_file = Path(config['openpi_root'])/'src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py'
        spec = importlib.util.spec_from_file_location('transformers.models.gemma.pb_openpi_reference', source_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cfg = GemmaConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=18,
                          num_attention_heads=8, num_key_value_heads=1, head_dim=256,
                          hidden_activation='gelu_pytorch_tanh', vocab_size=1,
                          use_adarms=True, adarms_cond_dim=1024)
        cfg._attn_implementation = 'eager'
        with torch.device('meta'):
            expert = module.GemmaModel(cfg)
        expert.embed_tokens = None
        expert.rotary_emb = module.GemmaRotaryEmbedding(cfg, device='cuda')
        with safe_open(config['model_path'], framework='pt', device='cpu') as f:
            prefix = 'paligemma_with_expert.gemma_expert.model.'
            weights = {k[len(prefix):]:f.get_tensor(k).to(
                device='cuda', dtype=torch.float32 if 'layernorm' in k or '.model.norm.' in k else torch.bfloat16)
                for k in f.keys() if k.startswith(prefix) and 'embed_tokens' not in k}
            expert.load_state_dict(weights, assign=True, strict=True)
            del weights
            def linear(name):
                w, b = f.get_tensor(name+'.weight'), f.get_tensor(name+'.bias')
                obj = torch.nn.Linear(w.shape[1], w.shape[0], device='meta')
                # PI0Pytorch owns these Linear modules in FP32 outside the
                # PaliGemma BF16 conversion. Normal load_state_dict casts into
                # their parameter dtype; meta assign must reproduce that.
                obj.load_state_dict({'weight':w.to(device='cuda',dtype=torch.float32),
                                     'bias':b.to(device='cuda',dtype=torch.float32)}, assign=True)
                return obj
            action_in, action_out, time_in, time_out = [linear(n) for n in
                ('action_in_proj','action_out_proj','time_mlp_in','time_mlp_out')]
        for obj in (expert, action_in, action_out, time_in, time_out):
            obj.requires_grad_(False)
        # Execute original time-embedding function without importing full OpenPI/JAX.
        tree = ast.parse((Path(config['openpi_root'])/'src/openpi/models_pytorch/pi0_pytorch.py').read_text())
        nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
                 ('get_safe_dtype','create_sinusoidal_pos_embedding')]
        scope = dict(torch=torch, Tensor=torch.Tensor, math=math)
        exec(compile(ast.Module(body=nodes,type_ignores=[]), '<openpi-time-functions>', 'exec'), scope)
        trace = np.load(source/'trace.npz')
        def from_bits(array):
            return torch.from_numpy(array.copy()).to('cuda').view(torch.bfloat16)
        # FlashRT permutes Q/K weight rows from split-half RoPE to adjacent
        # pairs. Undo that basis permutation on recorded rotated prefix K.
        keys = [from_bits(trace[f'21_encoder_layer{i:02d}_key']).reshape(-1,1,128,2)
                .transpose(-1,-2).flatten(-2).permute(1,0,2).unsqueeze(0) for i in range(18)]
        values = [from_bits(trace[f'21_encoder_layer{i:02d}_value']).permute(1,0,2).unsqueeze(0) for i in range(18)]
        valid = keys[0].shape[-2]
        positions = torch.arange(valid, valid+20, device='cuda').unsqueeze(0)
        mask = torch.zeros(1,1,20,valid+20,device='cuda',dtype=torch.float32)
        outputs = {}
        active_step = [-1]
        hooks = []
        def capture(label):
            def hook(mod, inputs, output):
                if active_step[0] != 0:
                    return
                value = output[0] if isinstance(output,tuple) else output
                outputs[label] = value.detach().float().cpu().numpy()
            return hook
        for index,layer in enumerate(expert.layers):
            hooks.append(layer.register_forward_hook(capture(f'layer{index:02d}')))
            if index == 0:
                for name in ('input_layernorm','self_attn','post_attention_layernorm','mlp'):
                    hooks.append(getattr(layer,name).register_forward_hook(capture('layer00_'+name)))
        if args.reference_variant == 'bf16-time':
            time_in.to(torch.bfloat16); time_out.to(torch.bfloat16)
        if args.reference_variant == 'bf16-input':
            action_in.to(torch.bfloat16)
        folded_w = (action_out.weight.detach()*(-1/5)).to(torch.bfloat16)
        folded_b = (action_out.bias.detach()*(-1/5)).to(torch.bfloat16)
        def velocity(x,k):
            active_step[0] = k
            t = torch.tensor([1-k/5],device='cuda',dtype=torch.float32)
            emb = scope['create_sinusoidal_pos_embedding'](t,1024,4e-3,4.,device=t.device).to(t.dtype)
            emb = emb.to(time_in.weight.dtype)
            cond = torch.nn.functional.silu(time_out(torch.nn.functional.silu(time_in(emb))))
            cond = cond.float()
            hidden = action_in(x.unsqueeze(0).to(action_in.weight.dtype))
            if args.reference_variant == 'split-input-bias':
                hidden = ((x.unsqueeze(0).bfloat16() @ action_in.weight.bfloat16().t()).float()+
                          action_in.bias.bfloat16().float()).bfloat16()
            if k == 0:
                outputs['action_projection_bf16']=hidden.detach().bfloat16().float().cpu().numpy()
                outputs['time_condition']=cond.detach().float().cpu().numpy()
            cache = DynamicCache()
            for i in range(18): cache.update(keys[i],values[i],i)
            hidden = expert(inputs_embeds=hidden, attention_mask=mask, position_ids=positions,
                            past_key_values=cache, use_cache=False, adarms_cond=cond).last_hidden_state
            if args.reference_variant == 'folded-output':
                delta = ((hidden.to(torch.bfloat16) @ folded_w.t()).float()+folded_b.float()).to(torch.bfloat16)
                return (-5*delta.float()).squeeze(0)
            return action_out(hidden.float()).squeeze(0)
        noise = torch.as_tensor(trace['32_fixed_noise'],device='cuda')
        target = torch.zeros_like(noise); target[:15] = noise[5:]
        # Author mask direct expression, not project helper.
        i = torch.arange(20,device='cuda',dtype=torch.float32)
        w = ((5-1-i)/(15-5+1)+1).clamp(0,1)
        w = torch.where(i>=15,0,w*torch.expm1(w)/(math.e-1))[:,None]
        x = noise.clone()
        for k in range(5):
            x = x.detach().requires_grad_(True)
            v = velocity(x,k)
            clean = x-(1-k/5)*v
            error = ((target-clean)*w).detach()
            g, = torch.autograd.grad(clean,x,grad_outputs=error)
            beta = 5 if k==0 else min(5,((k/5)**2+(1-k/5)**2)/((k/5)*(1-k/5)))
            outputs[f'velocity{k}']=v.detach().cpu().numpy()
            outputs[f'vjp{k}']=g.detach().cpu().numpy()
            outputs[f'input{k}']=x.detach().cpu().numpy()
            outputs[f'error{k}']=error.detach().cpu().numpy()
            x = ((x+(1/5)*(-v+beta*g)) if args.author_euler else (x-v/5+beta*g/5)).detach()
            print('step='+str(k)+' finite='+str(bool(torch.isfinite(x).all())),flush=True)
        outputs['raw']=x.cpu().numpy()
        fixed_reference=Path(config['output_root'])/'M1/20261019T090506.707357Z-openpi-expert-independent-4bfa20b7/reference.npz'
        with np.load(fixed_reference) as fixed:
            fixed_x=torch.as_tensor(fixed['input0'],device='cuda').requires_grad_(True)
            fixed_error=torch.as_tensor(fixed['error0'],device='cuda')
            fixed_v=velocity(fixed_x,0)
            fixed_g,=torch.autograd.grad(fixed_x-fixed_v,fixed_x,grad_outputs=fixed_error)
            outputs['fixed_velocity0']=fixed_v.detach().cpu().numpy()
            outputs['fixed_vjp0']=fixed_g.detach().cpu().numpy()
        np.savez_compressed(run/'reference.npz',**outputs)
        difference = outputs['raw']-trace['33_rtc_raw']
        result=dict(status='completed', scope='independent original OpenPI expert, shared recorded encoder KV; not full model parity',
            raw_max_abs=float(np.max(np.abs(difference))), raw_relative_l2=float(np.linalg.norm(difference)/np.linalg.norm(trace['33_rtc_raw'])),
            variant=args.reference_variant,
            author_euler_order=args.author_euler,
            all_weights_frozen=all(not p.requires_grad for p in expert.parameters()), reference_source=str(source_file))
        base._write_json_atomic(run/'summary.json', result)
        print(json.dumps(result),flush=True)


def project_check(config, source, run, reference):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch, _ = base._check_cuda_and_native(config)
    torch.use_deterministic_algorithms(True)
    import numpy as np
    from perceptbudget.rtc_expert import DifferentiableExpert
    from flash_rt.subgraphs.pi05.context_action import enable
    config.update(state_prompt_capacity=80,native_gemm_policy='heuristic-first')
    fixture = source.parent/'fixture.npz'
    with np.load(fixture) as f:
        snapshot=dict(images=[f['image0'].copy(),f['image1'].copy()],state=f['state'].copy())
    prompt=json.loads((source.parent/'fixture.json').read_text())['prompt']
    model=base._load_and_check_model(config,torch)
    enable(model)
    base._infer_actions(model,snapshot,prompt,20,{})
    expert=DifferentiableExpert(model._pipe,attention_backend='flash-varlen').snapshot_context(fixed_prefix_capacity=True)
    expert.enable_fused_attention_ops(config['output_root'])
    trace=np.load(source/'trace.npz')
    valid=int(json.loads((source/'summary.json').read_text())['valid_prefix'])
    kv_equal=all(np.array_equal(expert._context[name][i,:valid].contiguous().view(torch.uint16).cpu().numpy(),
        trace[f'21_encoder_layer{i:02d}_{suffix}']) for name,suffix in (('key','key'),('value','value')) for i in range(18))
    if not kv_equal:
        raise RuntimeError('reference prefix KV differs from current project KV; invalid comparison')
    saved=np.load(reference/'reference.npz')
    metrics=[]
    project_stages={}
    import inspect
    lines, first_line=inspect.getsourcelines(expert._step_delta)
    source_lines={first_line+i:s.strip() for i,s in enumerate(lines)}
    def tracer(frame,event,arg):
        if frame.f_code is not expert._step_delta.__func__.__code__:
            return None
        if event=='line':
            line=source_lines.get(frame.f_lineno,'')
            loc=frame.f_locals
            label=None
            if line=='for i in range(18):':
                label=f'layer{loc["i"]:02d}' if 'i' in loc else 'action_projection_bf16'
                value=loc['hidden']
            elif line=="qkv = norm @ w['attn_qkv_w'][i]" and loc['i']==0:
                label='layer00_input_layernorm'; value=loc['norm']
            if label:
                project_stages[label]=value.detach().float().cpu().numpy()
        return tracer
    for mode in ('fused','unfused','reference-projections'):
        expert.fused_attention_ops = mode=='fused'
        expert.native_fused_ffn = mode=='fused'
        if mode=='reference-projections':
            expert.enable_reference_projections(config['model_path'])
        for k in range(5):
            x=torch.as_tensor(saved[f'input{k}'],device='cuda').requires_grad_(True)
            trace_active=mode=='unfused' and k==0
            if trace_active: sys.settrace(tracer)
            try:
                delta=expert.step_delta(x,k).float()
            finally:
                if trace_active: sys.settrace(None)
            velocity=-5*delta
            estimate=x-(1-k/5)*velocity
            error=torch.as_tensor(saved[f'error{k}'],device='cuda')
            g,=torch.autograd.grad(estimate,x,grad_outputs=error)
            row=dict(step=k,mode=mode)
            for name,value in (('velocity',velocity),('vjp',g)):
                a=value.detach().cpu().numpy(); b=saved[f'{name}{k}']; diff=a-b
                row[name]=dict(max_abs=float(np.max(np.abs(diff))),relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(b),1e-12)),
                               project_norm=float(np.linalg.norm(a)), reference_norm=float(np.linalg.norm(b)))
            # clean=x-t*v has cancellation: inspect underlying velocity VJP
            # too, without labeling a large relative correction error a bug.
            a=(saved[f'error{k}']-g.detach().cpu().numpy())/(1-k/5)
            b=(saved[f'error{k}']-saved[f'vjp{k}'])/(1-k/5)
            row['velocity_vjp_recovered']=dict(relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)))
            metrics.append(row)
            print(json.dumps(row),flush=True)
    np.savez_compressed(run/'project_stages.npz',**project_stages)
    base._write_json_atomic(run/'summary.json',dict(status='completed',reference=str(reference),
        prefix_kv_exact=kv_equal,
        scope='same independent input and residual per step; original OpenPI vs fused FlashRT expert',metrics=metrics))


if __name__ == '__main__':
    main()
