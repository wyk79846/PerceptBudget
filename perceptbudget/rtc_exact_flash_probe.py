"""Opt-in experimental exact-context FlashAttention in the frozen suffix.

No KV permutation/padding, projection/conditioning changes or missing VJP.
Attention's numerical backend changes: not a bit-exact replacement for eager.
Installed Transformers and FlashRT are never patched.
"""


def enable_exact_flash_probe(expert):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    model = expert.suffix.paligemma_with_expert.gemma_expert.model
    if model.training or any(p.requires_grad for p in expert.suffix.parameters()):
        raise ValueError('exact flash probe requires frozen eval suffix')
    if expert.context_capacity is not None or not bool(expert.prefix_mask.all()):
        raise ValueError('exact flash probe requires unpadded all-valid prefix')
    scope = model.layers[0].self_attn.forward.__func__.__globals__
    original = scope['eager_attention_forward']
    checked = set()

    def attention(module, query, key, value, attention_mask, scaling,
                  dropout=0.0, **kwargs):
        if module.training or dropout:
            raise ValueError('eval attention only')
        # Validate every layer during warmup, before capture; never inspect a
        # device value on the host inside CUDA capture.
        signature=(id(module),tuple(query.shape),tuple(key.shape))
        if signature not in checked:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('attention mask must be checked before capture')
            if attention_mask is not None and bool(torch.any(attention_mask != 0)):
                raise ValueError('cannot omit a nontrivial attention mask')
            checked.add(signature)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(query, key, value,
                attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None

    scope['eager_attention_forward'] = attention

    def restore():
        scope['eager_attention_forward'] = original
    return restore


def audit_attention_same_input(expert, *, beta=5., seed=20261020, backend='flash'):
    """Initial live KV, identical noise/old target; never consume episode RNG.

    Both paths run the full original source-order solver, changing attention
    only. This preflight is not a rollout or an all-phase quality guarantee.
    """
    if backend not in ('flash','suffix_vjp'):
        raise ValueError('unknown attention audit backend')
    def install():
        if backend=='flash':return enable_exact_flash_probe(expert)
        from perceptbudget.rtc_suffix_vjp import enable_suffix_vjp
        return enable_suffix_vjp(expert)
    return audit_expert_change(expert,install,beta=beta,seed=seed,backend=backend)


def audit_expert_change(expert,install,*,beta=5.,seed=20261020,backend):
    """Paired solver inputs for one reversible expert change; preserve RNG."""
    import torch
    from perceptbudget.openpi_reference import sample_original_rtc
    if expert.horizon < 10:
        raise ValueError('attention audit requires H>=10')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        noise=torch.randn(expert.horizon,32,device=expert.device,dtype=torch.float32)
        # The exact same old target is used by both guided solvers.
        previous=noise[5:].clone()
        cases=[('bootstrap',None,0,0),('guided',previous,5,5)]
        reference=[]
        for _,old,d,s in cases:
            raw,norms=sample_original_rtc(expert,noise,old,d,s,beta=beta,author_order=True)
            reference.append((raw.clone(),norms))
        restore=install()
        records=[]
        try:
            for (name,old,d,s),(raw,norms) in zip(cases,reference):
                got,actual=sample_original_rtc(expert,noise,old,d,s,beta=beta,author_order=True)
                delta=got-raw
                rows={}
                for label,a,b in (('raw32',raw,got),('action7_normalized',raw[:,:7],got[:,:7])):
                    diff=b-a
                    rows[label]=dict(max_abs=float(diff.abs().max()),
                        relative_l2=float(diff.norm()/a.norm().clamp_min(1e-12)),
                        rmse=float(diff.square().mean().sqrt()),
                        max_abs_per_dim=diff.abs().amax(dim=0).cpu().tolist())
                if not bool(torch.isfinite(delta).all()):
                    raise RuntimeError('nonfinite attention comparison')
                records.append(dict(case=name,delay=d,executed=s,**rows,
                    vjp_norm_max_abs=max((abs(a-b) for a,b in zip(norms,actual)),default=0.)))
        finally:
            restore()
    return dict(seed=seed,backend=backend,records=records,
        scope='initial live KV, identical FP32 noise and fixed old target; normalized action error, not physical trajectory error')
