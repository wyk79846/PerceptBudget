"""Probe-only exact-context FlashAttention in the original frozen suffix.

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
        if id(module) not in checked:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('attention mask must be checked before capture')
            if attention_mask is not None and bool(torch.any(attention_mask != 0)):
                raise ValueError('cannot omit a nontrivial attention mask')
            checked.add(id(module))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(query, key, value,
                attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None

    scope['eager_attention_forward'] = attention

    def restore():
        scope['eager_attention_forward'] = original
    return restore
