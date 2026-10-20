"""Experimental eager attention: omit discarded *constant-prefix* KV gradients.

Forward operations, softmax and query gradients retain the original eager
implementation. Only dK/dV GEMMs are restricted to action rows; the prefix is
detached in OriginalOpenPiExpert. This is NOT permission to omit action VJP.
Different GEMM shapes can round differently: require a live full-solver gate.
Only an explicit diagnostic flag enables this alternative backend. It does
not combine with Flash SDPA, which replaces these same matmuls and softmax.
"""
import torch


class _SuffixQK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, suffix_length):
        ctx.save_for_backward(query, key)
        ctx.suffix_length = suffix_length
        return torch.matmul(query, key.transpose(-2, -1))

    @staticmethod
    def backward(ctx, grad):
        query, key = ctx.saved_tensors
        n = ctx.suffix_length
        dq = torch.matmul(grad, key)
        dk = torch.zeros_like(key)
        dk[..., -n:, :] = torch.matmul(grad[..., -n:].transpose(-2, -1), query)
        return dq, dk, None


class _SuffixAV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, value, suffix_length):
        ctx.save_for_backward(weights, value)
        ctx.suffix_length = suffix_length
        return torch.matmul(weights, value)

    @staticmethod
    def backward(ctx, grad):
        weights, value = ctx.saved_tensors
        n = ctx.suffix_length
        dw = torch.matmul(grad, value.transpose(-2, -1))
        dv = torch.zeros_like(value)
        dv[..., -n:, :] = torch.matmul(weights[..., -n:].transpose(-2, -1), grad)
        return dw, dv, None


def enable_suffix_vjp(expert, mode='both'):
    """Patch only the isolated source module; return an explicit restore hook."""
    if mode not in ('both', 'qk', 'av'):
        raise ValueError('unknown suffix VJP mode')
    model = expert.suffix.paligemma_with_expert.gemma_expert.model
    if model.training or any(p.requires_grad for p in expert.suffix.parameters()):
        raise ValueError('suffix VJP requires frozen eval expert')
    if expert.keys.requires_grad or expert.values.requires_grad:
        raise ValueError('prefix must be constant')
    scope = model.layers[0].self_attn.forward.__func__.__globals__
    original = scope['eager_attention_forward']
    repeat = scope['repeat_kv']

    def attention(module, query, key, value, attention_mask, scaling,
                  dropout=0.0, **kwargs):
        if module.training or dropout or not key.shape[-2] > query.shape[-2]:
            raise ValueError('suffix VJP only supports eval prefix+action attention')
        k = repeat(key, module.num_key_value_groups)
        v = repeat(value, module.num_key_value_groups)
        n = query.shape[-2]
        weights = (_SuffixQK.apply(query, k, n) if mode in ('both','qk')
                   else torch.matmul(query, k.transpose(-2,-1))) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :k.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
        output = (_SuffixAV.apply(weights, v, n) if mode in ('both','av')
                  else torch.matmul(weights,v))
        return output.transpose(1, 2).contiguous(), weights

    scope['eager_attention_forward'] = attention

    def restore():
        scope['eager_attention_forward'] = original
    return restore
