"""Opt-in variable-length attention for the otherwise original OpenPI expert.

Changes attention backend only. FlashAttention is not bitwise equivalent to
original eager attention; this backend must pass independent gates before use.
"""


def enable_varlen_attention(expert, *, order='action-first'):
    import torch
    from perceptbudget.rtc_expert import _TorchVarlenGqaAttention
    if order not in ('action-first','context-first'):
        raise ValueError('unknown KV order')
    if expert.keys is None:
        raise ValueError('initialize a fixed-capacity or exact context before enabling varlen attention')
    if getattr(expert,'varlen_attention_enabled',False):
        raise ValueError('varlen attention already enabled')
    expert.cum_q = torch.tensor([0,expert.horizon],device=expert.device,dtype=torch.int32)
    expert.cum_k = torch.zeros(2,device=expert.device,dtype=torch.int32)
    original_refresh = expert.refresh_context

    def refresh(keys, values, valid):
        original_refresh(keys,values,valid)
        expert.cum_k[1] = expert.horizon+valid

    expert.refresh_context = refresh
    expert.cum_k[1] = expert.horizon+expert.prefix_mask.sum()

    class PrefixFirst(torch.autograd.Function):
        @staticmethod
        def forward(ctx,q,k,v):
            query=q[0].transpose(0,1).contiguous()
            key=k[0].transpose(0,1).contiguous()
            value=v[0].transpose(0,1).contiguous()
            h,total=query.shape[0],key.shape[0]
            out,lse,rng,unused,_=torch.ops.aten._flash_attention_forward(
                query,key,value,expert.cum_q,expert.cum_k,h,total,0.,False,False,
                scale=query.shape[-1]**-.5)
            ctx.save_for_backward(query,key,value,out,lse,rng,unused)
            return out.transpose(0,1).unsqueeze(0)

        @staticmethod
        def backward(ctx,gradient):
            q,k,v,out,lse,rng,unused=ctx.saved_tensors
            dq,dk,dv=torch.ops.aten._flash_attention_backward(
                gradient[0].transpose(0,1).contiguous(),q,k,v,out,lse,
                expert.cum_q,expert.cum_k,q.shape[0],k.shape[0],0.,False,rng,unused,
                scale=q.shape[-1]**-.5)
            return tuple(t.transpose(0,1).unsqueeze(0) for t in (dq,dk,dv))

    def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        if module.training or dropout:
            raise ValueError('varlen RTC attention requires frozen eval mode')
        if order=='context-first':
            if expert.context_capacity is not None:
                raise ValueError('context-first comparison requires cropped exact context')
            out=PrefixFirst.apply(query,key,value)
            return out.transpose(1,2).contiguous(),None
        capacity = expert.context_capacity or expert.prefix_mask.shape[1]
        # Original Gemma concatenates prefix then actions. Supply action first
        # only inside this permutation-invariant bidirectional attention call.
        action_k = key[0,:,capacity:,:].transpose(0,1).contiguous()
        action_v = value[0,:,capacity:,:].transpose(0,1).contiguous()
        context_k = key[0,:,:capacity,:].transpose(0,1).detach().contiguous()
        context_v = value[0,:,:capacity,:].transpose(0,1).detach().contiguous()
        out = _TorchVarlenGqaAttention.apply(query,action_k,action_v,
            context_k,context_v,expert.cum_q,expert.cum_k)
        return out.transpose(1,2).contiguous(),None

    # Isolated original-source module globals only, never installed Transformers.
    model = expert.suffix.paligemma_with_expert.gemma_expert.model
    scopes = {id(layer.self_attn.forward.__func__.__globals__):
              layer.self_attn.forward.__func__.__globals__ for layer in model.layers}
    for scope in scopes.values():
        if scope.get('__name__') != 'transformers.models.gemma.pb_original_suffix':
            raise ValueError('only isolated original-source attention may be replaced')
        scope['eager_attention_forward'] = attention
    expert.varlen_attention_enabled = True
