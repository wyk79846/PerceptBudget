"""Exact frozen time-only linear results; no attention or RTC math changes."""


def enable_condition_cache(expert, *, graph_live=False):
    import torch
    suffix=expert.suffix
    if getattr(expert,'_condition_cache',None) is not None:
        raise ValueError('conditioning cache already installed')
    if not suffix.pi05 or any(p.requires_grad for p in suffix.parameters()) or suffix.training:
        raise ValueError('conditioning cache requires frozen eval pi05')
    model=suffix.paligemma_with_expert.gemma_expert.model
    modules=[suffix.time_mlp_in,suffix.time_mlp_out]
    for layer in model.layers:
        modules.extend((layer.input_layernorm.dense,layer.post_attention_layernorm.dense))
    modules.append(model.norm.dense)
    if any(m is None for m in modules):
        raise ValueError('expected pi05 adaptive normalization')
    cache={id(m):dict(outputs=[],inputs=[],versions=tuple(p._version for p in m.parameters()),
                      original_forward=m.forward,module=m) for m in modules}
    with torch.no_grad():
        dummy=torch.zeros(1,expert.horizon,32,device=expert.device,dtype=torch.float32)
        for step in range(expert.num_steps):
            handles=[]
            def record(module,args,out):
                slot=cache[id(module)]
                slot['inputs'].append((args[0].shape,args[0].dtype,args[0].device))
                slot['outputs'].append(out.detach().clone())
            for module in modules[:2]:
                handles.append(module.register_forward_hook(record))
            try:
                _,_,_,cond=suffix.embed_suffix(None,dummy,expert.times[step])
            finally:
                for handle in handles:
                    handle.remove()
            for module in modules[2:]:
                record(module,(cond,),module(cond))
    if graph_live:
        if any(x.dtype!=torch.float32 for slot in cache.values() for x in slot['outputs']):
            raise ValueError('packed conditioning must preserve original FP32 results')
        if any(any(sig!=slot['inputs'][0] for sig in slot['inputs']) for slot in cache.values()):
            raise ValueError('packed conditioning requires identical per-step input signatures')
        packed=torch.stack([torch.cat([cache[id(m)]['outputs'][k].reshape(-1)
            for m in modules]) for k in range(expert.num_steps)])
        live=packed[0].clone()
        offset=0
        for module in modules:
            slot=cache[id(module)]
            shape=slot['outputs'][0].shape
            size=slot['outputs'][0].numel()
            slot['live']=live[offset:offset+size].view(shape)
            offset+=size
        expert._packed_condition_cache=PackedConditionCache(packed,live)
    for module in modules:
        slot=cache[id(module)]
        if len(slot['outputs'])!=expert.num_steps:
            raise RuntimeError('time conditioning was not called once per step')
        def make_forward(slot):
            def forward(x):
                step=0 if graph_live else getattr(expert,'_rtc_condition_step',None)
                if type(step) is not int or not 0<=step<expert.num_steps:
                    raise RuntimeError('conditioning cache needs explicit expert step')
                if x.requires_grad:
                    raise RuntimeError('refusing to cache activation-dependent gradients')
                if (x.shape,x.dtype,x.device)!=slot['inputs'][step]:
                    raise RuntimeError('time conditioning input signature changed')
                if tuple(p._version for p in slot['module'].parameters())!=slot['versions']:
                    raise RuntimeError('frozen conditioning weights changed')
                return slot['live'] if graph_live else slot['outputs'][step]
            return forward
        module.forward=make_forward(slot)
    expert._condition_cache=cache
    return dict(modules=len(modules),steps=expert.num_steps,
        cached_mib=sum(x.numel()*x.element_size() for v in cache.values() for x in v['outputs'])/2**20,
        limitation='fixed frozen time schedule only; no weight/time/state modification permitted')


class PackedConditionCache:
    """Sequential-only stable views for CUDA graphs; upload outside capture."""
    def __init__(self,packed,live):
        self.packed,self.live=packed,live

    def select_step(self,step):
        self.live.copy_(self.packed[step])
