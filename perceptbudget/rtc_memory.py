"""One-way context-only conversion for an isolated OpenPI RTC process."""
import gc


def memory_record(torch, stage):
    available=None
    with open('/proc/meminfo',encoding='ascii') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                available=int(line.split()[1])/1024
                break
    result = dict(stage=stage,available_mib=available,
                torch_allocated_mib=torch.cuda.memory_allocated()/2**20,
                torch_reserved_mib=torch.cuda.memory_reserved()/2**20)
    # Jetson native CUDA/EGL allocations are not all visible to PyTorch.
    # Keep process residency separate from system-wide MemAvailable.
    with open('/proc/self/status', encoding='ascii') as stream:
        for line in stream:
            name = line.split(':', 1)[0]
            if name in ('VmRSS', 'VmSize', 'RssAnon', 'RssFile', 'VmSwap'):
                result[name.lower()+'_mib'] = int(line.split()[1])/1024
    return result


def resident_memory_record(torch, stage, *, cleanup, floor_mib, collect=True):
    """Measure setup memory; collect at a boundary or when headroom is low."""
    import time
    started = time.monotonic()
    collected = collect
    if collect:
        cleanup(torch)
    value = memory_record(torch, stage)
    if not collect and value['available_mib'] < floor_mib:
        cleanup(torch)
        collected = True
        value = memory_record(torch, stage)
    value.update(cleanup_performed=collected,
                 monitor_seconds=time.monotonic()-started)
    return value


def release_unused_native_expert(model,torch):
    """Only BF16, fixed, prewarmed pipelines. No CUDA handle destruction.

    Native full/decoder graphs retain their handles until process exit; they
    must not be replayed. Context graph retains all its original input storage.
    This is deliberately irreversible in this experimental process.
    """
    pipe=model._pipe
    pl=pipe.pipeline
    if getattr(pipe,'_pb_context_only',False):
        raise RuntimeError('native expert was already released')
    if (not getattr(pipe,'_force_bf16',False) or pl is not getattr(pipe,'_fixed_pipeline',None)
            or getattr(pl,'_context_graph',None) is None):
        raise RuntimeError('cleanup requires prewarmed fixed BF16 context graph')
    if any(getattr(pipe,name,{}) for name in ('_fp8_weights','_int8_weights')):
        raise RuntimeError('cleanup does not support quantized stores')
    names=[name for name in pipe._ckpt_bf16 if name.startswith('decoder_')]
    if not names:
        raise RuntimeError('no native expert tensors found; refusing unknown layout')
    torch.cuda.synchronize()
    before=memory_record(torch,'before_native_expert_release')
    sizes={name:pipe._ckpt_bf16[name].numel()*pipe._ckpt_bf16[name].element_size() for name in names}

    def forbidden(*args,**kwargs):
        raise RuntimeError('context-only RTC process: native expert released; native inference/rebuild forbidden')

    # Install guards before removing storage. Never manually destroy native
    # graph handles, whose ownership may also be held by the execution context.
    pipe.infer=pipe.infer_batch=pipe._build_pipeline_weights=forbidden
    pl.transformer_decoder=pl.forward=pl.record_infer_graph=forbidden
    class DisabledGraph:
        def __init__(self,graph): self.retained_handle=graph
        replay=staticmethod(forbidden)
    for name in ('_graph','_decoder_only_graph'):
        graph=getattr(pl,name,None)
        if graph is not None:
            setattr(pl,name,DisabledGraph(graph))
    for name in names:
        del pipe._ckpt_bf16[name]
    for name in list(pl.weights):
        if name.startswith('decoder_'):
            del pl.weights[name]
    # Context capture only executes vision+encoder. Disabled native decode
    # graphs retain handles, but their private buffers/styles are never replayed.
    # Prompt setup updates decoder_rope_weights before every context replay;
    # retain that small buffer even though the decoder itself is disabled.
    decoder_buffers={name:buf.nbytes for name,buf in getattr(pl,'bufs',{}).items()
        if (name.startswith('decoder_') and name!='decoder_rope_weights')
        or name in ('diffusion_noise','rtc_prev_action_chunk',
            'rtc_prefix_weights','rtc_guidance_weight','x_normed_buf','gate_buf')}
    for name in decoder_buffers:
        del pl.bufs[name]
    precomputed=pl.weights.pop('precomputed',{})
    precomputed_bytes=sum(getattr(value,'nbytes',0) for value in precomputed.values())
    if hasattr(pipe,'_precomputed_styles'):
        pipe._precomputed_styles={}
    del precomputed
    pipe._pb_context_only=True
    gc.collect()
    torch.cuda.empty_cache()
    return dict(before=before,after=memory_record(torch,'after_native_expert_release'),
                released_weight_mib=sum(sizes.values())/2**20,released_names=names,
                released_buffer_mib=sum(decoder_buffers.values())/2**20,
                released_precomputed_reference_mib=precomputed_bytes/2**20,
                released_buffer_names=list(decoder_buffers),
                limitation='native graph handles retained but disabled; context-only for this process')
