"""Prepare only native vision/language context for the separate RTC expert."""
import ctypes
import time


def prepare_context_only(model,snapshot,prompt,torch):
    """Use upstream context operations without capturing unused native actions.

    Both the eager and captured context must produce exactly the same valid
    K/V. No padding, precision, model weights or RTC sampling is changed.
    """
    from flash_rt.core.cuda_graph import CUDAGraph
    from flash_rt.subgraphs.pi05.context_action import _run_context
    from flash_rt.subgraphs.capture import register_captured_graph
    from perceptbudget.openpi_reference import original_prefix

    started=time.monotonic();pipe=model._pipe
    pipe.set_prompt(prompt,state=snapshot.get('state'))
    pl=pipe.pipeline
    if not pipe._force_bf16 or not pl._fixed_shape or pl._graph is not None:
        raise ValueError('Context-only preparation requires a fresh fixed BF16 pipeline')
    if getattr(pipe,'_graph_torch_stream',None) is None:
        pipe._graph_torch_stream=torch.cuda.Stream()
    stream=pipe._graph_torch_stream;handle=ctypes.c_void_p(stream.cuda_stream)
    with torch.cuda.stream(stream):
        pipe._fill_img_buf(snapshot)
        pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf,pl.input_images_buf,stream.cuda_stream)
        pl.autotune_gemms()
        for _ in range(3):_run_context(pl,stream.cuda_stream)
        stream.synchronize()
        valid=int(pl.attn.dec_devpos.item())
        before_k,before_v=original_prefix(pl.attn.enc_K,pl.attn.enc_V,valid)
        before_k=before_k.clone();before_v=before_v.clone()
        graph=CUDAGraph();graph.begin_capture(handle)
        _run_context(pl,stream.cuda_stream)
        graph.end_capture(handle);stream.synchronize()
        graph.replay(handle);stream.synchronize()
        after_k,after_v=original_prefix(pl.attn.enc_K,pl.attn.enc_V,valid)
        key_error=float((after_k-before_k).abs().max())
        value_error=float((after_v-before_v).abs().max())
    if key_error!=0. or value_error!=0.:
        raise RuntimeError('Context-only capture differs from native eager context')
    pl._context_graph=graph;pl._graph_stream=handle
    register_captured_graph(pl,'context',graph,exec_name='pi05_context',stream=0,variants=(0,))
    pipe.calibrated=True
    return dict(backend='native_context_only',passed=True,valid_prefix=valid,
        key_max_abs=key_error,value_max_abs=value_error,native_action_graphs=0,
        prepare_seconds=time.monotonic()-started)
