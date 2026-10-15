"""Project-side use of FlashRT's opt-in prefix graph; not full RTC guidance."""

import ctypes
import time
from contextlib import contextmanager, nullcontext


def _clean_t0_styles(weights, chunk_size):
    """Same BF16 MLP/modulations as FlashRT, evaluated at t=0.

    Keep H rows for the matrix products, matching the frontend's computation
    shape. Return BF16 bits, without changing weights or existing style tables.
    """
    import torch
    w = weights
    device = w['decoder_time_mlp_in_w'].device
    dim = w['decoder_time_mlp_in_w'].shape[0]
    if dim % 2:
        raise ValueError('time embedding dimension must be even')
    # sin(0)=0, cos(0)=1 for every period in the original sinusoid.
    te = torch.cat((torch.zeros(dim // 2, device=device),
                    torch.ones(dim // 2, device=device))).to(torch.bfloat16)[None]
    with torch.no_grad():
        tmp = te @ w['decoder_time_mlp_in_w'] + w['decoder_time_mlp_in_b'][None]
        tmp = (tmp.float() * torch.sigmoid(tmp.float())).to(torch.bfloat16)
        tmp = tmp @ w['decoder_time_mlp_out_w'] + w['decoder_time_mlp_out_b'][None]
        tmp = (tmp.float() * torch.sigmoid(tmp.float())).to(torch.bfloat16)
        expanded = tmp.expand(chunk_size, -1).contiguous()
        result = {'time_emb': expanded}
        for kind, name in [('attn', 'pre_attn'), ('ffn', 'pre_ffn')]:
            mw = w[f'decoder_{name}_norm_mod_w']
            mb = w[f'decoder_{name}_norm_mod_b']
            result[f'style_{kind}'] = torch.stack([
                expanded @ mw[i] + mb[i][None] for i in range(mw.shape[0])])
        result['style_final'] = (expanded @ w['decoder_final_norm_mod_w']
                                 + w['decoder_final_norm_mod_b'][None])
        return {name: value.contiguous().view(torch.uint16).cpu().numpy().copy()
                for name, value in result.items()}


@contextmanager
def _temporary_t0_styles(pl, rows, prefix_len, synchronize, timing=None):
    """Replace only prefix token rows; restore actual device bytes on all exits.

    No graph capture happens here. All replays must finish before restoration;
    the caller supplies synchronization for the frontend graph stream.
    """
    import numpy as np
    timing = {} if timing is None else timing
    backup_start = time.monotonic()
    if not 0 < prefix_len < pl.chunk_size:
        raise ValueError('invalid clean prefix length')
    originals, modified = {}, {}
    # Validate and back up ALL buffers before the first mutation.
    for name in ('time_emb', 'style_attn', 'style_ffn', 'style_final'):
        row = rows[name]
        width = 1024 if name == 'time_emb' else 3072
        expected = ((18, pl.chunk_size, width) if name in ('style_attn', 'style_ffn')
                    else (pl.chunk_size, width))
        if row.shape != expected or row.dtype != np.uint16:
            raise ValueError(f'invalid t0 {name} shape/dtype: {row.shape}/{row.dtype}')
        shape = (pl.num_steps,) + expected
        buf = pl.bufs['decoder_' + name]
        original = np.empty(shape, dtype=np.uint16)
        if buf.nbytes != original.nbytes:
            raise ValueError(f'invalid decoder_{name} buffer size')
        originals[name] = original
        modified[name] = None
    synchronize()
    for name, original in originals.items():
        pl.bufs['decoder_' + name].download(original)
        changed = original.copy()
        changed[..., :prefix_len, :] = rows[name][..., :prefix_len, :]
        modified[name] = changed
    timing['prefix_style_backup_seconds'] = time.monotonic() - backup_start
    timing['prefix_style_tail_policy'] = 'original bytes copied; only prefix rows assigned'
    switch_start = time.monotonic()
    try:
        for name, changed in modified.items():
            pl.bufs['decoder_' + name].upload(changed)
        # upload is a synchronous CudaBuffer operation; explicitly fence before
        # launching the non-default graph stream as well.
        synchronize()
        timing['prefix_style_switch_seconds'] = time.monotonic() - switch_start
        yield
    finally:
        timing.setdefault('prefix_style_switch_seconds', time.monotonic() - switch_start)
        restore_start = time.monotonic()
        try:
            synchronize()
        finally:
            # Attempt every restore even if one upload raises.
            failure = None
            for name, original in originals.items():
                try:
                    pl.bufs['decoder_' + name].upload(original)
                except BaseException as exc:
                    failure = exc
            try:
                synchronize()
            finally:
                timing['prefix_style_restore_seconds'] = time.monotonic() - restore_start
                timing['prefix_style_overhead_seconds'] = sum(timing[k] for k in (
                    'prefix_style_backup_seconds', 'prefix_style_switch_seconds',
                    'prefix_style_restore_seconds'))
            if failure is not None:
                raise failure


class PrefixModel:
    def __init__(self, model, prefix_len=8, clean_prefix_t0=False):
        from flash_rt.subgraphs.pi05.rtc_prefix import enable
        self.model = model
        self.prefix_len = prefix_len
        self._variants = {}
        self.clean_prefix_t0 = clean_prefix_t0
        self._t0_rows = None
        enable(model, prefix_len=prefix_len)

    def prepare_clean_prefix_styles(self):
        """Precompute t=0 rows after ordinary pipeline warmup, before timing.

        This does not upload styles, replay/capture graphs, or change the mode.
        Call on the same frontend stream used by predict. Existing captured
        graphs continue to read their original shared buffer addresses.
        """
        import torch
        pipe = self.model._pipe
        pl = pipe.pipeline
        if (pl is None or not hasattr(pl, '_decode_rtc_prefix_graph')
                or pl.chunk_size != 20 or pipe._force_bf16 is not True):
            raise RuntimeError('warm up the BF16/H20 prefix pipeline before preparing t0 styles')
        if self._t0_rows is None:
            with torch.cuda.stream(pipe._graph_torch_stream):
                self._t0_rows = _clean_t0_styles(pipe._ckpt_bf16, pl.chunk_size)

    def prepare_variants(self, lengths):
        """Capture extra plans before task timing; share weights and buffers."""
        if self.clean_prefix_t0:
            raise ValueError('clean_prefix_t0 uses only the fixed prefix_len; no variants')
        import torch
        from flash_rt.subgraphs.capture import capture_graph
        pipe = self.model._pipe
        pl = pipe.pipeline
        graphs = self._variants.setdefault(id(pl), {self.prefix_len: pl._decode_rtc_prefix_graph})
        with torch.cuda.stream(pipe._graph_torch_stream):
            for length in lengths:
                if type(length) is not int or not 0 < length < pl.chunk_size:
                    raise ValueError('invalid prefix variant length')
                if length not in graphs:
                    graphs[length] = capture_graph(
                        pl, pl._graph_stream,
                        lambda length=length: pl.transformer_decoder(
                            pipe._graph_torch_stream.cuda_stream, rtc_prefix_len=length))

    def predict(self, snapshot, prompt, timing, raw_prefix=None, prefix_len=None):
        import numpy as np
        import torch
        from flash_rt.core.utils.actions import unnormalize_actions

        pipe = self.model._pipe
        timing['inference_started_monotonic'] = time.monotonic()
        if raw_prefix is None:
            actions = self.model.predict(snapshot['images'], prompt=prompt,
                                         state=snapshot.get('state'))
            raw = pipe._noise_out.float().cpu().numpy().copy()
            timing['inference_ended_monotonic'] = time.monotonic()
            return actions, raw

        length = self.prefix_len if prefix_len is None else prefix_len
        if self.clean_prefix_t0 and length != self.prefix_len:
            raise ValueError('clean_prefix_t0 requires the fixed prefix_len')
        raw_prefix = np.asarray(raw_prefix, dtype=np.float32)
        if raw_prefix.shape != (length, 32) or not np.isfinite(raw_prefix).all():
            raise ValueError('prefix must be finite (prefix_len,32) raw model actions')
        # Match public API's prompt/state setup; do not change frontend methods.
        with torch.cuda.stream(pipe._graph_torch_stream):
            self.model.set_prompt(prompt, state=snapshot.get('state'))
            pl = pipe.pipeline
            if not hasattr(pl, '_decode_rtc_prefix_graph'):
                raise RuntimeError('FlashRT prefix graph was not captured')
            if pl.chunk_size != 20 or pipe._force_bf16 is not True:
                raise RuntimeError('prefix adapter requires verified BF16/H20')
            stream = pipe._graph_torch_stream.cuda_stream
            graphs = self._variants.get(id(pl), {self.prefix_len: pl._decode_rtc_prefix_graph})
            if length not in graphs:
                raise RuntimeError('prefix length was not prepared before task execution')
            packed = torch.zeros((pl.chunk_size, 32), dtype=torch.bfloat16, device='cuda')
            packed[:length].copy_(torch.as_tensor(raw_prefix, device='cuda', dtype=torch.bfloat16))
            pipe._copy_tensor_to_pipeline_buf_stream(packed, pl.input_rtc_prev_action_chunk_buf, stream)
            pipe._noise_buf.normal_()
            pipe._copy_tensor_to_pipeline_buf_stream(pipe._noise_buf, pl.input_noise_buf, stream)
            observation = {'images': snapshot['images']}
            pipe._fill_img_buf(observation)
            pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf, pl.input_images_buf, stream)
            styles = nullcontext()
            if self.clean_prefix_t0:
                prepare_start = time.monotonic()
                if self._t0_rows is None:
                    self.prepare_clean_prefix_styles()
                timing['prefix_style_prepare_seconds'] = time.monotonic() - prepare_start
                styles = _temporary_t0_styles(
                    pl, self._t0_rows, length, pipe._graph_torch_stream.synchronize,
                    timing)
            with styles:
                pl._context_graph.replay(pl._graph_stream)
                graphs[length].replay(pl._graph_stream)
                pipe._cudart.cudaMemcpyAsync(ctypes.c_void_p(pipe._noise_out.data_ptr()),
                                            pl.input_noise_buf.ptr,
                                            pipe._noise_out.numel() * 2, 3, stream)
        pipe._cudart.cudaStreamSynchronize(ctypes.c_void_p(pipe._graph_torch_stream.cuda_stream))
        raw = pipe._noise_out.float().cpu().numpy().copy()
        expected = packed[:length].float().cpu().numpy()
        if not np.array_equal(raw[:length], expected):
            raise RuntimeError('prefix graph did not preserve committed raw actions exactly')
        actions = unnormalize_actions(raw, pipe.norm_stats)[:, :7]
        if not np.isfinite(actions).all():
            raise RuntimeError('nonfinite prefix action output')
        timing.update(inference_ended_monotonic=time.monotonic(), prefix_exact=True)
        timing['clean_prefix_t0'] = bool(self.clean_prefix_t0)
        return actions, raw
