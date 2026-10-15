"""Project-side use of FlashRT's opt-in prefix graph; not full RTC guidance."""

import ctypes
import time


class PrefixModel:
    def __init__(self, model, prefix_len=8):
        from flash_rt.subgraphs.pi05.rtc_prefix import enable
        self.model = model
        self.prefix_len = prefix_len
        self._variants = {}
        enable(model, prefix_len=prefix_len)

    def prepare_variants(self, lengths):
        """Capture extra plans before task timing; share weights and buffers."""
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
        return actions, raw
