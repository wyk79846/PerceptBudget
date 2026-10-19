"""Correctness-first RTC adapter: FlashRT context, original OpenPI suffix."""
import time


class OriginalOpenPiRTCAdapter:
    def __init__(self, model, torch, config):
        from perceptbudget.openpi_reference import OriginalOpenPiExpert
        self.model, self.torch, self.config = model, torch, config
        self.pipe, self.pipeline = model._pipe, model._pipe.pipeline
        self.horizon, self.num_steps = config['H'], config['N']
        self.expert = OriginalOpenPiExpert(config, self.horizon, self.num_steps)
        self.latency_estimate_seconds = 0.
        self.graph = None
        self.capture_seconds = 0.
        self.graph_enabled = False

    def prepare_graph(self, previous_raw):
        """Gate one exact prefix shape; never pad state tokens for capture."""
        import numpy as np
        from perceptbudget.openpi_reference import sample_original_rtc
        from perceptbudget.openpi_graph import CapturedOriginalRtc
        torch = self.torch
        noise = torch.randn(self.horizon,32,device='cuda',dtype=torch.float32)
        previous = torch.as_tensor(np.asarray(previous_raw),device='cuda',dtype=torch.float32)
        executed = self.horizon-len(previous)
        delay = min(5,len(previous))
        reference, norms = sample_original_rtc(self.expert,noise,previous,delay,executed)
        started = time.monotonic()
        self.graph = CapturedOriginalRtc(self.expert,noise)
        self.capture_seconds = time.monotonic()-started
        raw, _, corrections = self.graph.replay(noise,previous,delay,executed)
        raw_error = float((raw-reference).abs().max())
        norm_error = float((corrections.norm(dim=(1,2))-torch.tensor(norms,device='cuda')).abs().max())
        import math
        if not math.isfinite(raw_error) or not math.isfinite(norm_error) or raw_error >= 1e-4 or norm_error >= 1e-4:
            raise RuntimeError('live original CUDA graph parity failed')
        self.graph_prefix_length = int(self.expert.prefix_mask.shape[1])
        self.graph_enabled = True
        return dict(raw_max_abs=raw_error,vjp_norm_max_abs=norm_error,passed=True,
                    prefix_length=self.graph_prefix_length,capture_seconds=self.capture_seconds,
                    limitation='length changes require explicit recapture; capture latency is not hidden')

    def _drop_graph(self):
        """Bound capture memory to one length, including on capture failure."""
        import gc
        self.graph = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def estimate_delay_steps(self, period):
        import math
        return max(1, math.ceil(self.latency_estimate_seconds/period))

    def _predict(self, snapshot, prompt, previous_raw, delay, executed, request):
        import numpy as np
        from flash_rt.core.utils.actions import unnormalize_actions
        from perceptbudget.openpi_reference import sample_original_rtc
        torch, pipe, pl = self.torch, self.pipe, self.pipeline
        request['inference_started_monotonic'] = time.monotonic()
        with torch.cuda.stream(pipe._graph_torch_stream):
            self.model.set_prompt(prompt, state=snapshot.get('state'))
            pipe._fill_img_buf({'images': snapshot['images']})
            pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf, pl.input_images_buf,
                                                     pipe._graph_torch_stream.cuda_stream)
            pl._context_graph.replay(pl._graph_stream)
        pipe._graph_torch_stream.synchronize()
        valid = int(pl.attn.dec_devpos.item()) if pl._fixed_shape else int(pl.encoder_seq_len)
        rebuilt = False
        prepare_seconds = 0.
        if self.graph is not None and valid != self.graph_prefix_length:
            self._drop_graph()
        self.expert.refresh_context(pl.attn.enc_K, pl.attn.enc_V, valid)
        if self.graph_enabled and self.graph is None and previous_raw is not None:
            started = time.monotonic()
            try:
                gate = self.prepare_graph(previous_raw)
            except BaseException:
                self._drop_graph()
                raise
            rebuilt = True
            prepare_seconds = time.monotonic()-started
            request['original_graph_gate'] = gate
        noise = torch.randn(self.horizon, 32, device='cuda', dtype=torch.float32)
        previous = None if previous_raw is None else torch.as_tensor(np.asarray(previous_raw), device='cuda', dtype=torch.float32)
        if self.graph is not None and previous is not None:
            raw, _, corrections = self.graph.replay(noise,previous,delay,executed)
            norms = corrections.norm(dim=(1,2)).cpu().tolist()
        else:
            raw, norms = sample_original_rtc(self.expert, noise, previous, delay, executed)
        raw_actions = raw.cpu().numpy().copy()
        if not np.isfinite(raw_actions).all() or not np.isfinite(norms).all():
            raise RuntimeError('nonfinite original OpenPI RTC output or VJP')
        request['actions_raw'] = raw_actions.tolist()
        request.setdefault('rtc_guidance', {}).update(dict(
            delay_estimate_steps=delay, executed_steps=executed, guidance_enabled=previous is not None,
            solver_backend='original_openpi_cuda_graph' if self.graph is not None and previous is not None else 'original_openpi_eager_vjp',
            expert_backend='original_openpi_source',
            valid_prefix_length=valid, vjp_norms=norms, beta=5.,
            context_graph_rebuilt=rebuilt, context_prepare_seconds=prepare_seconds,
            state_in_prompt=snapshot.get('state') is not None))
        actions = unnormalize_actions(raw_actions, pipe.norm_stats)[:, :7]
        request['inference_ended_monotonic'] = time.monotonic()
        self.latency_estimate_seconds = request['inference_ended_monotonic']-request['inference_started_monotonic']
        return actions

    def bootstrap(self, snapshot, prompt, request):
        return self._predict(snapshot, prompt, None, 0, 0, request)

    def predict(self, snapshot, prompt, previous_raw, *, delay, executed, request):
        return self._predict(snapshot, prompt, previous_raw, delay, executed, request)

    def release_graph(self):
        import gc
        used_graph = self.graph is not None
        self.graph = None
        self.expert = None
        gc.collect()
        self.torch.cuda.empty_cache()
        return {'original_openpi_suffix_released': True, 'cuda_graph_used': used_graph}
