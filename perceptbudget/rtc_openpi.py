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
        self.expert.refresh_context(pl.attn.enc_K, pl.attn.enc_V, valid)
        noise = torch.randn(self.horizon, 32, device='cuda', dtype=torch.float32)
        previous = None if previous_raw is None else torch.as_tensor(np.asarray(previous_raw), device='cuda', dtype=torch.float32)
        raw, norms = sample_original_rtc(self.expert, noise, previous, delay, executed)
        raw_actions = raw.cpu().numpy().copy()
        request['actions_raw'] = raw_actions.tolist()
        request.setdefault('rtc_guidance', {}).update(dict(
            delay_estimate_steps=delay, executed_steps=executed, guidance_enabled=previous is not None,
            solver_backend='original_openpi_eager_vjp', expert_backend='original_openpi_source',
            valid_prefix_length=valid, vjp_norms=norms, beta=5.,
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
        self.expert = None
        gc.collect()
        self.torch.cuda.empty_cache()
        return {'original_openpi_suffix_released': True, 'cuda_graph_used': False}
