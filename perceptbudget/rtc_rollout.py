"""Opt-in full RTC-guided chunk predictor for the task rollout probe.

This reuses the experiment-side differentiable expert and does not modify
FlashRT. It refreshes observation KV, guides one BF16 chunk, then returns both
raw 32D model actions and the corresponding 7D LIBERO actions.
"""

import time


class RTCGuidanceAdapter:
    def __init__(self, model, torch, *, horizon, num_steps,
                 latency_prior_seconds=0.5994,
                 attention_backend='math'):
        from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedRtcChunk
        from perceptbudget.rtc_expert import DifferentiableExpert

        self.model = model
        self.pipe = model._pipe
        self.pipeline = self.pipe.pipeline
        self.torch = torch
        self.horizon = int(horizon)
        self.num_steps = int(num_steps)
        self.latency_estimate_seconds = float(latency_prior_seconds)
        self.attention_backend = attention_backend
        sample = torch.zeros((self.horizon, 32), device=self.pipe._noise_buf.device,
                             dtype=torch.bfloat16).float()
        self.expert = DifferentiableExpert(
            # FA2 seqused masks variable state-prompt padding while preserving
            # one fixed-shape CUDA graph. Math remains the default for parity.
            self.pipe, attention_backend=attention_backend).snapshot_context(
                fixed_prefix_capacity=True)
        capture_started = time.monotonic()
        self.graph = CapturedRtcChunk(self.expert.step_delta, sample,
                                      num_steps=self.num_steps)
        torch.cuda.synchronize()
        self.capture_seconds = time.monotonic() - capture_started

    def estimate_delay_steps(self, control_period_seconds):
        if control_period_seconds <= 0:
            raise ValueError('control period must be positive')
        import math
        return max(1, math.ceil(self.latency_estimate_seconds /
                                control_period_seconds))

    def predict(self, snapshot, prompt, previous_raw, *, delay, executed, request):
        import numpy as np
        from flash_rt.core.utils.actions import unnormalize_actions
        from perceptbudget.rtc_guidance import soft_mask

        torch = self.torch
        pipe, pl = self.pipe, self.pipeline
        request['inference_started_monotonic'] = time.monotonic()
        with torch.cuda.stream(pipe._graph_torch_stream):
            self.model.set_prompt(prompt, state=snapshot.get('state'))
            pipe._fill_img_buf({'images': snapshot['images']})
            pipe._copy_tensor_to_pipeline_buf_stream(
                pipe._img_buf, pl.input_images_buf,
                pipe._graph_torch_stream.cuda_stream)
            pl._context_graph.replay(pl._graph_stream)
        pipe._graph_torch_stream.synchronize()

        prefix = (int(pl.attn.dec_devpos.item()) if pl._fixed_shape
                  else int(pl.encoder_seq_len))
        self.expert.refresh_context(
            pl.attn.enc_K[:, :self.expert.context_capacity],
            pl.attn.enc_V[:, :self.expert.context_capacity], prefix)

        expected = (self.horizon - executed, 32)
        if tuple(previous_raw.shape) != expected:
            raise ValueError(f'old RTC target shape {previous_raw.shape}, expected {expected}')
        device = self.pipe._noise_buf.device
        noise = torch.empty((self.horizon, 32), device=device,
                            dtype=torch.bfloat16)
        noise.normal_()
        noise = noise.float()
        target = torch.zeros_like(noise)
        target[:len(previous_raw)].copy_(torch.as_tensor(
            np.asarray(previous_raw, dtype=np.float32), device=device))
        mask = torch.tensor(soft_mask(self.horizon, delay, executed),
                            device=device, dtype=torch.float32)[:, None]
        raw, vjp_norms, error_norms = self.graph.replay(
            noise, target, mask, beta=5.)
        raw = raw.detach().clone()
        vjp_norms = vjp_norms.detach().clone()
        error_norms = error_norms.detach().clone()
        torch.cuda.synchronize()

        raw_actions = raw.cpu().numpy().astype(np.float32, copy=True)
        actions = unnormalize_actions(raw_actions, self.pipe.norm_stats)[:, :7]
        request['actions_raw'] = raw_actions.tolist()
        request.setdefault('rtc_guidance', {}).update({
            'delay_estimate_steps': int(delay),
            'executed_steps': int(executed),
            'old_target_shape': list(previous_raw.shape),
            'beta': 5.0,
            'attention_backend': self.attention_backend,
            'vjp_norms': vjp_norms.cpu().tolist(),
            'weighted_error_norms': error_norms.cpu().tolist(),
        })
        request['inference_ended_monotonic'] = time.monotonic()
        self.latency_estimate_seconds = (
            request['inference_ended_monotonic']
            - request['inference_started_monotonic'])
        return actions
