"""SEAM-VLS adapter for a small asynchronous LIBERO pilot.

The published SEAM method is evaluated at synchronous chunk boundaries. This
adapter keeps our existing async queue/expired-prefix crop, and steers the new
chunk toward the still-queued old tail captured when inference starts. It is
therefore an explicitly labeled async adaptation, not an exact reproduction.
"""

import math
import time


class SEAMGuidanceAdapter:
    def __init__(self, model, torch, *, horizon, num_steps,
                 strength=0.1, max_window=20, latency_prior_seconds=0.4):
        from perceptbudget.rtc_expert import DifferentiableExpert
        from perceptbudget.seam_guidance import CapturedSeamChunk

        self.model = model
        self.pipe = model._pipe
        self.pipeline = self.pipe.pipeline
        self.torch = torch
        self.horizon = int(horizon)
        self.num_steps = int(num_steps)
        self.latency_estimate_seconds = float(latency_prior_seconds)
        self.strength = float(strength)
        self.max_window = int(max_window)
        sample = torch.zeros((self.horizon, 32), device=self.pipe._noise_buf.device,
                             dtype=torch.bfloat16)
        self.expert = DifferentiableExpert(self.pipe).snapshot_context(
            pad_context=True)
        self.output_parity = self._check_vanilla_parity()
        started = time.monotonic()
        self.graph = CapturedSeamChunk(
            self.expert.step_delta, sample, num_steps=self.num_steps,
            strength=self.strength, max_window=self.max_window)
        torch.cuda.synchronize()
        self.capture_seconds = time.monotonic() - started

    def _check_vanilla_parity(self):
        """Refuse the policy experiment if the frozen expert diverges."""
        torch = self.torch
        noise = self.pipe._noise_buf.detach().clone()
        reference = self.pipe._noise_out.detach().clone().float()
        with torch.no_grad():
            candidate = self.expert(noise).float()
        torch.cuda.synchronize(noise.device)
        difference = candidate - reference
        reference_rms = reference.square().mean().sqrt().clamp_min(1e-6)
        relative_rmse = difference.square().mean().sqrt() / reference_rms
        result = {
            'relative_rmse': float(relative_rmse.item()),
            'max_absolute_error': float(difference.abs().max().item()),
            'threshold': 0.02,
            'passed': bool(relative_rmse.item() <= 0.02),
        }
        if not result['passed']:
            raise RuntimeError('SEAM expert parity gate failed: '
                               f"relative RMSE {result['relative_rmse']:.4%} > 2%")
        return result

    def estimate_delay_steps(self, control_period_seconds):
        if control_period_seconds <= 0:
            raise ValueError('control period must be positive')
        return max(1, math.ceil(self.latency_estimate_seconds / control_period_seconds))

    def _refresh_context(self):
        torch = self.torch
        pl = self.pipeline
        prefix = (int(pl.attn.dec_devpos.item()) if pl._fixed_shape
                  else int(pl.encoder_seq_len))
        capacity = self.expert.context_capacity
        if prefix <= 0 or prefix > capacity:
            raise RuntimeError('SEAM context length is outside the captured fixed capacity')
        with torch.no_grad():
            self.expert._context['key'].zero_()
            self.expert._context['value'].zero_()
            self.expert._context['key'][:, :prefix].copy_(
                pl.attn.enc_K[:, :prefix])
            self.expert._context['value'][:, :prefix].copy_(
                pl.attn.enc_V[:, :prefix])
            key_mask = self.expert._context['key_mask']
            key_mask.zero_()
            key_mask[..., :prefix] = True
            key_mask[..., capacity:] = True
        self.expert.prefix_length = prefix

    def predict(self, snapshot, prompt, previous_tail, *, executed,
                delay_estimate, request):
        import numpy as np
        from flash_rt.core.utils.actions import unnormalize_actions

        torch = self.torch
        pipe, pl = self.pipe, self.pipeline
        request['inference_started_monotonic'] = time.monotonic()
        self.model.set_prompt(prompt, state=snapshot.get('state'))
        torch.cuda.synchronize(pipe._noise_buf.device)
        with torch.cuda.stream(pipe._graph_torch_stream):
            pipe._fill_img_buf({'images': snapshot['images']})
            pipe._copy_tensor_to_pipeline_buf_stream(
                pipe._img_buf, pl.input_images_buf,
                pipe._graph_torch_stream.cuda_stream)
            pl._context_graph.replay(pl._graph_stream)
        pipe._graph_torch_stream.synchronize()
        self._refresh_context()

        expected = (self.horizon - int(executed), 32)
        if tuple(previous_tail.shape) != expected:
            raise ValueError(f'old SEAM tail shape {previous_tail.shape}, expected {expected}')
        device = pipe._noise_buf.device
        noise = torch.empty((self.horizon, 32), device=device,
                            dtype=torch.bfloat16)
        noise.normal_()
        prior = torch.as_tensor(np.asarray(previous_tail, dtype=np.float32),
                                device=device)
        raw, vls = self.graph.replay(noise, prior)
        raw = raw.detach().clone()
        torch.cuda.synchronize()

        raw_actions = raw.float().cpu().numpy().astype(np.float32, copy=True)
        actions = unnormalize_actions(raw_actions, pipe.norm_stats)[:, :7]
        request['actions_raw'] = raw_actions.tolist()
        request['seam_guidance'] = {
            'method': 'SEAM VLS async adaptation',
            'paper_is_synchronous': True,
            'execution_steps_before_request': int(executed),
            'estimated_inference_delay_steps': int(delay_estimate),
            'old_tail_start_offset': int(executed),
            'old_tail_length': int(len(previous_tail)),
            **vls,
        }
        request['inference_ended_monotonic'] = time.monotonic()
        self.latency_estimate_seconds = (
            request['inference_ended_monotonic']
            - request['inference_started_monotonic'])
        return actions
