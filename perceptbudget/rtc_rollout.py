"""Opt-in full RTC-guided chunk predictor for the task rollout probe.

This reuses the experiment-side differentiable expert and does not modify
FlashRT. It refreshes observation KV, guides one BF16 chunk, then returns both
raw 32D model actions and the corresponding 7D LIBERO actions.
"""

import time


class RTCGuidanceAdapter:
    def __init__(self, model, torch, *, horizon, num_steps,
                 latency_prior_seconds=0.5994,
                 attention_backend='math', guidance_enabled=True, audit_reference=False,
                 reference_backend=False, eager_backend=False):
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
        self.guidance_enabled = bool(guidance_enabled)
        self.audit_reference = bool(audit_reference)
        self._reference_audited = False
        self._audited_prefixes = set()
        self.reference_backend = bool(reference_backend)
        self.eager_backend = bool(eager_backend)
        sample = torch.zeros((self.horizon, 32), device=self.pipe._noise_buf.device,
                             dtype=torch.bfloat16).float()
        self.expert = DifferentiableExpert(
            # FA2 seqused masks variable state-prompt padding while preserving
            # one fixed-shape CUDA graph. Math remains the default for parity.
            self.pipe, attention_backend=attention_backend).snapshot_context(
                fixed_prefix_capacity=attention_backend != 'flash')
        capture_started = time.monotonic()
        self.graph = (None if self.reference_backend or self.eager_backend else CapturedRtcChunk(
            self.expert.step_delta, sample, num_steps=self.num_steps))
        torch.cuda.synchronize()

        self.capture_seconds = time.monotonic() - capture_started

    def release_graph(self):
        """Drop inference-only workspaces before CPU trajectory compression."""
        import gc
        def memory():
            with open('/proc/meminfo') as stream:
                available = next(int(line.split()[1])*1024 for line in stream
                                 if line.startswith('MemAvailable:'))
            return {'available_ram_bytes': available,
                    'cuda_allocated_bytes': self.torch.cuda.memory_allocated(),
                    'cuda_reserved_bytes': self.torch.cuda.memory_reserved()}
        self.torch.cuda.synchronize()
        before = memory()
        self.graph = None
        gc.collect()
        self.torch.cuda.empty_cache()
        return {'before': before, 'after': memory()}

    def _refresh_context_for_prefix(self, prefix):
        """Flash SDPA has no padding mask: replace, never retain, length graphs."""
        started = time.monotonic()
        rebuilt = self.attention_backend == 'flash' and prefix != self.expert.context_capacity
        if rebuilt:
            import gc
            from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedRtcChunk
            # Release every graph reference before replacing its context buffers.
            self.torch.cuda.synchronize()
            self.graph = None
            gc.collect()
            self.torch.cuda.empty_cache()
            self.expert.snapshot_context(fixed_prefix_capacity=False)
            if self.expert.context_capacity != prefix:
                raise RuntimeError('new RTC context did not use the actual state prompt length')
            if not self.reference_backend and not self.eager_backend:
                sample = self.torch.zeros((self.horizon, 32), device=self.pipe._noise_buf.device,
                                          dtype=self.torch.float32)
                self.graph = CapturedRtcChunk(self.expert.step_delta, sample, num_steps=self.num_steps)
            self._reference_audited = prefix in self._audited_prefixes
        else:
            self.expert.refresh_context(
                self.pipeline.attn.enc_K[:, :self.expert.context_capacity],
                self.pipeline.attn.enc_V[:, :self.expert.context_capacity], prefix)
        return {'valid_prefix_length': prefix, 'context_capacity': self.expert.context_capacity,
                'context_graph_rebuilt': rebuilt and not (self.reference_backend or self.eager_backend),
                'context_resized': rebuilt,
                'context_prepare_seconds': time.monotonic()-started}

    def estimate_delay_steps(self, control_period_seconds):
        if control_period_seconds <= 0:
            raise ValueError('control period must be positive')
        import math
        return max(1, math.ceil(self.latency_estimate_seconds /
                                control_period_seconds))

    def predict(self, snapshot, prompt, previous_raw, *, delay, executed, request):
        import numpy as np
        from flash_rt.core.utils.actions import unnormalize_actions
        from perceptbudget.rtc_guidance import soft_mask_tensor

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
        context_diagnostic = self._refresh_context_for_prefix(prefix)
        request.setdefault('rtc_guidance', {}).update(context_diagnostic)

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
        mask = soft_mask_tensor(self.horizon, delay, executed, device=device)
        if not self.guidance_enabled:
            mask.zero_()
        if self.reference_backend:
            if not self.guidance_enabled:
                raise ValueError('paper-reference backend requires guidance')
            from perceptbudget.rtc_guidance import paper_reference
            raw, norms = paper_reference(
                self.expert.step_delta, noise, torch.as_tensor(previous_raw, device=device),
                delay=delay, executed=executed, num_steps=self.num_steps)
            vjp_norms = torch.tensor(norms, device=device)
            error_norms = torch.empty(0, device=device)
        elif self.eager_backend:
            from perceptbudget.rtc_guidance import guided_sample
            if not self.guidance_enabled:
                raise ValueError('eager validation path requires RTC guidance')
            raw, records = guided_sample(
                self.expert.step_delta, noise, torch.as_tensor(previous_raw, device=device),
                delay=delay, executed=executed, num_steps=self.num_steps)
            vjp_norms = torch.tensor([r['vjp_norm'] for r in records], device=device)
            error_norms = torch.tensor([r['weighted_error_norm'] for r in records], device=device)
        else:
            raw, vjp_norms, error_norms = self.graph.replay(
                noise, target, mask, beta=5.)
        raw = raw.detach().clone()
        vjp_norms = vjp_norms.detach().clone()
        error_norms = error_norms.detach().clone()
        torch.cuda.synchronize()

        if self.audit_reference and not self.reference_backend and self.guidance_enabled and not self._reference_audited:
            import gc
            from perceptbudget.rtc_guidance import paper_reference
            from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedRtcChunk
            # CUDA graph's private pool must not coexist with a full eager
            # backward workspace on a 16 GiB shared-memory device.
            self.graph = None
            gc.collect()
            torch.cuda.empty_cache()
            reference, ref_norms = paper_reference(
                self.expert.step_delta, noise, torch.as_tensor(previous_raw, device=device),
                delay=delay, executed=executed, num_steps=self.num_steps)
            difference = float((raw-reference).abs().max())
            norm_difference = float((vjp_norms-torch.tensor(ref_norms, device=device)).abs().max())
            request['rtc_reference_audit'] = {
                'raw_max_absolute_error': difference,
                'per_step_vjp_norm_max_error': norm_difference,
                'passed': difference < 1e-4 and norm_difference < 1e-4,
                'reference': 'independent decreasing-time equations and functional VJP'}
            if not request['rtc_reference_audit']['passed']:
                raise RuntimeError(f'RTC paper-reference mismatch: {request["rtc_reference_audit"]}')
            self._reference_audited = True
            self._audited_prefixes.add(prefix)
            del reference
            torch.cuda.empty_cache()
            if not self.eager_backend:
                self.graph = CapturedRtcChunk(self.expert.step_delta, noise,
                                              num_steps=self.num_steps)

        raw_actions = raw.cpu().numpy().astype(np.float32, copy=True)
        actions = unnormalize_actions(raw_actions, self.pipe.norm_stats)[:, :7]
        request['actions_raw'] = raw_actions.tolist()
        request.setdefault('rtc_guidance', {}).update({
            'delay_estimate_steps': int(delay),
            'executed_steps': int(executed),
            'old_target_shape': list(previous_raw.shape),
            'beta': 5.0,
            'guidance_enabled': self.guidance_enabled,
            'solver_backend': ('paper_reference_eager' if self.reference_backend else
                               'eager_single_forward' if self.eager_backend else 'captured_graph'),
            'attention_backend': self.attention_backend,
            'vjp_norms': vjp_norms.cpu().tolist(),
            'weighted_error_norms': error_norms.cpu().tolist(),
        })
        request['inference_ended_monotonic'] = time.monotonic()
        self.latency_estimate_seconds = (
            request['inference_ended_monotonic']
            - request['inference_started_monotonic'])
        return actions
