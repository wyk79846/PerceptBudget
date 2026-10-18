"""Differentiable replica of the FlashRT RTX Pi05 BF16 action expert.

Only decoder weights are referenced (shared storage, never copied). Context,
RoPE and time styles are immutable snapshots of an already prepared pipeline.
This is a numerical replica, not backward support for FlashRT custom kernels.
GPU parity MUST be gated: math SDPA versus FA2, reduction order, CUDA FMA and
cuBLAS algorithms can differ. BF16 casts use PyTorch's usual cast derivative;
the VJP is the conventional mixed-precision network VJP, not the derivative of
the discontinuous quantized bit mapping. No paper guidance math lives here.

Source contracts in FlashRT-upstream:
  frontends/torch/pi05_rtx.py: output weights already scaled by -1/N;
  models/pi05/pipeline_rtx.py: transformer_decoder / _decoder_layer;
  csrc/kernels/norm.cu: ada_rms_norm_style_kernel (scale, shift, gate);
  csrc/kernels/rope.cu: qkv_split_rope_kernel (adjacent pairs);
  csrc/kernels/activation.cu: gate_silu_mul_kernel actually computes GELU;
  csrc/bindings.cpp: gate_geglu binds that GELU kernel.
"""

import operator

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class _TorchVarlenGqaAttention(torch.autograd.Function):
    """Torch FlashAttention forward/backward with device-side valid lengths.

    Put action KV first, followed by the valid frozen context. Physical buffer
    shapes remain fixed; cum_k excludes trailing padding without a custom mask.
    Only first-order input VJP is required by RTC; weights remain frozen.
    """
    @staticmethod
    def forward(ctx, q, action_k, action_v, context_k, context_v, cum_q, cum_k):
        query = q.squeeze(0).transpose(0, 1).contiguous()
        key = torch.cat((action_k, context_k), 0).contiguous()
        value = torch.cat((action_v, context_v), 0).contiguous()
        h, capacity = query.shape[0], key.shape[0]
        out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
            query, key, value, cum_q, cum_k, h, capacity, 0., False, False,
            scale=query.shape[-1]**-.5)
        ctx.save_for_backward(query, key, value, out, lse, cum_q, cum_k, rng, unused)
        return out.transpose(0, 1).unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_out):
        q, key, value, out, lse, cum_q, cum_k, rng, unused = ctx.saved_tensors
        h = q.shape[0]
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad_out.squeeze(0).transpose(0, 1).contiguous(), q, key, value,
            out, lse, cum_q, cum_k, h, key.shape[0], 0., False, rng, unused,
            scale=q.shape[-1]**-.5)
        return dq.transpose(0, 1).unsqueeze(0), dk[:h], dv[:h], None, None, None, None


class _FlashRtSequsedGqaAttention(torch.autograd.Function):
    """FlashRT FA2 forward with an exact, query-side/action-side VJP.

    The action chunk is bidirectional, so its KV rows may be placed before
    the frozen observation KV. This leaves all valid keys in one contiguous
    prefix; FlashRT's existing ``seqused`` kernel can then skip state-prompt
    padding without a materialized attention mask. Only the action KV needs
    gradients; observation KV is frozen.
    """

    @staticmethod
    def forward(ctx, q, action_k, action_v, context_k, context_v,
                seqused_k, expert):
        q_fa2 = q.transpose(1, 2).contiguous()
        key = torch.cat((action_k, context_k), dim=0).unsqueeze(0).contiguous()
        value = torch.cat((action_v, context_v), dim=0).unsqueeze(0).contiguous()
        batch, seqlen_q, heads_q, head_dim = q_fa2.shape
        seqlen_k, heads_kv = key.shape[1:3]
        if batch != 1 or head_dim != 256 or heads_kv != 1:
            raise ValueError('FA2 RTC attention expects batch=1, GQA 8:1, head_dim=256')
        out = torch.empty_like(q_fa2)
        # FA2 stores LSE densely with a per-head stride of seqlen_q (the
        # backend's oversized rounded buffer is only a capacity allocation).
        lse = torch.empty((batch, heads_q, seqlen_q),
                          device=q.device, dtype=torch.float32)
        max_splits = max(1, min(128, (seqlen_k + 63) // 64))
        lse_accum = torch.empty((max_splits, batch, heads_q, seqlen_q),
                                device=q.device, dtype=torch.float32)
        o_accum = torch.empty((max_splits, batch, heads_q, seqlen_q, head_dim),
                              device=q.device, dtype=torch.float32)
        lse_accum.fill_(float('-inf'))
        stream = torch.cuda.current_stream(q.device).cuda_stream
        expert._fa2.fwd_bf16_seqused_splitkv(
            Q=q_fa2.data_ptr(), K=key.data_ptr(), V=value.data_ptr(),
            O=out.data_ptr(), softmax_lse=lse.data_ptr(),
            seqused_k=seqused_k.data_ptr(), batch=batch,
            softmax_lse_accum=lse_accum.data_ptr(), o_accum=o_accum.data_ptr(),
            seqlen_q=seqlen_q, seqlen_k=seqlen_k,
            num_heads_q=heads_q, num_heads_kv=heads_kv, head_dim=head_dim,
            q_strides=(q_fa2.stride(0), q_fa2.stride(1), q_fa2.stride(2)),
            k_strides=(key.stride(0), key.stride(1), key.stride(2)),
            v_strides=(value.stride(0), value.stride(1), value.stride(2)),
            o_strides=(out.stride(0), out.stride(1), out.stride(2)),
            softmax_scale=head_dim ** -0.5,
            num_sms=expert._fa2_num_sms, stream=stream)
        ctx.save_for_backward(q, key, value, out, lse, seqused_k)
        ctx.expert = expert
        return out.transpose(1, 2)

    @staticmethod
    def backward(ctx, grad_out):
        q, key, value, out, lse, seqused_k = ctx.saved_tensors
        batch, heads_q, seqlen_q, head_dim = q.shape
        heads_kv = key.shape[2]
        seqlen_k = key.shape[1]
        group = heads_q // heads_kv
        scale = head_dim ** -0.5

        qf = q.float()
        # K/V are GQA-shared across query-head groups. Use broadcast views
        # rather than materializing eight copies of the full observation KV.
        kf = key.transpose(1, 2).float().expand(
            batch, heads_q, seqlen_k, head_dim)
        vf = value.transpose(1, 2).float().expand(
            batch, heads_q, seqlen_k, head_dim)
        go = grad_out.float()
        logits = torch.matmul(qf, kf.transpose(-2, -1)) * scale
        valid_keys = (torch.arange(seqlen_k, device=q.device).view(1, 1, 1, -1)
                      < seqused_k.view(1, 1, 1, 1))
        log_probs = torch.where(
            valid_keys,
            logits - lse[:, :, :seqlen_q].unsqueeze(-1),
            torch.full_like(logits, float('-inf')))
        probs = torch.exp(log_probs)
        out_f = out.transpose(1, 2).float()
        dprob = torch.matmul(go, vf.transpose(-2, -1))
        delta = (go * out_f).sum(dim=-1, keepdim=True)
        dscore = probs * (dprob - delta)
        dq = torch.matmul(dscore, kf) * scale
        # Observation K/V are constants for RTC: only action-side KV receives
        # gradients. Avoid calculating and allocating gradients for the much
        # longer frozen image/language prefix.
        action_len = seqlen_q
        dscore_action = dscore[..., :action_len]
        probs_action = probs[..., :action_len]
        dk_action = torch.matmul(dscore_action.transpose(-2, -1), qf) * scale
        dv_action = torch.matmul(probs_action.transpose(-2, -1), go)
        dk_action = dk_action.reshape(
            batch, heads_kv, group, action_len, head_dim).sum(dim=2)
        dv_action = dv_action.reshape(
            batch, heads_kv, group, action_len, head_dim).sum(dim=2)
        dk_action = dk_action.transpose(1, 2).squeeze(0).to(key.dtype)
        dv_action = dv_action.transpose(1, 2).squeeze(0).to(value.dtype)
        return (dq.to(q.dtype), dk_action, dv_action,
                None, None, None, None)


class DifferentiableExpert:
    """Single-sample, 32D, RTX BF16 expert with gradients only through x.

    ``pipe`` is the frontend owning ``_ckpt_bf16`` and ``pipeline``. Call
    snapshot_context AFTER context replay and prompt/RoPE setup, outside any
    inference_mode scope. Caller must serialize snapshotting with all FlashRT
    replays/prompt changes. The method fences CUDA but cannot lock other
    threads. No snapshot is inferred automatically from possibly stale KV.

    step_delta returns the already dt-scaled BF16 update; forward repeats all
    N steps and BF16 residual rounding, returning the raw Hx32 final chunk.
    Both Hx32 and 1xHx32 inputs are accepted and preserve their rank.
    """

    def __init__(self, pipe, *, attention_backend='math'):
        self.pipe = pipe
        self.pipeline = pipe.pipeline
        pl = self.pipeline
        if pl is None:
            raise RuntimeError('prepare the FlashRT pipeline before creating expert')
        for flag in ('use_fp8_decoder', 'use_int8_decoder'):
            if getattr(pl, flag, False):
                raise ValueError('DifferentiableExpert requires a BF16 decoder')
        self.chunk_size = int(pl.chunk_size)
        self.num_steps = int(pl.num_steps)
        if self.num_steps <= 0:
            raise ValueError('num_steps must be positive')
        names = ('attn_qkv_w', 'attn_o_w', 'ffn_gate_w', 'ffn_up_w',
                 'ffn_down_w', 'action_in_proj_w', 'action_in_proj_b',
                 'action_out_proj_w', 'action_out_proj_b')
        # detach aliases storage and does not alter the frontend's tensors.
        self.weights = {name: pipe._ckpt_bf16['decoder_' + name].detach()
                        for name in names}
        self.device = self.weights['action_in_proj_w'].device
        self.dtype = torch.bfloat16
        self.set_attention_backend(attention_backend)
        expected = {
            'attn_qkv_w': (18, 1024, 2560),
            'attn_o_w': (18, 2048, 1024),
            'ffn_gate_w': (18, 1024, 4096),
            'ffn_up_w': (18, 1024, 4096),
            'ffn_down_w': (18, 4096, 1024),
            'action_in_proj_w': (32, 1024),
            'action_in_proj_b': (1024,),
            'action_out_proj_w': (1024, 32),
            'action_out_proj_b': (32,),
        }
        for name, weight in self.weights.items():
            if (weight.dtype != self.dtype or weight.device != self.device
                    or tuple(weight.shape) != expected[name]):
                raise ValueError(f'unsupported decoder tensor {name}: '
                                 f'{tuple(weight.shape)}, {weight.dtype}, {weight.device}')
        self._context = None

    def set_attention_backend(self, name):
        """Select the math, PyTorch Flash SDPA, or FlashRT FA2 backend."""
        backends = {
            'math': SDPBackend.MATH,
            'flash': SDPBackend.FLASH_ATTENTION,
        }
        if name not in backends and name not in ('fa2-seqused', 'flash-varlen'):
            raise ValueError("attention backend must be math, flash, flash-varlen, or fa2-seqused")
        if name == 'flash-varlen' and not all(hasattr(torch.ops.aten, op) for op in
                                             ('_flash_attention_forward', '_flash_attention_backward')):
            raise RuntimeError('installed Torch lacks varlen FlashAttention forward/backward')
        self.attention_backend_name = name
        self.attention_backend = backends.get(name)
        if name == 'fa2-seqused':
            from flash_rt import flash_rt_fa2
            if not hasattr(flash_rt_fa2, 'fwd_bf16_seqused_splitkv'):
                raise RuntimeError('FlashRT FA2 seqused split-KV kernel is unavailable')
            self._fa2 = flash_rt_fa2
            self._fa2_num_sms = torch.cuda.get_device_properties(
                self.device).multi_processor_count

    def _download_bf16(self, name, shape):
        """Read exact CudaBuffer bits; no writes or guessed host RoPE math."""
        buf = self.pipeline.bufs[name]
        bits = np.empty(shape, dtype=np.uint16)
        if bits.nbytes != buf.nbytes:
            raise ValueError(f'unexpected {name} buffer size')
        buf.download(bits)
        return torch.from_numpy(bits).view(torch.bfloat16).to(self.device)

    def snapshot_context(self, *, fixed_prefix_capacity=False):
        """Freeze valid prefix KV, actual decoder RoPE and actual style rows.

        Includes state tokens already encoded by FlashRT. Excludes ALL cached
        action rows, even when called after vanilla decode. Existing K already
        has encoder RMS folding and RoPE applied: never rotate/normalize again.
        Host downloads of small tables are deliberate probe overhead.

        ``fixed_prefix_capacity`` keeps the pipeline's full fixed-shape encoder
        allocation and masks padding. It is intended for CUDA graph replay when
        state-token length can vary between observations.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError('snapshot_context must run outside inference_mode')
        if self.pipe.pipeline is not self.pipeline:
            raise RuntimeError('pipeline replaced; create a new DifferentiableExpert')
        pl = self.pipeline
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        attn = pl.attn
        prefix = (int(attn.dec_devpos.item()) if pl._fixed_shape
                  else int(pl.encoder_seq_len))
        keys, values = attn.enc_K, attn.enc_V
        use_prefix_mask = bool(fixed_prefix_capacity and pl._fixed_shape)
        capacity = int(pl.encoder_seq_len) if use_prefix_mask else prefix
        if (prefix <= 0 or keys.ndim != 4 or values.shape != keys.shape
                or keys.shape[0] != 18 or tuple(keys.shape[2:]) != (1, 256)
                or capacity < prefix or capacity + self.chunk_size > keys.shape[1]
                or keys.dtype != self.dtype or values.dtype != self.dtype
                or keys.device != self.device or values.device != self.device):
            raise ValueError('unsupported or invalid context KV layout/valid length')
        n, h = self.num_steps, self.chunk_size
        # Build locally then publish together; failed download keeps old snapshot.
        with torch.no_grad():
            context = {
                'key': keys[:, :capacity].detach().clone(),
                'value': values[:, :capacity].detach().clone(),
                'seqused_k': torch.full(
                    (1,), prefix + h, device=self.device, dtype=torch.int32),
                'cum_q': torch.tensor([0, h], device=self.device, dtype=torch.int32),
                'cum_k': torch.tensor([0, prefix+h], device=self.device, dtype=torch.int32),
                'rope': self._download_bf16('decoder_rope_weights', (h, 256)),
                'attn': self._download_bf16('decoder_style_attn', (n, 18, h, 3072)),
                'ffn': self._download_bf16('decoder_style_ffn', (n, 18, h, 3072)),
                'final': self._download_bf16('decoder_style_final', (n, h, 3072)),
            }
            if use_prefix_mask:
                context['attn_mask'] = torch.zeros(
                    (1, 1, 1, capacity + h), device=self.device,
                    dtype=torch.bool)
                context['attn_mask'][..., :prefix] = True
                context['attn_mask'][..., capacity:] = True
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self._context = context
        self.prefix_length = prefix
        self.context_capacity = capacity
        self._rope_by_prefix = {prefix: context['rope'].detach().clone()}
        return self

    def refresh_context(self, keys, values, valid_prefix):
        """Refresh KV, valid length and the matching decoder RoPE in place.

        All tensor addresses/shapes remain stable for an already captured CUDA
        graph; prefix-specific RoPE values are cached when a new prompt length
        first appears, then copied into the captured graph's stable buffer.
        """
        if self._context is None:
            raise RuntimeError('call snapshot_context before refreshing context')
        prefix = int(valid_prefix)
        capacity = self.context_capacity
        if (prefix <= 0 or prefix > capacity
                or tuple(keys.shape[:2]) != (18, capacity)
                or values.shape != keys.shape
                or keys.dtype != self.dtype or values.dtype != self.dtype
                or keys.device != self.device or values.device != self.device):
            raise ValueError('refreshed context KV has unsupported shape or valid length')
        if prefix not in self._rope_by_prefix:
            self._rope_by_prefix[prefix] = self._download_bf16(
                'decoder_rope_weights', (self.chunk_size, 256)).detach()
        with torch.no_grad():
            self._context['key'].copy_(keys)
            self._context['value'].copy_(values)
            self._context['rope'].copy_(self._rope_by_prefix[prefix])
            if 'attn_mask' in self._context:
                mask = self._context['attn_mask']
                mask.fill_(False)
                mask[..., :prefix] = True
                mask[..., capacity:] = True
            self._context['seqused_k'].fill_(prefix + self.chunk_size)
            self._context['cum_k'][1].fill_(prefix + self.chunk_size)
        self.prefix_length = prefix

    @staticmethod
    def _ada(x, style):
        scale, shift, gate = style.float().chunk(3, dim=-1)
        xf = x.float()
        norm = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        return (norm * (1.0 + scale) + shift).to(x.dtype), gate

    @staticmethod
    def _rope(x, rope):
        # x: H, heads, 256; table: H, 256 alternating cos/sin.
        pair = x.float().reshape(*x.shape[:-1], 128, 2)
        cs = rope.float().reshape(rope.shape[0], 1, 128, 2)
        a, b = pair[..., 0], pair[..., 1]
        c, s = cs[..., 0], cs[..., 1]
        return torch.stack((a * c - b * s, b * c + a * s), -1).flatten(-2).to(x.dtype)

    @staticmethod
    def _geglu(gate, up):
        # Same logistic GELU polynomial as the expf kernel. Use stable sigmoid
        # for backward: literal division by (1+exp(...)) yields inf*0 NaN
        # gradients in negative tails. Sigmoid/expf rounding needs GPU gating.
        g = gate.float()
        return (g * torch.sigmoid(
            1.5957691216057308 * g * (1.0 + 0.044715 * g * g))
                * up.float()).to(gate.dtype)

    def _validate_x(self, x):
        if self._context is None:
            raise RuntimeError('call snapshot_context after FlashRT context replay')
        if torch.is_inference_mode_enabled():
            raise RuntimeError('expert forward must run outside inference_mode')
        if (not isinstance(x, torch.Tensor) or not x.is_floating_point()
                or x.device != self.device
                or tuple(x.shape) not in ((self.chunk_size, 32),
                                          (1, self.chunk_size, 32))):
            raise ValueError('x must be Hx32 or 1xHx32 floating tensor on weight device')

    def step_delta(self, x, step):
        """One differentiable raw update, already multiplied by dt=-1/N.

        No extra dt multiplication, action slicing, normalization or guidance.
        Casting floating x to BF16 preserves autograd connectivity to x.
        """
        self._validate_x(x)
        step = operator.index(step)
        if not 0 <= step < self.num_steps:
            raise ValueError('step outside precomputed time schedule')
        # Disable inherited autocast so all GEMMs really use BF16 inputs and
        # FP32 elementwise math matches the source kernel's precision policy.
        with torch.autocast(device_type=self.device.type, enabled=False):
            result = self._step_delta(x.reshape(self.chunk_size, 32).to(self.dtype), step)
        return result.reshape(x.shape)

    def _step_delta(self, x, step):
        w, ctx = self.weights, self._context
        hidden = ((x @ w['action_in_proj_w']).float()
                  + w['action_in_proj_b'].float()).to(self.dtype)
        for i in range(18):
            norm, gate = self._ada(hidden, ctx['attn'][step, i])
            qkv = norm @ w['attn_qkv_w'][i]
            q = self._rope(qkv[:, :2048].reshape(self.chunk_size, 8, 256), ctx['rope'])
            k = self._rope(qkv[:, 2048:2304].reshape(self.chunk_size, 1, 256), ctx['rope'])
            v = qkv[:, 2304:].reshape(self.chunk_size, 1, 256)
            q = q.transpose(0, 1).unsqueeze(0)
            if self.attention_backend_name == 'flash-varlen':
                attended = _TorchVarlenGqaAttention.apply(
                    q, k, v, ctx['key'][i], ctx['value'][i], ctx['cum_q'], ctx['cum_k'])
            elif self.attention_backend_name == 'fa2-seqused':
                # The suffix is bidirectional; moving it before the context
                # preserves the exact attention set while making valid keys a
                # contiguous prefix for the device-side seqused mask.
                attended = _FlashRtSequsedGqaAttention.apply(
                    q, k, v, ctx['key'][i], ctx['value'][i],
                    ctx['seqused_k'], self)
            else:
                k = torch.cat((ctx['key'][i], k), 0).transpose(0, 1).unsqueeze(0)
                v = torch.cat((ctx['value'][i], v), 0).transpose(0, 1).unsqueeze(0)
                # Eight Q heads share one KV head. Full bidirectional suffix,
                # no causal mask; math supports the dynamic fixed-capacity mask.
                with sdpa_kernel(self.attention_backend):
                    attended = F.scaled_dot_product_attention(
                        q, k, v, dropout_p=0.0, is_causal=False,
                        scale=256 ** -0.5, enable_gqa=True,
                        attn_mask=ctx.get('attn_mask'))
            attended = attended.squeeze(0).transpose(0, 1).reshape(self.chunk_size, 2048)
            projected = attended @ w['attn_o_w'][i]
            hidden = (hidden.float() + projected.float() * gate).to(self.dtype)
            norm, gate = self._ada(hidden, ctx['ffn'][step, i])
            activated = self._geglu(norm @ w['ffn_gate_w'][i], norm @ w['ffn_up_w'][i])
            projected = activated @ w['ffn_down_w'][i]
            hidden = (hidden.float() + projected.float() * gate).to(self.dtype)
        norm, _ = self._ada(hidden, ctx['final'][step])
        return ((norm @ w['action_out_proj_w']).float()
                + w['action_out_proj_b'].float()).to(self.dtype)

    def forward(self, x):
        """N-step vanilla decode with the supplied SAME initial noise."""
        self._validate_x(x)
        result = x.to(self.dtype)
        for step in range(self.num_steps):
            delta = self.step_delta(result, step)
            result = (result.float() + delta.float()).to(self.dtype)
        return result

    __call__ = forward
