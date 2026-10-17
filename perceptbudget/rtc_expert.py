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
        self.set_attention_backend(attention_backend)
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
        """Select the SDPA backend before building a new graph capture."""
        backends = {
            'math': SDPBackend.MATH,
            'flash': SDPBackend.FLASH_ATTENTION,
        }
        if name not in backends:
            raise ValueError("attention backend must be 'math' or 'flash'")
        self.attention_backend_name = name
        self.attention_backend = backends[name]

    def _download_bf16(self, name, shape):
        """Read exact CudaBuffer bits; no writes or guessed host RoPE math."""
        buf = self.pipeline.bufs[name]
        bits = np.empty(shape, dtype=np.uint16)
        if bits.nbytes != buf.nbytes:
            raise ValueError(f'unexpected {name} buffer size')
        buf.download(bits)
        return torch.from_numpy(bits).view(torch.bfloat16).to(self.device)

    def snapshot_context(self):
        """Freeze valid prefix KV, actual decoder RoPE and actual style rows.

        Includes state tokens already encoded by FlashRT. Excludes ALL cached
        action rows, even when called after vanilla decode. Existing K already
        has encoder RMS folding and RoPE applied: never rotate/normalize again.
        Host downloads of small tables are deliberate probe overhead.
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
        if (prefix <= 0 or keys.ndim != 4 or values.shape != keys.shape
                or keys.shape[0] != 18 or tuple(keys.shape[2:]) != (1, 256)
                or prefix + self.chunk_size > keys.shape[1]
                or keys.dtype != self.dtype or values.dtype != self.dtype
                or keys.device != self.device or values.device != self.device):
            raise ValueError('unsupported or invalid context KV layout/valid length')
        n, h = self.num_steps, self.chunk_size
        # Build locally then publish together; failed download keeps old snapshot.
        with torch.no_grad():
            context = {
                'key': keys[:, :prefix].detach().clone(),
                'value': values[:, :prefix].detach().clone(),
                'rope': self._download_bf16('decoder_rope_weights', (h, 256)),
                'attn': self._download_bf16('decoder_style_attn', (n, 18, h, 3072)),
                'ffn': self._download_bf16('decoder_style_ffn', (n, 18, h, 3072)),
                'final': self._download_bf16('decoder_style_final', (n, h, 3072)),
            }
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self._context = context
        self.prefix_length = prefix
        return self

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
            k = torch.cat((ctx['key'][i], k), 0).transpose(0, 1).unsqueeze(0)
            v = torch.cat((ctx['value'][i], v), 0).transpose(0, 1).unsqueeze(0)
            q = q.transpose(0, 1).unsqueeze(0)
            # Eight Q heads share one KV head. Full bidirectional suffix, no
            # causal mask; padding was removed by the valid-prefix snapshot.
            with sdpa_kernel(self.attention_backend):
                attended = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False,
                    scale=256 ** -0.5, enable_gqa=True)
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
