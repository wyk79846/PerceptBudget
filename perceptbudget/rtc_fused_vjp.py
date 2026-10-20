"""Opt-in fused input-VJP for the original pi05 suffix inside CUDA graphs.

Patches only this project's AST-built suffix instances; installed
Transformers and FlashRT are untouched:
- GeGLU: forward = bf16-rounded fp32 tanh-GELU (matches PytorchGELUTanh on
  bf16 tensors bit-exactly); backward = fused analytic FP32 kernel.
- RMSNorm+conditioning: forward unchanged (served by the packed conditioning
  cache views); backward = fused analytic FP32 formula.

Backward composition changes; forward GEMMs and all tensors' values are
unchanged. Every use must pass a live parity gate against the unpatched step
graph on identical inputs before the capture is accepted.
"""
import torch


def _tanh_gelu_act(gate):
    """bf16(fp32 tanh-GELU) — bit-exact match of PytorchGELUTanh on bf16."""
    g = gate.float()
    c = 0.7978845608028654
    act = 0.5 * g * (1 + torch.tanh(c * g * (1 + 0.044715 * g * g)))
    return act.to(gate.dtype)


class _FusedGeGLU(torch.autograd.Function):
    """act(gate)*up in the author's bf16 rounding; fused FP32 backward."""

    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return _tanh_gelu_act(gate) * up

    @staticmethod
    def backward(ctx, grad):
        from perceptbudget.rtc_fused_backward import geglu_tanh_backward
        gate, up = ctx.saved_tensors
        dg, du = geglu_tanh_backward(gate, up, grad)
        return dg, du


def enable_fused_geglu(expert):
    """Swap suffix GeGLU composition for fused-backward versions.

    Idempotent; call before graph capture. Returns count of patched layers.
    The down_proj stays outside the custom Function so its GEMM and its
    backward keep the exact original ops.
    """
    model = expert.suffix.paligemma_with_expert.gemma_expert.model
    patched = 0
    for layer in model.layers:
        mlp = layer.mlp
        if getattr(mlp, '_pb_fused_geglu', False):
            continue
        gate_proj, up_proj, down_proj = mlp.gate_proj, mlp.up_proj, mlp.down_proj

        def make_forward(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj):
            def forward(x):
                return down_proj(_FusedGeGLU.apply(gate_proj(x), up_proj(x)))
            return forward

        mlp.forward = make_forward()
        mlp._pb_fused_geglu = True
        patched += 1
    return patched
