"""Experimental native FFN forward + input-only backward; not rollout backend.

Native kernels run on Torch's current stream into fresh, Torch-owned buffers.
Weights/style are frozen. Backward uses existing Torch GPU operators; no
weight gradients, finite-difference approximations, or straight-through VJP.
"""
import torch


class NativeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, gemm):
        out = torch.empty((x.shape[0], weight.shape[1]), device=x.device, dtype=x.dtype)
        gemm.bf16_nn(x.data_ptr(), weight.data_ptr(), out.data_ptr(),
                     x.shape[0], weight.shape[1], x.shape[1],
                     stream=torch.cuda.current_stream().cuda_stream)
        ctx.save_for_backward(weight)
        return out

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        return grad @ weight.T, None, None


class NativeAda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, style, ones, fvk):
        out, gate = torch.empty_like(x), torch.empty_like(x)
        fvk.ada_rms_norm_style(x.data_ptr(), ones.data_ptr(), style.data_ptr(),
            out.data_ptr(), gate.data_ptr(), *x.shape, 1e-6,
            stream=torch.cuda.current_stream().cuda_stream)
        ctx.save_for_backward(x, style)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, style = ctx.saved_tensors
        xf = x.float()
        scale = style.float().chunk(3, -1)[0]
        r = torch.rsqrt(xf.square().mean(-1, keepdim=True)+1e-6)
        z = grad.float()*(1+scale)
        dx = z*r-xf*r.pow(3)*(z*xf).mean(-1, keepdim=True)
        return dx.to(x.dtype), None, None, None


class NativeGeGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, fvk):
        out = torch.empty_like(gate)
        fvk.gate_geglu(gate.data_ptr(), up.data_ptr(), out.data_ptr(), gate.numel(),
                      stream=torch.cuda.current_stream().cuda_stream)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        g, u, dy = gate.float(), up.float(), grad.float()
        p = torch.sigmoid(1.5957691216057308*g*(1+0.044715*g*g))
        dg = dy*u*(p+g*p*(1-p)*1.5957691216057308*(1+3*0.044715*g*g))
        du = dy*g*p
        return dg.to(gate.dtype), du.to(up.dtype), None


class FusedAda(NativeAda):
    @staticmethod
    def backward(ctx, grad):
        from perceptbudget.rtc_fused_backward import ada_backward
        x, style = ctx.saved_tensors
        return ada_backward(x,style,grad), None, None, None


class FusedGeGLU(NativeGeGLU):
    @staticmethod
    def backward(ctx, grad):
        from perceptbudget.rtc_fused_backward import geglu_backward
        gate, up = ctx.saved_tensors
        dg, du = geglu_backward(gate,up,grad)
        return dg,du,None


def ffn(x, style, weights, *, native=False, gemm=None, fvk=None, ones=None,
        include_residual=True, fused_backward=False):
    """One real-width FFN sublayer, including Ada norm and gated residual."""
    from perceptbudget.rtc_expert import DifferentiableExpert
    gate = style.float().chunk(3, -1)[2]
    if native:
        norm = (FusedAda if fused_backward else NativeAda).apply(x, style, ones, fvk)
        a = NativeLinear.apply(norm, weights[0], gemm)
        b = NativeLinear.apply(norm, weights[1], gemm)
        act = (FusedGeGLU if fused_backward else NativeGeGLU).apply(a, b, fvk)
        projected = NativeLinear.apply(act, weights[2], gemm)
    else:
        norm, _ = DifferentiableExpert._ada(x, style)
        act = DifferentiableExpert._geglu(norm @ weights[0], norm @ weights[1])
        projected = act @ weights[2]
    branch = projected.float()*gate
    return ((x.float()+branch) if include_residual else branch).to(x.dtype)
