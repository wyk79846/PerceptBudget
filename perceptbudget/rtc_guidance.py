"""Inference-only RTC Eq. 2-5, using a differentiable scaled expert step.

Weights remain frozen. Only the current noisy action receives a VJP.
FlashRT integrates from t=1 to 0; paper tau is 1-t. step_delta must return
the existing Euler delta (-v_flashrt / N), not an unscaled velocity.
"""

import math


def soft_mask(horizon, delay, executed):
    """Paper Eq. 5. executed=s, delay=d, with d<=s<=H-d."""
    if not (0 <= delay <= executed <= horizon - delay and horizon > 0):
        raise ValueError('RTC requires 0 <= d <= s <= H-d')
    weights = []
    for index in range(horizon):
        if index < delay:
            weights.append(1.)
        elif index >= horizon - executed:
            weights.append(0.)
        else:
            c = (horizon - executed - index) / (horizon - executed - delay + 1)
            weights.append(c * math.expm1(c) / math.expm1(1.))
    return weights


def guidance_scale(tau, beta=5.):
    if not 0 <= tau < 1 or not math.isfinite(beta) or beta <= 0:
        raise ValueError('invalid RTC tau/beta')
    if tau == 0:
        return beta
    r2 = (1-tau)**2 / (tau*tau + (1-tau)**2)
    return min(beta, (1-tau)/(tau*r2))


def guided_sample(step_delta, noise, previous, *, delay, executed, num_steps=10, beta=5.):
    """Return (raw action, per-step diagnostics); no output averaging or clamp."""
    import torch
    if num_steps < 1 or noise.ndim != 2 or noise.shape[1] != 32:
        raise ValueError('expected Hx32 noise and positive N')
    if previous.shape != (noise.shape[0]-executed, noise.shape[1]):
        raise ValueError('previous must be remaining H-s raw32 actions')
    mask = torch.tensor(soft_mask(noise.shape[0], delay, executed),
                        dtype=torch.float32, device=noise.device)[:, None]
    target = torch.zeros_like(noise, dtype=torch.float32)
    target[:len(previous)] = previous.detach().float()
    x = noise.detach().float()
    diagnostics = []
    for step in range(num_steps):
        tau = step / num_steps
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            delta = step_delta(x, step).float()
            estimate = x + (1-tau) * num_steps * delta
            error = ((target - estimate).detach() * mask).detach()
            correction, = torch.autograd.grad(estimate, x, grad_outputs=error)
            scale = guidance_scale(tau, beta)
            next_x = x + delta + (scale / num_steps) * correction
        if not torch.isfinite(next_x).all():
            raise RuntimeError('nonfinite RTC update')
        diagnostics.append({'step': step, 'tau': tau, 'scale': scale,
                            'vjp_norm': float(correction.detach().norm()),
                            'weighted_error_norm': float(error.norm())})
        x = next_x.detach()
    return x, diagnostics
