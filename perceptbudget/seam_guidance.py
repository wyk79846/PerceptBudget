"""SEAM-style velocity-guided loss steering for action-chunk overlap.

This project-side implementation applies the paper's closed-form VLS update
to a differentiable FlashRT expert replica. It does not backpropagate through
the policy. Async callers pass the old chunk tail that is still queued when a
new observation is submitted; expired actions are cropped by the controller.
"""

import math


def aligned_prior(previous_tail, horizon):
    """Pad an old chunk tail to H by repeating its last action, as in SEAM."""
    import torch

    if (not isinstance(previous_tail, torch.Tensor) or previous_tail.ndim != 2
            or horizon < 1 or previous_tail.shape[0] < 1
            or previous_tail.shape[1] < 1 or previous_tail.shape[0] > horizon):
        raise ValueError('previous_tail must be a nonempty LxD tensor with L<=H')
    if not torch.isfinite(previous_tail).all():
        raise ValueError('previous_tail must be finite')
    result = torch.empty((horizon, previous_tail.shape[1]),
                         device=previous_tail.device, dtype=previous_tail.dtype)
    result[:len(previous_tail)].copy_(previous_tail)
    if len(previous_tail) < horizon:
        result[len(previous_tail):].copy_(previous_tail[-1])
    return result


def vls_step(candidate, prior, *, step, num_steps, strength=0.1,
             guided_window=None):
    """Apply one paper Eq. 8 correction after the ordinary Euler update."""
    import torch

    if (candidate.ndim != 2 or prior.shape != candidate.shape
            or num_steps < 1 or not 0 <= step < num_steps
            or not math.isfinite(strength) or strength < 0):
        raise ValueError('invalid SEAM candidate, prior, step, or strength')
    window = candidate.shape[0] if guided_window is None else int(guided_window)
    if not 0 <= window <= candidate.shape[0]:
        raise ValueError('guided_window must be within the action horizon')
    result = candidate.clone()
    if window == 0 or strength == 0:
        return result

    # The paper writes t_i=1-i/N, so 1-t_{i+1}=(i+1)/N.
    alpha = (step + 1) / num_steps
    x = candidate[:window].float()
    target = alpha * prior[:window].float()
    correction = -2.0 * (x - target)
    result[:window] = (x + strength * alpha * correction).to(candidate.dtype)
    return result


def seam_sample(step_delta, noise, previous_tail, *, num_steps=10,
                strength=0.1, max_window=20):
    """Generate a chunk with SEAM VLS and no autograd/backward pass.

    ``step_delta(x, i)`` returns FlashRT's already dt-scaled Euler update.
    The first M=min(len(previous_tail), max_window) positions are guided.
    """
    import torch

    if (noise.ndim != 2 or noise.shape[1] != 32 or num_steps < 1
            or max_window < 0 or not math.isfinite(strength) or strength < 0):
        raise ValueError('expected Hx32 noise and valid SEAM settings')
    if (previous_tail.ndim != 2 or previous_tail.shape[1] != noise.shape[1]
            or not 1 <= previous_tail.shape[0] <= noise.shape[0]):
        raise ValueError('previous_tail must have shape (H-s)x32, with 0<s<H')
    window = min(int(previous_tail.shape[0]), int(max_window))
    prior = aligned_prior(previous_tail.detach(), noise.shape[0])
    model_dtype = noise.dtype
    state = noise.detach().clone()
    corrections = []
    with torch.no_grad():
        for step in range(num_steps):
            delta = step_delta(state, step)
            if delta.shape != state.shape or not delta.is_floating_point():
                raise ValueError('step_delta must return a floating tensor with noise shape')
            # FlashRT stores the Euler state in BF16 between denoising steps.
            candidate = (state.float() + delta.float()).to(model_dtype)
            updated = vls_step(candidate, prior, step=step,
                               num_steps=num_steps, strength=strength,
                               guided_window=window)
            state = updated.to(model_dtype)
            corrections.append(float((state.float() - candidate.float()).norm()))
    return state, {'guided_window': window,
                   'overlap_length': int(previous_tail.shape[0]),
                   'strength': float(strength),
                   'correction_norms': corrections}


class CapturedSeamChunk:
    """Capture the N-step SEAM loop once; noise and old tail stay runtime inputs."""

    def __init__(self, step_delta, sample, *, num_steps=10, strength=0.1,
                 max_window=20, warmups=2):
        import torch

        if (torch.is_inference_mode_enabled() or sample.device.type != 'cuda'
                or sample.ndim != 2 or sample.shape[1] != 32
                or sample.dtype != torch.bfloat16
                or num_steps < 1 or max_window < 0 or warmups < 1
                or not math.isfinite(strength) or strength < 0):
            raise ValueError('captured SEAM requires CUDA BF16 Hx32 input and valid settings')
        self.step_delta = step_delta
        self.num_steps = int(num_steps)
        self.strength = float(strength)
        self.max_window = int(max_window)
        self.model_dtype = sample.dtype
        self.noise = sample.detach().clone()
        self.prior = torch.zeros(sample.shape, device=sample.device,
                                 dtype=torch.float32)
        self.mask = torch.zeros((sample.shape[0], 1), device=sample.device,
                                dtype=torch.float32)
        self.output = torch.empty_like(sample)

        current = torch.cuda.current_stream(sample.device)
        side = torch.cuda.Stream(device=sample.device)
        side.wait_stream(current)
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(warmups):
                self._run()
        current.wait_stream(side)
        torch.cuda.synchronize(sample.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self._run()
        torch.cuda.synchronize(sample.device)

    def _run(self):
        import torch

        state = self.noise
        for step in range(self.num_steps):
            delta = self.step_delta(state, step)
            candidate = (state.float() + delta.float()).to(self.model_dtype)
            alpha = (step + 1) / self.num_steps
            target = alpha * self.prior
            correction = -2.0 * (candidate.float() - target) * self.mask
            state = (candidate.float() + self.strength * alpha * correction).to(
                self.model_dtype)
        self.output.copy_(state)
        return self.output

    def replay(self, noise, previous_tail):
        import torch

        if (tuple(noise.shape) != tuple(self.noise.shape)
                or noise.device != self.noise.device
                or noise.dtype != self.model_dtype
                or previous_tail.ndim != 2
                or previous_tail.shape[1] != self.noise.shape[1]
                or not 1 <= previous_tail.shape[0] <= self.noise.shape[0]
                or previous_tail.device != self.noise.device):
            raise ValueError('SEAM runtime noise/tail shape or device mismatch')
        window = min(int(previous_tail.shape[0]), self.max_window)
        with torch.no_grad():
            self.noise.copy_(noise)
            self.prior[:len(previous_tail)].copy_(previous_tail.float())
            if len(previous_tail) < self.prior.shape[0]:
                self.prior[len(previous_tail):].copy_(previous_tail[-1].float())
            self.mask.zero_()
            self.mask[:window].fill_(1.)
        self.graph.replay()
        return self.output.detach().clone(), {
            'guided_window': window,
            'overlap_length': int(previous_tail.shape[0]),
            'strength': self.strength,
        }
