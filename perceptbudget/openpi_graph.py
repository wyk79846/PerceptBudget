"""CUDA replay of unchanged original OpenPI expert + author RTC arithmetic.

This initial implementation binds a single context shape. It must never be
reused after context tensors are replaced; callers must validate bindings.
"""


class CapturedOriginalRtc:
    def __init__(self, expert, noise, warmups=2):
        import torch
        self.expert = expert
        expert.enable_graph_constants()
        # Keep external capture inputs alive: allocator address reuse must not
        # make a replaced context appear to be the old binding.
        self.context = (expert.keys, expert.values, expert.prefix_mask)
        self.binding = tuple(t.data_ptr() for t in (expert.keys, expert.values, expert.prefix_mask))
        self.noise = noise.detach().float().clone().requires_grad_(True)
        self.target = torch.zeros_like(self.noise)
        self.weights = torch.zeros(expert.horizon, 1, device=noise.device)
        self.beta = 5.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                self._run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.enable_grad():
            self.outputs = self._run()
        torch.cuda.synchronize()

    def _run(self):
        import torch
        x = self.noise
        velocities, corrections = [], []
        steps = self.expert.num_steps
        for step in range(steps):
            tau = step/steps
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                velocity = -self.expert.velocity(x, step)
                clean = x+(1-tau)*velocity
                residual = ((self.target-clean)*self.weights).detach()
                correction, = torch.autograd.grad(clean, x, grad_outputs=residual)
                gain = self.beta if step == 0 else min(self.beta, (tau*tau+(1-tau)**2)/(tau*(1-tau)))
                x = (x+(1/steps)*(velocity+gain*correction)).detach()
                velocities.append(-velocity.detach())
                corrections.append(correction.detach())
        return x, torch.stack(velocities), torch.stack(corrections)

    def replay(self, noise, previous, delay, executed):
        import math
        import torch
        h = self.expert.horizon
        current = tuple(t.data_ptr() for t in (self.expert.keys, self.expert.values, self.expert.prefix_mask))
        if current != self.binding:
            raise RuntimeError('captured original context was replaced; refusing stale KV replay')
        if not 0 <= executed <= h or not 0 <= delay <= h-executed or previous.shape != (h-executed,32):
            raise ValueError('invalid RTC overlap')
        if noise.shape != self.noise.shape:
            raise ValueError('noise shape mismatch')
        i = torch.arange(h, device=noise.device, dtype=torch.float32)
        weights = ((delay-1-i)/(h-executed-delay+1)+1).clamp(0,1)
        weights = torch.where(i>=h-executed, 0, weights*torch.expm1(weights)/(math.e-1))[:,None]
        with torch.no_grad():
            self.noise.copy_(noise)
            self.target.zero_()
            self.target[:len(previous)].copy_(previous)
            self.weights.copy_(weights)
        self.graph.replay()
        return self.outputs
