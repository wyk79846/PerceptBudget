"""Optional SAFE-style feature tap; no policy, model or FlashRT changes.

The action projection input is the final expert hidden state. The tap reads a
detached view, pools action positions and writes fixed buffers that also work
inside the existing CUDA graphs. Reset between requests, outside capture.
"""


class ExpertFeatureTap:
    def __init__(self, expert, torch):
        self.torch = torch
        layer = expert.suffix.action_out_proj
        width = layer.in_features
        self.last = torch.zeros(width, device=layer.weight.device, dtype=torch.float32)
        self.total = torch.zeros_like(self.last)
        self.calls = torch.zeros((), device=layer.weight.device, dtype=torch.int32)
        self.handle = layer.register_forward_pre_hook(self._record)

    def _record(self, module, inputs):
        x = inputs[0]
        with self.torch.no_grad():
            pooled = x.detach().float().mean(dim=tuple(range(x.ndim-1)))
            self.last.copy_(pooled)
            self.total.add_(pooled)
            self.calls.add_(1)

    def reset(self):
        self.last.zero_()
        self.total.zero_()
        self.calls.zero_()

    def read(self, expected_steps):
        actual = int(self.calls.item())
        if actual != expected_steps:
            raise RuntimeError(f'Feature tap observed {actual} forwards, expected {expected_steps}')
        return self.torch.stack((self.last, self.total/expected_steps)).cpu().numpy()

    def close(self):
        self.handle.remove()
