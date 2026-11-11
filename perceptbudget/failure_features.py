"""Optional SAFE-style feature tap; no policy, model or FlashRT changes.

The action projection input is the final expert hidden state. The tap reads a
detached view and stores action means or first/last position endpoints in fixed
buffers that work inside existing CUDA graphs. Reset outside capture.
"""


FEATURE_LAYOUTS={
    'pooled':['last_denoise_mean_actions','mean_denoise_mean_actions'],
    'corners':['first_denoise_first_action','first_denoise_last_action',
               'last_denoise_first_action','last_denoise_last_action'],
}


class ExpertFeatureTap:
    def __init__(self, expert, torch, *, layout='pooled'):
        if layout not in FEATURE_LAYOUTS:
            raise ValueError('Unknown expert feature layout')
        self.torch = torch
        self.layout = layout
        layer = expert.suffix.action_out_proj
        width = layer.in_features
        self.feature_shape=(len(FEATURE_LAYOUTS[layout]),width)
        shape=(width,) if layout=='pooled' else (2,width)
        self.last = torch.zeros(shape, device=layer.weight.device, dtype=torch.float32)
        if layout=='pooled':
            self.total = torch.zeros_like(self.last)
        else:
            self.first = torch.zeros_like(self.last)
        self.calls = torch.zeros((), device=layer.weight.device, dtype=torch.int32)
        self.handle = layer.register_forward_pre_hook(self._record)

    def _record(self, module, inputs):
        x = inputs[0]
        with self.torch.no_grad():
            if self.layout=='pooled':
                pooled = x.detach().float().mean(dim=tuple(range(x.ndim-1)))
                self.last.copy_(pooled)
                self.total.add_(pooled)
            else:
                flat=x.detach().reshape(-1,x.shape[-1])
                endpoints=self.torch.stack((flat[0],flat[-1])).float()
                # The device-side counter is evaluated during graph replay;
                # a Python first-call branch would freeze capture-time state.
                self.first.copy_(self.torch.where(self.calls==0,endpoints,self.first))
                self.last.copy_(endpoints)
            self.calls.add_(1)

    def reset(self):
        self.last.zero_()
        if self.layout=='pooled':
            self.total.zero_()
        else:
            self.first.zero_()
        self.calls.zero_()

    def read(self, expected_steps):
        actual = int(self.calls.item())
        if actual != expected_steps:
            raise RuntimeError(f'Feature tap observed {actual} forwards, expected {expected_steps}')
        if self.layout=='pooled':
            result=self.torch.stack((self.last, self.total/expected_steps))
        else:
            result=self.torch.cat((self.first,self.last),dim=0)
        return result.cpu().numpy()

    def close(self):
        self.handle.remove()
