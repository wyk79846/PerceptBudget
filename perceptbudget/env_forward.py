"""Opt-in experiment: reuse outer physics forward inside robot control only."""
from contextlib import contextmanager


@contextmanager
def reuse_control_forward(env):
    inner = env.env
    original_forward, original_pre = inner.sim.forward, inner._pre_action
    in_control = False
    audit = {'enabled': True, 'skipped_forward_calls': 0,
             'scope': 'process-local experimental controller forward reuse'}

    def forward(*args, **kwargs):
        if in_control:
            audit['skipped_forward_calls'] += 1
            return None
        return original_forward(*args, **kwargs)

    def pre_action(*args, **kwargs):
        nonlocal in_control
        previous = in_control
        in_control = True
        try:
            return original_pre(*args, **kwargs)
        finally:
            in_control = previous

    inner.sim.forward, inner._pre_action = forward, pre_action
    try:
        yield audit
    finally:
        inner.sim.forward, inner._pre_action = original_forward, original_pre
