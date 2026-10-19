from types import SimpleNamespace
import pytest
from perceptbudget.env_forward import reuse_control_forward


def test_only_control_forward_skipped_and_restored_on_error():
    calls = []
    sim = SimpleNamespace(forward=lambda: calls.append('forward'))
    inner = SimpleNamespace(sim=sim)
    inner._pre_action = lambda: sim.forward()
    env = SimpleNamespace(env=inner)
    old_forward, old_pre = sim.forward, inner._pre_action
    with pytest.raises(RuntimeError):
        with reuse_control_forward(env) as audit:
            sim.forward()
            inner._pre_action()
            assert calls == ['forward']
            assert audit['skipped_forward_calls'] == 1
            raise RuntimeError('restore')
    assert sim.forward is old_forward and inner._pre_action is old_pre
