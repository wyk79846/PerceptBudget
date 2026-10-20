import pytest


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('prefix,n', [(2, 3), (19, 5)])
def test_suffix_matmuls_preserve_all_action_gradients(dtype, prefix, n):
    import torch
    from perceptbudget.rtc_suffix_vjp import _SuffixQK, _SuffixAV
    dtype = getattr(torch, dtype)
    torch.manual_seed(4)
    q = torch.randn(1, 2, n, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 2, n, 8, dtype=dtype, requires_grad=True)
    v = torch.randn(1, 2, n, 8, dtype=dtype, requires_grad=True)
    kp = torch.randn(1, 2, prefix, 8, dtype=dtype)
    vp = torch.randn_like(kp)
    key, value = torch.cat((kp, k), -2), torch.cat((vp, v), -2)
    seed = torch.randn_like(q)

    def run(candidate):
        scores = _SuffixQK.apply(q, key, n) if candidate else q @ key.transpose(-2, -1)
        weights = scores.softmax(-1)
        out = _SuffixAV.apply(weights, value, n) if candidate else weights @ value
        grads = torch.autograd.grad(out, (q, k, v), seed, retain_graph=True)
        return out, grads

    out, grads = run(False)
    got, actual = run(True)
    torch.testing.assert_close(out, got, rtol=0, atol=0)
    for a, b in zip(grads, actual):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
