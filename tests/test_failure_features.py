from types import SimpleNamespace

import torch

from perceptbudget.failure_features import ExpertFeatureTap


def test_feature_read_does_not_change_projection_or_input_gradient():
    torch.manual_seed(3)
    layer=torch.nn.Linear(16,7).eval().requires_grad_(False)
    expert=SimpleNamespace(suffix=SimpleNamespace(action_out_proj=layer))
    x=torch.randn(1,5,16,requires_grad=True)
    reference=layer(x)
    expected,=torch.autograd.grad(reference.sum(),x)
    tap=ExpertFeatureTap(expert,torch)
    try:
        tap.reset()
        output=layer(x)
        actual,=torch.autograd.grad(output.sum(),x)
        assert torch.equal(output,reference)
        assert torch.equal(actual,expected)
        assert not tap.last.requires_grad and tap.last.grad_fn is None
        assert torch.allclose(torch.from_numpy(tap.read(1)[0]),x.detach().mean((0,1)))
    finally:
        tap.close()


def test_denoise_aggregation_resets_between_requests():
    layer=torch.nn.Linear(8,7).eval().requires_grad_(False)
    tap=ExpertFeatureTap(SimpleNamespace(suffix=SimpleNamespace(action_out_proj=layer)),torch)
    try:
        tap.reset()
        layer(torch.ones(1,3,8))
        layer(torch.full((1,3,8),3.))
        result=torch.from_numpy(tap.read(2))
        assert torch.equal(result[0],torch.full((8,),3.))
        assert torch.equal(result[1],torch.full((8,),2.))
        tap.reset()
        layer(torch.full((1,3,8),5.))
        assert torch.equal(torch.from_numpy(tap.read(1)),torch.full((2,8),5.))
    finally:
        tap.close()
