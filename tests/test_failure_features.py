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


def test_corner_features_keep_action_and_denoising_endpoints_without_changing_gradients():
    layer=torch.nn.Linear(4,2).eval().requires_grad_(False)
    tap=ExpertFeatureTap(SimpleNamespace(suffix=SimpleNamespace(action_out_proj=layer)),torch,layout='corners')
    try:
        tap.reset()
        first=torch.arange(12,dtype=torch.float32).reshape(1,3,4).requires_grad_()
        expected_gradient=layer.weight.sum(0).expand_as(first)
        result=layer(first)
        gradient,=torch.autograd.grad(result.sum(),first)
        for step in range(1,5):layer(first.detach()+100*step)
        actual=torch.from_numpy(tap.read(5))
        expected=torch.stack((first.detach()[0,0],first.detach()[0,-1],
                              first.detach()[0,0]+400,first.detach()[0,-1]+400))
        assert torch.equal(actual,expected)
        assert torch.equal(gradient,expected_gradient)
        assert not tap.first.requires_grad and tap.first.grad_fn is None
        tap.reset();layer(first.detach()+1000)
        expected=torch.stack((first.detach()[0,0]+1000,first.detach()[0,-1]+1000))
        assert torch.equal(torch.from_numpy(tap.read(1)),expected.repeat(2,1))
    finally:
        tap.close()
