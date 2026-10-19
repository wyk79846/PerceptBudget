def build_expert():
    import torch
    from types import SimpleNamespace
    class Suffix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pi05=True
            self.time_mlp_in=torch.nn.Linear(4,4)
            self.time_mlp_out=torch.nn.Linear(4,4)
            self.dense=torch.nn.Linear(4,12)
            self.paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(
                model=SimpleNamespace(layers=[],norm=SimpleNamespace(dense=self.dense))))
        def embed_suffix(self,state,dummy,t):
            cond=self.time_mlp_out(torch.nn.functional.silu(self.time_mlp_in(t.expand(1,4))))
            return None,None,None,cond
    suffix=Suffix().eval().requires_grad_(False)
    return SimpleNamespace(suffix=suffix,horizon=25,num_steps=5,device='cpu',
                           times=[torch.tensor([1-k/5]) for k in range(5)])


def test_condition_cache_outputs_exact_and_preserves_activation_gradients():
    import torch
    from perceptbudget.rtc_condition_cache import enable_condition_cache
    expert=build_expert()
    suffix=expert.suffix
    expected=[]
    for t in expert.times:
        cond=suffix.embed_suffix(None,None,t)[3]
        expected.append(suffix.dense(cond))
    metadata=enable_condition_cache(expert)
    assert metadata['modules']==3
    for step,t in enumerate(expert.times):
        expert._rtc_condition_step=step
        cond=suffix.embed_suffix(None,None,t)[3]
        scale=suffix.dense(cond)
        torch.testing.assert_close(scale,expected[step],atol=0,rtol=0)
        x=torch.ones_like(scale,requires_grad=True)
        dx,=torch.autograd.grad((x*scale).sum(),x)
        torch.testing.assert_close(dx,expected[step],atol=0,rtol=0)


def test_condition_cache_rejects_grad_input_and_modified_weights():
    import torch
    import pytest
    from perceptbudget.rtc_condition_cache import enable_condition_cache
    expert=build_expert()
    enable_condition_cache(expert)
    expert._rtc_condition_step=0
    with pytest.raises(RuntimeError,match='activation-dependent'):
        expert.suffix.time_mlp_in(torch.ones(1,4,requires_grad=True))
    with torch.no_grad():
        expert.suffix.time_mlp_in.weight.add_(1)
    with pytest.raises(RuntimeError,match='weights changed'):
        expert.suffix.time_mlp_in(torch.ones(1,4))
