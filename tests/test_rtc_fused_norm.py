def test_norm_wrapper_preserves_fp32_modulation_gate_and_restoration(monkeypatch):
    import torch
    from types import SimpleNamespace
    from perceptbudget import rtc_fused_norm as fused

    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense=torch.nn.Linear(8,24)
            self.eps=1e-6

        def forward(self,x,cond=None):
            style=self.dense(cond).unsqueeze(1)
            scale,shift,gate=style.chunk(3,dim=-1)
            r=(x.float().square().mean(-1,keepdim=True)+self.eps).rsqrt()
            return ((x*r)*(1+scale)+shift).to(x.dtype),gate.to(x.dtype)

    model=torch.nn.Module()
    layer=torch.nn.Module()
    layer.input_layernorm=Norm();layer.post_attention_layernorm=Norm()
    model.layers=torch.nn.ModuleList([layer]);model.norm=Norm()
    model.eval().requires_grad_(False)
    expert=SimpleNamespace(suffix=SimpleNamespace(parameters=model.parameters,
        paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(model=model))))
    monkeypatch.setattr(fused,'load',lambda root:None)
    seen=[]

    def kernel(x,style,eps):
        assert style.dtype==torch.float32 and not style.requires_grad
        seen.append(style)
        scale,shift,_=style.unsqueeze(1).chunk(3,-1)
        return ((x*(x.float().square().mean(-1,keepdim=True)+eps).rsqrt())*(1+scale)+shift).to(x.dtype)

    monkeypatch.setattr(fused,'fused_norm',kernel)
    module=layer.input_layernorm
    original=module.forward
    x=torch.randn(1,4,8,dtype=torch.bfloat16,requires_grad=True)
    cond=torch.randn(1,8)
    y,gate=module(x,cond)
    expected,=torch.autograd.grad(y.sum(),x)
    restore,meta=fused.enable_fused_norm(expert,'/tmp/unused')
    got,g=module(x,cond)
    actual,=torch.autograd.grad(got.sum(),x)
    torch.testing.assert_close(y,got,rtol=0,atol=0)
    torch.testing.assert_close(gate,g,rtol=0,atol=0)
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    assert seen and meta['style_dtype']=='float32'
    restore()
    assert module.forward==original
