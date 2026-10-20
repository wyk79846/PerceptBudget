def make_expert():
    import torch
    from types import SimpleNamespace, FunctionType, MethodType

    def forward(self):
        pass

    original=object()
    scope={'eager_attention_forward':original}
    method=MethodType(FunctionType(forward.__code__,scope),object())
    model=SimpleNamespace(training=False,layers=[SimpleNamespace(self_attn=SimpleNamespace(forward=method))])
    suffix=SimpleNamespace(parameters=lambda:[],paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(model=model)))
    expert=SimpleNamespace(suffix=suffix,context_capacity=None,prefix_mask=torch.ones(1,4,dtype=torch.bool))
    return expert,scope,original


def test_exact_flash_probe_rejects_padding():
    import pytest
    from perceptbudget.rtc_exact_flash_probe import enable_exact_flash_probe
    expert,_,_=make_expert()
    expert.prefix_mask[0,-1]=False
    with pytest.raises(ValueError,match='all-valid'):
        enable_exact_flash_probe(expert)


def test_exact_flash_probe_checks_mask_and_can_restore(monkeypatch):
    import pytest
    import torch
    from types import SimpleNamespace
    from perceptbudget.rtc_exact_flash_probe import enable_exact_flash_probe
    expert,scope,original=make_expert()
    restore=enable_exact_flash_probe(expert)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    q=torch.zeros(1,2,3,8)
    mask=torch.zeros(1,1,3,7)
    mask[0,0,0,0]=-1000
    try:
        with pytest.raises(ValueError,match='nontrivial'):
            scope['eager_attention_forward'](SimpleNamespace(training=False),q,q,q,mask,1.)
    finally:
        restore()
    assert scope['eager_attention_forward'] is original


def test_exact_flash_probe_preserves_order_and_requests_full_attention(monkeypatch):
    import torch
    from types import SimpleNamespace
    from perceptbudget.rtc_exact_flash_probe import enable_exact_flash_probe
    expert,scope,original=make_expert()
    restore=enable_exact_flash_probe(expert)
    q=torch.zeros(1,2,3,8)
    k=torch.ones(1,1,7,8)
    v=k*2
    seen={}

    def fake(query,key,value,**kwargs):
        seen.update(query=query,key=key,value=value,**kwargs)
        return query

    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(torch.nn.functional,'scaled_dot_product_attention',fake)
    try:
        out,_=scope['eager_attention_forward'](SimpleNamespace(training=False),q,k,v,torch.zeros(1,1,3,7),.25)
    finally:
        restore()
    assert seen['key'] is k and seen['value'] is v
    assert seen['enable_gqa'] and not seen['is_causal'] and seen['dropout_p']==0
    assert out.shape==(1,3,2,8)
