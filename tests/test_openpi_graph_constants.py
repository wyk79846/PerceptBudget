def test_fixed_mask_constructor_is_local_and_exact():
    import torch
    from perceptbudget.openpi_reference import _MaskConstantTorch
    constructor = torch.tensor
    constant = torch.tensor([1.,0.,0.])
    proxy = _MaskConstantTorch(torch, constant)
    assert proxy.tensor([1,0,0],dtype=torch.float32,device=torch.device('cpu')) is constant
    assert proxy.tensor([0,0,0],dtype=torch.float32,device='cpu') is not constant
    assert proxy.tensor([1,0,0],dtype=torch.bfloat16,device='cpu').dtype == torch.bfloat16
    assert torch.tensor is constructor


def test_fixed_context_refresh_preserves_addresses_and_clears_padding():
    import torch
    from perceptbudget.openpi_reference import OriginalOpenPiExpert
    expert = OriginalOpenPiExpert.__new__(OriginalOpenPiExpert)
    expert.context_capacity = 9
    expert.keys = expert.values = None
    k,v = torch.randn(18,7,1,256),torch.randn(18,7,1,256)
    expert.refresh_context(k,v,7)
    pointers = tuple(t.data_ptr() for t in (expert.keys,expert.values,expert.prefix_mask))
    expert.refresh_context(k,v*.9,5)
    assert pointers == tuple(t.data_ptr() for t in (expert.keys,expert.values,expert.prefix_mask))
    assert expert.prefix_mask.tolist() == [[True]*5+[False]*4]
    assert torch.count_nonzero(expert.keys[...,5:,:]) == 0
    assert torch.count_nonzero(expert.values[...,5:,:]) == 0


def test_graph_rejects_replaced_context_before_gpu_replay():
    import pytest
    import torch
    from types import SimpleNamespace
    from perceptbudget.openpi_graph import CapturedOriginalRtc
    expert = SimpleNamespace(horizon=20,keys=torch.zeros(1),values=torch.zeros(1),prefix_mask=torch.ones(1))
    graph = CapturedOriginalRtc.__new__(CapturedOriginalRtc)
    graph.expert = expert
    graph.binding = tuple(t.data_ptr() for t in (expert.keys,expert.values,expert.prefix_mask))
    expert.keys = expert.keys.clone()
    with pytest.raises(RuntimeError,match='refusing stale KV'):
        graph.replay(torch.zeros(20,32),torch.zeros(15,32),5,5)


def test_exact_length_refresh_changes_shape_without_padding():
    import torch
    from perceptbudget.openpi_reference import OriginalOpenPiExpert
    expert = OriginalOpenPiExpert.__new__(OriginalOpenPiExpert)
    expert.context_capacity = None
    expert.keys = expert.values = None
    keys, values = torch.randn(18,9,1,256), torch.randn(18,9,1,256)
    for valid in (9,5,8,9):
        expert.refresh_context(keys,values,valid)
        assert expert.keys.shape[-2] == valid
        assert expert.prefix_mask.shape == (1,valid)
        assert expert.prefix_mask.all()


def test_changed_length_fallback_keeps_captured_context_and_reuses_weights():
    from types import SimpleNamespace
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    adapter = OriginalOpenPiRTCAdapter.__new__(OriginalOpenPiRTCAdapter)
    suffix,keys,values = object(),object(),object()
    cached = SimpleNamespace(suffix=suffix,keys=keys,values=values)
    adapter.graph = object()
    adapter.graph_prefix_length = 570
    adapter.graph_expert = adapter.expert = cached
    adapter.fallback_expert = None
    assert not adapter._select_expert(569)
    assert adapter.expert is not cached
    assert adapter.expert.suffix is suffix
    assert adapter.expert.keys is adapter.expert.values is None
    assert (cached.keys,cached.values) == (keys,values)
    fallback = adapter.expert
    assert not adapter._select_expert(563)
    assert adapter.expert is fallback
    assert adapter._select_expert(570)
    assert adapter.expert is cached


def test_prepared_adapter_refuses_runtime_recapture():
    import pytest
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    adapter=OriginalOpenPiRTCAdapter.__new__(OriginalOpenPiRTCAdapter)
    adapter.graph=object()
    with pytest.raises(RuntimeError,match='runtime recapture is forbidden'):
        adapter.prepare_graph(None)
