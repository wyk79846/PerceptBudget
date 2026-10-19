def test_varlen_requires_initialized_fixed_capacity():
    from types import SimpleNamespace
    import pytest
    from perceptbudget.openpi_varlen import enable_varlen_attention
    with pytest.raises(ValueError,match='fixed-capacity'):
        enable_varlen_attention(SimpleNamespace(context_capacity=None,keys=None))


def test_varlen_does_not_change_installed_transformers_attention():
    import transformers.models.gemma.modeling_gemma as original
    from perceptbudget.openpi_varlen import enable_varlen_attention
    # Merely importing the opt-in function must never patch global modules.
    assert original.eager_attention_forward.__module__ == original.__name__


def test_unknown_kv_order_is_rejected():
    from types import SimpleNamespace
    import pytest
    from perceptbudget.openpi_varlen import enable_varlen_attention
    with pytest.raises(ValueError,match='KV order'):
        enable_varlen_attention(SimpleNamespace(),order='unknown')


def test_varlen_adapter_keeps_same_graph_for_changed_valid_length():
    from types import SimpleNamespace
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    adapter=OriginalOpenPiRTCAdapter.__new__(OriginalOpenPiRTCAdapter)
    adapter.graph=object()
    adapter.graph_prefix_length=592
    adapter.graph_expert=SimpleNamespace(varlen_attention_enabled=True)
    for valid in (570,557,573):
        assert adapter._select_expert(valid)
        assert adapter.expert is adapter.graph_expert
