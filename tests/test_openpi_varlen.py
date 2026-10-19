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
