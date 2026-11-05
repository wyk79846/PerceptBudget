import pytest


@pytest.mark.parametrize('source_type',['bfloat16','float32'])
def test_mapped_gather_matches_native_bf16_table_and_transfers_only_requested_rows(source_type):
    import torch
    from torch.nn import functional as F
    from perceptbudget.mapped_embedding import MappedCudaEmbedding
    torch.manual_seed(7)
    table=torch.randn(73,32,dtype=getattr(torch,source_type))
    record=dict(lookup_calls=0,lookup_rows=0)
    mapped=MappedCudaEmbedding(table,'cpu',torch.bfloat16,record)
    for ids in [torch.tensor([0,72,4,4]),torch.tensor([[1,2],[3,5]])]:
        expected=F.embedding(ids,table.to(torch.bfloat16))
        actual=F.embedding(ids,mapped)
        assert torch.equal(actual.view(torch.uint16),expected.view(torch.uint16))
        assert actual.device.type=='cpu' and actual.dtype==torch.bfloat16
    assert record==dict(lookup_calls=2,lookup_rows=8)
    assert table.shape==mapped.shape


def test_mapped_embedding_rejects_mutating_embedding_options():
    import torch
    from torch.nn import functional as F
    from perceptbudget.mapped_embedding import MappedCudaEmbedding
    table=torch.ones(8,4,dtype=torch.bfloat16)
    mapped=MappedCudaEmbedding(table,'cpu',torch.bfloat16,dict(lookup_calls=0,lookup_rows=0))
    with pytest.raises(ValueError,match='frozen plain embedding'):
        F.embedding(torch.tensor([1]),mapped,max_norm=1.)
    assert torch.equal(table,torch.ones_like(table))
