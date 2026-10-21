"""Independent CPU pooling reference must detect spatial/view layout errors."""
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('factor',[2,4])
@pytest.mark.parametrize('pooled_view',['both','agent','wrist'])
def test_pool_check_validates_both_camera_grids(factor,pooled_view):
    import torch
    from scripts.diagnose_pool_parity import pool_check
    original=((torch.arange(2*16*16*1152)%255).float()/64-2).reshape(2,16,16,1152)
    original=original.to(torch.bfloat16)
    expected=original.float().reshape(2,16//factor,factor,16//factor,factor,1152).mean((2,4))
    expected=expected.to(torch.bfloat16)
    if pooled_view=='agent':
        expected=torch.cat((expected[0].reshape(-1,1152),original[1].reshape(-1,1152)))
    elif pooled_view=='wrist':
        expected=torch.cat((original[0].reshape(-1,1152),expected[1].reshape(-1,1152)))
    class Buffer:
        def __init__(self,value):
            self.bits=value.contiguous().view(torch.uint16).numpy().reshape(-1)
        def download(self,destination):
            np.copyto(destination,self.bits)
    count=512//factor**2 if pooled_view=='both' else 256+256//factor**2
    pipeline=SimpleNamespace(vision_pool_factor=factor,vision_seq_enc=count,bufs={
        'vision_x':Buffer(original),'vision_x_pooled':Buffer(expected)})
    result=pool_check(pipeline,torch,np,pooled_view)
    assert result['max_abs']==0 and result['mismatched_elements']==0
    pipeline.bufs['vision_x_pooled']=Buffer(expected.flip(0))
    assert pool_check(pipeline,torch,np,pooled_view)['mismatched_elements']>0
