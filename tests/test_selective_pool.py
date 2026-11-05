import torch
import pytest
from perceptbudget.selective_pool import compact


def test_counts_order_and_wrist_exact():
    v=torch.arange(2*16*16*5,dtype=torch.float32).reshape(2,16,16,5)
    for count in [0,8,32,64]:
        result,mask=compact(v,count)
        assert result.shape==(512-3*count,5)
        assert torch.equal(result[-256:],v[1].reshape(256,5))
        if mask is None:assert torch.equal(result,v.reshape(512,5));continue
        assert int(mask.sum())==count
        expected=[]
        for r in range(16):
            for c in range(16):
                chosen=bool(mask[(r//2)*8+c//2])
                if chosen and (r%2 or c%2):continue
                expected.append(v[0,r:r+2,c:c+2].mean((0,1)) if chosen else v[0,r,c])
        assert torch.equal(result[:256-3*count],torch.stack(expected))


def test_redundant_half_selected_and_ties_stable():
    v=torch.ones(2,16,16,4)
    for r in range(8,16,2):
        for c in range(0,16,2):
            v[0,r:r+2,c:c+2]=torch.eye(4).reshape(2,2,4)
    _,mask=compact(v,32)
    assert mask[:32].all() and not mask[32:].any()
    _,tie=compact(torch.ones_like(v),32)
    assert tie[:32].all()


def test_both_cameras_preserve_distinct_details_and_camera_order():
    from perceptbudget.selective_pool import compact_both
    v=torch.ones(2,16,16,4)
    # Each view has a different detailed half; homogeneous backgrounds merge.
    v[0,8:]=torch.eye(4).reshape(2,2,4).repeat(4,8,1)
    v[1,:8]=torch.eye(4).reshape(2,2,4).repeat(4,8,1)
    output,mask=compact_both(v)
    assert output.shape==(320,4) and mask.shape==(2,64)
    assert mask[0,:32].all() and not mask[0,32:].any()
    assert not mask[1,:32].any() and mask[1,32:].all()
    assert torch.equal(output[32:160],v[0,8:].reshape(128,4))
    assert torch.equal(output[160:288],v[1,:8].reshape(128,4))


@pytest.mark.parametrize('merged',[8,32])
def test_both_cameras_native_rounding_reference_and_stable_ties(merged):
    from perceptbudget.selective_pool import compact_both
    v=torch.ones(2,16,16,4,dtype=torch.bfloat16)
    output,mask=compact_both(v,merged)
    assert mask[:,:merged].all() and not mask[:,merged:].any()
    assert torch.equal(output,torch.ones(512-6*merged,4,dtype=torch.bfloat16))
