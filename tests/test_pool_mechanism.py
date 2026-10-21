"""Controls isolate content averaging from sequence shortening; no CUDA needed."""
import pytest


@pytest.mark.parametrize('factor',[2,4])
@pytest.mark.parametrize('view',['agent','wrist'])
def test_token_controls_preserve_other_camera_and_original_selection(factor,view):
    import torch
    from scripts.diagnose_pool_mechanism import transform_tokens
    x=(torch.arange(512*1152).reshape(2,16,16,1152)%101).to(torch.bfloat16)
    small=256//factor**2
    for mode in ('full','mean_full_length','mean_compact','select_compact'):
        result=transform_tokens(x,torch,factor,view,mode)
        assert result.dtype==torch.bfloat16
        assert len(result)==(512 if mode in ('full','mean_full_length') else 256+small)
        if mode=='full':
            assert torch.equal(result,x.reshape(512,1152))
        elif view=='agent':
            assert torch.equal(result[-256:],x[1].reshape(256,1152))
        else:
            assert torch.equal(result[:256],x[0].reshape(256,1152))
        if mode=='select_compact':
            original=x[0 if view=='agent' else 1][::factor,::factor].reshape(-1,1152)
            selected=result[:small] if view=='agent' else result[-small:]
            assert torch.equal(selected,original)
    full_mean=transform_tokens(x,torch,factor,view,'mean_full_length')
    compact=transform_tokens(x,torch,factor,view,'mean_compact')
    camera=full_mean[:256] if view=='agent' else full_mean[-256:]
    reduced=camera.reshape(16,16,1152)[::factor,::factor].reshape(-1,1152)
    assert torch.equal(reduced,compact[:small] if view=='agent' else compact[-small:])
