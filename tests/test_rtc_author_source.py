from pathlib import Path
import pytest


def test_source_order_matches_unmodified_author_body_on_nonlinear_coupled_model():
    import torch
    from perceptbudget.openpi_reference import sample_original_rtc
    from perceptbudget.rtc_author_source import sample_author_source
    source=Path('/home/wyk/workspace/任务产出/perceptbudget/rtc-author-reference/src/model.py')
    if not source.exists():
        pytest.skip('local pinned author reference not available')
    class Expert:
        horizon,num_steps=25,5
        def velocity(self,x,k): return .1*x.sin()+.04*x.roll(1,0)+k*.01
    generator=torch.Generator().manual_seed(7)
    noise=torch.randn(25,32,generator=generator)
    previous=torch.randn(17,32,generator=generator)
    trace=[]
    ours,_=sample_original_rtc(Expert(),noise,previous,9,8,trace=trace,author_order=True)
    ref,steps,mask,_=sample_author_source(Expert(),noise,previous,9,8,source=source)
    # CPU autograd may accumulate +/- multiplication contributions in a
    # different order; this remains well below the 1e-4 GPU gate.
    torch.testing.assert_close(ours,ref,atol=1e-6,rtol=0)
    for a,b in zip(trace,steps):
        for name in ('input','velocity','clean','error','vjp','output'):
            torch.testing.assert_close(a[name],b[name],atol=1e-6,rtol=0)
    torch.testing.assert_close(trace[0]['weights'][:,0],mask,atol=0,rtol=0)
