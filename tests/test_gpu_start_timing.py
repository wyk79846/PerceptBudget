import pytest
from scripts.probe_gpu_start_timing import first_kernel_offset


def test_first_kernel_uses_device_event_not_cpu_api_or_marker():
    trace={'traceEvents':[
        {'name':'range','cat':'user_annotation','ph':'X','ts':100.,'dur':100.},
        {'name':'range','cat':'gpu_user_annotation','ph':'X','ts':120.,'dur':23.},
        {'name':'cudaGraphLaunch','cat':'cuda_runtime','ts':110.,'dur':2.},
        {'name':'previous','cat':'kernel','ts':90.,'dur':5.},
        {'name':'later','cat':'kernel','ts':140.,'dur':3.},
        {'name':'actual_first','cat':'kernel','ts':120.,'dur':4.}]}
    result=first_kernel_offset(trace,'range')
    assert result['first_kernel_name']=='actual_first'
    assert result['first_kernel_after_marker_us']==20.
    assert result['kernel_count']==2


def test_cpu_only_trace_cannot_pass_as_gpu_start_evidence():
    with pytest.raises(RuntimeError,match='No actual GPU kernel'):
        first_kernel_offset({'traceEvents':[{'name':'range','cat':'user_annotation','ph':'X','ts':100.,'dur':10.}]},'range')


def test_missing_marker_diagnostics_preserve_actual_gpu_evidence_without_claiming_timing():
    from scripts.probe_gpu_start_timing import trace_diagnostics
    trace={'traceEvents':[{'name':'real_kernel','cat':'kernel','ph':'X','ts':20.,'dur':1.}]}
    d=trace_diagnostics(trace,'range')
    assert d['actual_kernel_count']==1 and d['marker_events']==[]
    with pytest.raises(RuntimeError,match='found 0'):
        first_kernel_offset(trace,'range')
