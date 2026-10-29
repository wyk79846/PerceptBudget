import pytest
from perceptbudget.m4_visual import M4ViewRTCAdapter


def test_mode_is_explicit_and_shared_by_planning_and_prediction():
    adapter=object.__new__(M4ViewRTCAdapter)
    adapter.config={'m2_visual_fast_mode':'pool2_agent','m2_fast_no_state':True}
    assert adapter.context_mode('',0,guided=False)=='full'
    with pytest.raises(RuntimeError,match='explicit mode'):
        adapter.context_mode('',5,guided=True)
    adapter.select_next_mode('Fast')
    assert adapter.context_mode('',5,guided=True)=='pool2_agent'
    adapter.select_next_mode('Full')
    assert adapter.context_mode('',5,guided=True)=='full'
    with pytest.raises(ValueError,match='does not submit'):
        adapter.select_next_mode('Continue')
