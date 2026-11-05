import pytest

from scripts.diagnose_fast_fixed_input import recorded_frame_index


def test_request_frame_uses_saved_control_step_mapping():
    summary={'episode':{'continue_capture':{'frame_sampling':'request observations only',
        'frame_steps':[0,33,47,81]}}}
    assert recorded_frame_index(summary,33)==1
    with pytest.raises(ValueError):recorded_frame_index(summary,34)


def test_original_per_step_recording_keeps_direct_index():
    assert recorded_frame_index({'episode':{}},33)==33
