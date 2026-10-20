import numpy as np
from perceptbudget.h50_checkpoint import normalize_state,decode_actions


def test_state_padded_after_normalizing_and_not_clipped():
    stats={'state':{'q01':[0]*8,'q99':[2]*8}}
    result=normalize_state(np.array([3]*8,dtype=np.float32),stats)
    assert result.shape==(32,)
    np.testing.assert_array_equal(result[:8],np.full(8,2))
    np.testing.assert_array_equal(result[8:],np.zeros(24))


def test_new_actions_are_not_clipped_and_use_published_denominator():
    stats={'actions':{'q01':[-1]*7,'q99':[1]*7}}
    raw=np.full((20,32),2,dtype=np.float32)
    np.testing.assert_array_equal(decode_actions(raw,stats),np.full((20,7),2))


def test_independent_audit_new_decoder_agrees():
    from scripts.audit_rtc_system import decode_raw
    stats={'actions':{'q01':[-1]*7,'q99':[1]*7}}
    raw=np.arange(640,dtype=np.float32).reshape(20,32)/100
    np.testing.assert_array_equal(decode_actions(raw,stats),decode_raw(raw,stats,'lerobot_h50_quantiles_v044'))
