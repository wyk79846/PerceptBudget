import numpy as np
from scripts.audit_pooled_path import selected_requests,compare_arrays
import pytest


def test_selects_early_middle_late_pooling_not_full_requests():
    rows=[dict(prediction_id=i,context_refresh={'mode':'full' if i%2==0 else 'pool2'}) for i in range(10)]
    assert [r['prediction_id'] for r in selected_requests(rows)]==[1,5,9]
    with pytest.raises(ValueError):
        selected_requests([])


def test_reports_exact_differences_without_loose_pass_threshold():
    x=np.zeros((2,3),dtype=np.float32)
    assert compare_arrays(x,x.copy())['exact']
    y=x.copy();y[0,0]=1e-8
    result=compare_arrays(x,y)
    assert not result['exact'] and result['different_elements']==1
    assert not compare_arrays(x,x[:1])['shape_matches']
