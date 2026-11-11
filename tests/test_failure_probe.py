import math

from scripts.train_failure_probe import auc, choose_threshold, metrics


def test_failure_auc_handles_ordering_and_tied_risks():
    labels=[0,0,1,1]
    assert auc(labels,[.1,.2,.8,.9])==1.
    assert auc(labels,[.9,.8,.2,.1])==0.
    assert auc(labels,[.5,.5,.5,.5])==.5
    assert auc([1,1],[.1,.2]) is None


def test_calibration_threshold_and_constant_risk_are_finite():
    labels=[0,0,1,1]
    scores=[.1,.2,.8,.9]
    threshold=choose_threshold(labels,scores)
    assert metrics(labels,scores,threshold)['balanced_accuracy']==1.
    threshold=choose_threshold(labels,[.5]*4)
    result=metrics(labels,[.5]*4,threshold)
    assert math.isfinite(threshold)
    assert result['balanced_accuracy']==.5
    assert result['fp']==0 and result['fn']==2
