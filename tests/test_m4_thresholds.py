import pytest
from perceptbudget.m4_thresholds import total_energy,scan,crossings

WAIT={'baseline':[3]*4,'gpu_emc':[1]*4}
LOW=dict(profile='gpu_emc',mean_service_seconds=.6,mean_absolute_service_energy_j=5,p95_service_seconds=.65)
HIGH=dict(profile='baseline',mean_service_seconds=.4,mean_absolute_service_energy_j=4,p95_service_seconds=.45)
DOWN=dict(seconds=.25,energy_j=.5,max_readback_seconds=.02)


def test_common_window_counts_recovery_once_and_return_block_once():
    assert total_energy(1,LOW,WAIT,'gpu_emc',DOWN)==pytest.approx(5.4)
    assert total_energy(1,HIGH,WAIT,'gpu_emc',DOWN)==pytest.approx(4.85)
    assert total_energy(.5,HIGH,WAIT,'gpu_emc',DOWN) is None


def test_service_deadline_different_from_energy_and_return_state_deadline():
    r=scan(LOW,HIGH,WAIT,DOWN,maximum=2)
    assert r['high_ready_seconds']==pytest.approx(.545)
    assert r['low_ready_seconds']==pytest.approx(.745)
    assert r['high_return_code_seconds']==pytest.approx(.795)
    assert r['high_return_readback_estimate_seconds']==pytest.approx(.565)
    assert next(x for x in r['rows'] if x['window_seconds']==.6)['lowest']=='high_full_high'
    assert not r['equal_low_end_state_crossings_seconds']
    assert r['min_high_return_minus_low_j']<0 # No crossing does not imply low wins.


def test_crossing_search_does_not_assume_only_one_threshold():
    assert crossings([(0,1),(1,-1),(2,1)])==[.5,1.5]
