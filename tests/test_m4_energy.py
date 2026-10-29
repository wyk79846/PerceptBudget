import pytest
from perceptbudget.power import PowerReading
from perceptbudget.m4_energy import integrate


def test_absolute_energy_counts_board_baseline_and_rejects_missing_coverage():
    readings=[PowerReading(i/10,10000.) for i in range(11)]
    assert integrate(readings,.1,.9)==pytest.approx(8.)
    with pytest.raises(ValueError,match='coverage'):
        integrate([PowerReading(0,10000.),PowerReading(1,10000.)],0,1)
