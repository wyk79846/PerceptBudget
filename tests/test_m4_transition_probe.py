from unittest.mock import patch
import pytest
from perceptbudget.m4_transition_probe import plan,clock_session,directional_costs


def test_clock_helper_is_closed_on_failure():
    with patch('perceptbudget.m4_transition_probe.RuntimeClocks') as factory:
        with pytest.raises(RuntimeError):
            with clock_session('unused'):
                raise RuntimeError('injected')
        factory.return_value.close.assert_called_once()


def test_directional_windows_are_absolute_not_baseline_subtracted():
    trials=[]
    for i,mode in plan():
        trials.append(dict(mode=mode,windows={phase:dict(energy_j=4 if mode=='switch' else 3,
            frequency_check={'matches':True},readback_seconds=.02) for phase in ('down','up')}))
    costs=directional_costs(trials)
    assert len(trials)==10 and len(costs)==2
    assert all(c['seconds']==.25 and c['energy_j']==4 and c['baseline_window_energy_j']==3 for c in costs)
    assert costs[0]['source']=='baseline' and costs[1]['source']=='gpu_emc'
    trials[0]['windows']['up']['frequency_check']['matches']=False
    with pytest.raises(ValueError):directional_costs(trials)
