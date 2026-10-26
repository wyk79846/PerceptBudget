from unittest.mock import patch
import json

import numpy as np
import pytest

from perceptbudget.dvac import variance_scores, prefix_decision


def test_variance_compares_same_future_index_across_denoising_steps():
    x=np.zeros((5,4,7))
    x[:,0,0]=[1,1,1,1,1]
    x[:,1,0]=[0,1,2,3,4]
    x[:,2,1]=[0,0,0,0,10]
    v=variance_scores(x)
    assert v.tolist()==pytest.approx([0,2,16,0])
    assert variance_scores(x,tail=3)[1]==pytest.approx(2/3)


def test_variance_does_not_treat_motion_between_future_actions_as_instability():
    x=np.broadcast_to(np.arange(4)[None,:,None],(5,4,7))
    assert np.array_equal(variance_scores(x),np.zeros(4))


def test_prefix_threshold_uses_past_history_and_excludes_consumed_actions():
    history=[np.zeros(5),np.zeros(5)]
    result=prefix_decision(np.array([0,0,0,1,100.]),history,expired=2)
    assert result['threshold']==0 and result['prefix_actions']==3
    assert result['remaining_actions']==1
    assert prefix_decision(np.ones(5),history,expired=3)['remaining_actions']==0
    assert prefix_decision(np.ones(5),[])['warmup']


@pytest.mark.parametrize('shape',[(5,50,32),(1,50,7)])
def test_padded_dimensions_and_insufficient_denoising_steps_rejected(shape):
    with pytest.raises(ValueError):variance_scores(np.zeros(shape))


def test_dvac_dry_run_does_not_load_models_or_launch_processes(capsys):
    from scripts.run_m3 import main
    with patch('sys.argv',['m3','--mode','dvac-probe','--source-run','source','--dry-run']), \
         patch('subprocess.run') as launch:
        assert main()==0
    d=json.loads(capsys.readouterr().out)
    assert d['H']==50 and d['N']==5 and d['model_loads']==1
    assert not d['policy_changed'] and not d['clocks_changed']
    launch.assert_not_called()
