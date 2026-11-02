import copy
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from perceptbudget.rtc_sensitivity import variants, configure
from perceptbudget.device_quality import camera_view_for_condition


def test_five_conditions_change_only_initial_history():
    labels=variants();assert len(labels)==len(set(labels))==5
    counts=Counter()
    for label in labels:
        cfg={'H':50,'N':5,'state_enabled':True}
        configure(cfg,label)
        counts[cfg['rtc_sensitivity_condition']]+=1
        assert camera_view_for_condition(label)=='full'
        assert 'rtc_fixed_delay_estimate' not in cfg
        assert cfg['H']==50 and cfg['N']==5 and cfg['state_enabled']
        assert cfg['rtc_initial_delay_steps'] in (8,9,10,11)
        assert cfg['rtc_disable_guidance']==label.startswith('unguided_')
    assert set(counts.values())=={1}


def test_unguided_solver_has_no_backward_and_no_old_action_target():
    import torch
    from perceptbudget.openpi_reference import sample_original_rtc
    calls=[]
    def velocity(x,k):
        calls.append((k,torch.is_grad_enabled()))
        return torch.ones_like(x)*.2
    expert=SimpleNamespace(horizon=20,num_steps=5,velocity=velocity)
    with patch.object(torch.autograd,'grad',side_effect=AssertionError('Unguided must not use VJP')):
        result,norms=sample_original_rtc(expert,torch.zeros(20,32),None,9,5,author_order=True)
    assert calls==[(k,False) for k in range(5)]
    assert norms==[]
    assert torch.allclose(result,torch.full_like(result,-.2))


def test_bootstrap_comparison_accepts_all_five_trials(tmp_path):
    import json
    from perceptbudget.device_quality import audit_paired_bootstraps
    episodes=[]
    request=dict(observation_raw_state=[0.],model_state=[0.],actions_raw=[[0.]],actions=[[0.]],rtc_solver_inputs={'noise':[[1.]]})
    for i,label in enumerate(variants()):
        run=tmp_path/str(i);(run/'rtc').mkdir(parents=True)
        (run/'rtc/q5_requests.json').write_text(json.dumps([request]))
        episodes.append(dict(suite='libero_10',task_id=3,run=str(run),view=label,initial={'initial_hashes':{'state':'same'}}))
    result=audit_paired_bootstraps(episodes)
    assert result['passed'] and len(result['cases'])==4
