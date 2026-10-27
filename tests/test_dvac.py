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


def test_revision_aligns_absolute_indices_and_excludes_committed_prefixes():
    from perceptbudget.dvac import aligned_revision
    old=np.zeros((50,32))
    old[:,0]=np.arange(50)
    new=np.zeros_like(old)
    new[:,0]=np.arange(50)+8
    request=dict(observation_step=18,actions_raw=new.tolist(),
        rtc_solver_inputs=dict(delay=5),rtc_guidance=dict(actual_delay_steps=7))
    r=aligned_revision(np.arange(50),old,request,10,9)
    assert r['action_indices']==list(range(15,50))
    assert r['normalized_action_mse']==[0.]*35
    assert r['variance']==list(range(15,50))
    new[:,7:]=9999
    request['actions_raw']=new.tolist()
    assert aligned_revision(np.arange(50),old,request,10,9)['normalized_action_mse']==[0.]*35


def test_rank_correlation_handles_ties_and_constant_scores():
    from perceptbudget.dvac import rank_correlation
    assert rank_correlation([1,1,2,3],[4,4,5,6])==pytest.approx(1.)
    assert rank_correlation([1,1,1],[1,2,3]) is None


def test_tracker_uses_prior_history_and_new_episode_starts_fresh():
    from perceptbudget.dvac import DvacTracker
    x=np.zeros((5,50,7))
    tracker=DvacTracker()
    assert tracker.observe(x)['warmup']
    x[:,12,0]=np.arange(5)
    second=tracker.observe(x)
    assert second['history_count']==1 and second['threshold']==0
    assert second['prefix_actions']==12
    assert DvacTracker().observe(x)['warmup']
    for _ in range(8):tracker.observe(x)
    assert len(tracker.history)==5


def test_paired_dry_run_declares_six_episodes_without_clock_changes(capsys):
    from scripts.run_m3 import main
    with patch('sys.argv',['m3','--mode','dvac-paired','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes']==6 and plan['model_loads']==1
    assert plan['policy_changed'] and not plan['clocks_changed']
    launch.assert_not_called()


def test_horizon_paired_plan_keeps_model_h50_and_changes_only_offset_candidate(capsys):
    from scripts.run_m3 import main
    with patch('sys.argv',['m3','--mode','dvac-horizon-paired','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes']==6 and plan['model_loads']==1
    assert plan['H']==50 and plan['N']==5 and plan['state']
    assert plan['action_offset_limits']==[20,50]
    assert plan['maximum_observation_age_seconds']==2.
    assert not plan['clocks_changed']
    launch.assert_not_called()


def test_three_strategy_plan_declares_nine_episodes_h50_and_one_model(capsys):
    from scripts.run_m3 import main
    from perceptbudget.closed_loop import comparison_variants
    assert comparison_variants({'m3_trigger_comparison':True})==('events_only','variance_only','events_and_variance')
    with patch('sys.argv',['m3','--mode','trigger-comparison','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes']==9 and plan['model_loads']==1
    assert plan['H']==50 and plan['N']==5 and plan['state']
    assert plan['action_offset_limits']==[50] and not plan['clocks_changed']
    launch.assert_not_called()


def test_goal_two_strategy_repeats_are_six_distinct_resident_episodes(capsys):
    from scripts.run_m3 import main
    from perceptbudget.closed_loop import comparison_variants
    labels=comparison_variants(dict(m3_trigger_comparison=True,
        m3_trigger_variants=['variance_only','events_and_variance'],m3_trigger_repeats=3))
    assert len(labels)==len(set(labels))==6
    assert labels[:2]==('variance_only__repeat1','events_and_variance__repeat1')
    with patch('sys.argv',['m3','--mode','trigger-comparison','--trigger-suite','libero_goal',
        '--strategies','variance_only','events_and_variance','--repeats','3','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes']==6 and plan['tasks']==[['libero_goal',0]]
    assert plan['model_loads']==1 and plan['action_offset_limits']==[50]
    launch.assert_not_called()


def test_no_age_limit_dry_run_is_explicit_and_does_not_change_default(capsys):
    from scripts.run_m3 import main
    with patch('sys.argv',['m3','--mode','trigger-comparison','--strategies','variance_only',
                           '--no-age-limit','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['maximum_observation_age_seconds'] is None and plan['action_offset_limits']==[50]
    launch.assert_not_called()


def test_no_age_high_and_dvfs_pair_shares_one_model(capsys):
    from scripts.run_m3 import main
    from perceptbudget.closed_loop import comparison_variants
    assert comparison_variants(dict(m3_trigger_comparison=True,m3_trigger_variants=['variance_only'],
        m3_device_comparison=True))==('variance_only__device_high','variance_only__device_dvfs')
    with patch('sys.argv',['m3','--mode','trigger-comparison','--strategies','variance_only',
        '--trigger-suite','libero_spatial','--compare-device','--no-age-limit','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes']==2 and plan['model_loads']==1 and plan['clocks_changed']
    assert plan['tasks']==[['libero_spatial',0]] and plan['maximum_observation_age_seconds'] is None
    launch.assert_not_called()


def test_long_task0_device_pair_plan_retains_default_three_suites(capsys):
    from scripts.run_m3 import main
    from scripts.eval_q import SUITE_MAX_STEPS
    assert SUITE_MAX_STEPS['libero_10']==520
    with patch('sys.argv',['m3','--mode','trigger-comparison','--strategies','variance_only',
        '--trigger-suite','libero_10','--compare-device','--no-age-limit','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['tasks']==[['libero_10',0]] and plan['episodes']==2 and plan['model_loads']==1
    assert plan['maximum_observation_age_seconds'] is None and plan['H']==50
    launch.assert_not_called()
