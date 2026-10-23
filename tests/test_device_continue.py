from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import pytest

from perceptbudget.device_continue import control_window, ContinueLease


def test_window_executes_only_first_five_unique_committed_indices():
    actions = np.arange(50*7, dtype=np.float32).reshape(50, 7)
    executed = []
    def step(action):
        executed.append(action.copy())
        return {'step': len(executed)}, 0., False, {}
    env = SimpleNamespace(step=step, check_success=lambda: False)
    lease = SimpleNamespace(ensure=lambda *args: None)
    ticks = []
    with patch('perceptbudget.device_continue.time.sleep'):
        result, frames = control_window(env, {'step': 0}, actions, lease,
            lambda e, o: o.copy(), lambda o: o['step'], ticks.append, np=np)
    assert len(executed) == result['executed_steps'] == 5
    np.testing.assert_array_equal(executed, actions[:5])
    assert [r['source_action_index'] for r in ticks] == [0, 1, 2, 3, 4]
    assert not any(r['inference_active'] for r in ticks)
    assert frames == [0, 1, 2, 3, 4, 5]
    assert ticks[-1]['queue_after_step_actions'] == 0


def test_terminal_stops_without_padding_or_restarting_chunk():
    env = SimpleNamespace(step=lambda action: ({}, 0., True, {}), check_success=lambda: False)
    result, _ = control_window(env, {}, np.zeros((50, 7)), SimpleNamespace(ensure=lambda *a: None),
        lambda *a: {}, lambda o: 0, lambda r: None, np=np)
    assert result['executed_steps'] == 1


def test_history_then_window_consumes_unique_indices_without_restarting():
    actions=np.arange(50*7,dtype=np.float32).reshape(50,7)
    used=[];trace=[]
    env=SimpleNamespace(step=lambda a:(used.append(a.copy()) or {},0.,False,{}),check_success=lambda:False)
    lease=SimpleNamespace(ensure=lambda *a:None)
    with patch('perceptbudget.device_continue.time.sleep'):
        history,_=control_window(env,{},actions,lease,lambda *a:{},lambda o:None,trace.append,
            np=np,capture_frames=False,steps=15)
        measured,_=control_window(env,{},actions[15:],lease,lambda *a:{},lambda o:None,trace.append,
            np=np,capture_frames=False,source_offset=15)
    np.testing.assert_array_equal(used,actions[:20])
    assert [r['source_action_index'] for r in trace]==list(range(20))
    assert measured['rows'][0]['source_action_index']==15 and history['executed_steps']==15
    with pytest.raises(ValueError,match='H50'):
        control_window(env,{},actions,lease,None,None,None,np=np,source_offset=48)


def test_continue_history_plan_has_no_new_inference_and_bounded_indices(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv',['probe','--continue-history','--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as launch:
        assert main()==0
    result=json.loads(capsys.readouterr().out)
    assert result['windows']==45 and result['model_loads']==1
    assert result['history_actions']==[0,3,15] and result['measured_actions']==5
    assert result['max_control_steps']==495 and not result['hardware_sleep_claim']
    launch.assert_not_called()


def test_history_resume_rejects_unchecked_or_duplicate_windows(tmp_path):
    from perceptbudget.device_continue import load_history_resume
    root=tmp_path/'M2';p=root/'run';p.mkdir(parents=True)
    cfg={'output_root':str(tmp_path),'H':50,'N':5}
    (p/'run.json').write_text(json.dumps({'config_snapshot':cfg}))
    w=dict(profile='baseline',requested_history_seconds=0.,repeat=0,restored=True,
        frequency_check={'matches':True},refresh_frequency_check={'matches':True},
        committed_prefix_max_abs_error=0,executed_steps=5,deadline_overruns=0,
        rows=[{'source_action_index':i,'inference_active':False} for i in range(5)])
    for field in ('transient_continue','including_fresh_images'):
        w[field]={'energy_j':1.,'power_coverage':{'coverage_fraction':1.,'missing_count':0}}
    record={'scope':'No-inference Continue history','windows':[w]}
    def save(): (p/'continue_control.json').write_text(json.dumps(record))
    save();assert len(load_history_resume(p,cfg)['windows'])==1
    record['windows'].append(w);save()
    with pytest.raises(ValueError,match='duplicate'):load_history_resume(p,cfg)
    record['windows']=[w];w['restored']=False;save()
    with pytest.raises(ValueError,match='Unchecked'):load_history_resume(p,cfg)
    w['restored']=True;record['preflight_only']=True;save()
    with pytest.raises(ValueError,match='preflight'):load_history_resume(p,cfg)


def test_short_or_nonfinite_prefix_rejected_before_environment_step():
    env = SimpleNamespace(step=lambda a: pytest.fail('Should not execute'))
    for actions in (np.zeros((4, 7)), np.full((5, 7), np.nan)):
        with pytest.raises(ValueError):
            control_window(env, {}, actions, None, None, None, None, np=np)


def test_continue_lease_is_short_and_has_dense_actual_clock_samples(tmp_path):
    lease = ContinueLease('balanced', tmp_path, 'one')
    command = lease.command()
    assert command[command.index('--seconds')+1] == '2'
    assert command[-2:] == ['--trace-interval', '.02']
    with pytest.raises(ValueError):
        ContinueLease('cpu1190', tmp_path, 'bad')


def test_continue_dry_run_has_one_model_no_long_open_loop_claim(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--continue-control', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as run:
        assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['model_loads'] == 1 and result['max_control_steps'] == 75
    assert result['inference_during_continue'] is False
    assert result['steady_hold_claim'] is False
    run.assert_not_called()


def test_camera_comparison_plan_is_resident_and_paired(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--continue-camera-compare', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as run:
        assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['model_loads'] == 1 and result['windows'] == 30
    assert result['max_control_steps'] == 150 and result['step_period_seconds'] == .07
    assert result['fresh_images_after_both'] and not result['production_baseline_changed']
    run.assert_not_called()


def test_control_without_frames_does_not_access_camera_observation():
    env = SimpleNamespace(step=lambda a: ({'state': 1}, 0., False, {}), check_success=lambda: False)
    with patch('perceptbudget.device_continue.time.sleep'):
        result, frames = control_window(env, {'state': 0}, np.zeros((50, 7)),
            SimpleNamespace(ensure=lambda *a: None), lambda e,o: o.copy(),
            lambda o: pytest.fail('No image access allowed'), lambda row: None,
            np=np, period=.07, capture_frames=False)
    assert result['executed_steps'] == 5 and frames == []
