from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import pytest

from perceptbudget.device_service import fresh_observation, prepare_submission, measure_one


def test_fresh_observation_forces_real_sensors_without_stepping():
    state = np.zeros(3)
    seen = []
    env = SimpleNamespace(env=SimpleNamespace(
        sim=SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: state.copy())),
        _get_observations=lambda **kw: seen.append(kw) or {'fresh': True}))
    obs, record = fresh_observation(env)
    assert obs['fresh'] and seen == [{'force_update': True}]
    assert record['physics_advanced'] is False
    def mutate(**kw):
        state[0] += 1
        return {}
    env.env._get_observations = mutate
    with pytest.raises(RuntimeError, match='changed physics'):
        fresh_observation(env)


def test_submission_crops_elapsed_actions_without_extending_original_horizon():
    actions = np.arange(350, dtype=np.float32).reshape(50, 7)
    raw = np.zeros((50, 32), dtype=np.float32)
    old = {13: (np.zeros(7), 0, 0, 13)}
    packet = prepare_submission(actions, raw, old, observation_step=10, consumer_step=13)
    assert packet['expired_offsets'] == 3
    assert packet['new_interval_steps'] == [13, 60]
    assert packet['target_steps'] == list(range(13, 60))
    np.testing.assert_array_equal(packet['action_payload'], actions[3:])
    assert old[13][1] == 0  # preparing the packet did not mutate source queue
    packet = prepare_submission(actions, raw, {}, 10, 70)
    assert packet['target_steps'] == [] and packet['new_interval_steps'] == [60, 60]


@pytest.mark.parametrize('state_enabled',[True,False])
def test_measurement_acquires_and_preprocesses_after_device_readiness(state_enabled):
    events = []
    observation = {'agentview_image': np.zeros((2, 2, 3), dtype=np.uint8),
                   'robot0_eye_in_hand_image': np.zeros((2, 2, 3), dtype=np.uint8)}
    acquisition = {'start_monotonic': 0, 'stop_monotonic': 1}
    def fresh(env):
        events.append('acquire')
        return observation, acquisition
    def preprocess(obs, stats, enabled):
        assert enabled is state_enabled
        events.append('preprocess')
        return {'raw_state': np.zeros(8)}
    def predict(snapshot, prompt, previous, **kw):
        events.append('predict')
        kw['request']['actions_raw'] = np.zeros((50, 32)).tolist()
        kw['request']['rtc_guidance']={'state_in_prompt':state_enabled}
        return np.zeros((50, 7))
    with patch('perceptbudget.device_service.fresh_observation', side_effect=fresh), \
         patch('scripts.eval_q._make_snapshot', side_effect=preprocess):
        row = measure_one(SimpleNamespace(predict=predict,pipeline=SimpleNamespace(vision_seq_enc=320,num_views=2,vision_pool_factor=2)), object(), 'prompt', {},
                          np.zeros((45, 32)), {}, torch=None, np=np,state_enabled=state_enabled)
    assert events == ['acquire', 'preprocess', 'predict']
    assert row['submit_start_monotonic'] >= row['inference_stop_monotonic']
    assert row['ready_monotonic'] >= row['submit_start_monotonic']
    assert row['packet']['new_interval_steps'] == [5, 55]


def test_full_preparation_peer_shares_weights_without_mutating_captured_fast_context():
    from perceptbudget.device_service import full_preparation_peer
    weights=object(); keys=object()
    expert=SimpleNamespace(weights=weights,keys=keys,values=keys,prefix_mask=keys)
    adapter=SimpleNamespace(expert=object(),step_bank=SimpleNamespace(fallback=expert),graph=object())
    model=SimpleNamespace(_pipe=SimpleNamespace(pipeline=object()))
    peer=full_preparation_peer(adapter,model)
    assert peer.model is model and peer.expert.weights is weights
    assert peer.expert is not expert and peer.expert.keys is None
    assert expert.keys is keys and adapter.step_bank is not None
    assert peer.step_bank is None and peer.graph is None


def test_selected_fast_service_plan_and_mode_validation(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv',['probe','--selected-fast-clocks','--live-service','--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['services']==15 and len(plan['profiles'])==3 and plan['model_loads']==1
    assert plan['state'] is False and plan['full_preparation_state'] is True
    assert plan['visual_tokens']==320 and plan['live_observation']
    assert plan['consumer_steps_during_service']==0 and not plan['extra_idle_scan']
    launch.assert_not_called()
    for args in (['--live-service'],['--selected-fast-clocks','--live-service','--idle-recovery']):
        with patch('sys.argv',['probe',*args,'--dry-run']),pytest.raises(SystemExit):
            main()


def test_full_service_dry_run_is_small_and_not_an_idle_scan(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--full-service', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as launch:
        assert main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['services'] == 5 and plan['model_loads'] == 1
    assert plan['live_observation'] and plan['runtime_commit_reused']
    assert plan['consumer_steps_during_service'] == 0
    assert plan['extra_idle_scan'] is False
    launch.assert_not_called()


def test_multi_profile_service_plan(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv', ['probe', '--full-service', '--service-profiles',
                           'baseline', 'balanced', 'gpu_emc', '--dry-run']), \
         patch('scripts.profile_clock_screen.subprocess.run') as launch:
        assert main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['services'] == 15 and plan['model_loads'] == 1
    assert plan['profiles'] == ['baseline', 'balanced', 'gpu_emc']
    launch.assert_not_called()


def test_service_profiles_reject_duplicates_and_wrong_mode():
    from perceptbudget.device_service import validate_profiles
    from scripts.profile_clock_screen import main
    for profiles in ([], ['baseline', 'baseline'], ['unknown']):
        with pytest.raises(ValueError):
            validate_profiles(profiles)
    with patch('sys.argv', ['probe', '--service-profiles', 'baseline', '--dry-run']):
        with pytest.raises(SystemExit):
            main()
