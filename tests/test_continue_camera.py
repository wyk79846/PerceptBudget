from types import SimpleNamespace
import pytest
from perceptbudget.continue_camera import suspend_cameras


class Observable:
    def __init__(self, modality, enabled=True):
        self.modality, self.enabled = modality, enabled

    def is_enabled(self):
        return self.enabled

    def set_enabled(self, enabled):
        self.enabled = enabled


def test_only_images_disabled_and_flags_restored_on_exception():
    image, inactive, state = Observable('image'), Observable('image', False), Observable('proprio')
    env = SimpleNamespace(env=SimpleNamespace(_observables=dict(a=image,b=inactive,c=state)))
    with pytest.raises(RuntimeError, match='test failure'):
        with suspend_cameras(env):
            assert not image.enabled and not inactive.enabled and state.enabled
            raise RuntimeError('test failure')
    assert image.enabled and not inactive.enabled and state.enabled


def test_missing_camera_rejected():
    env = SimpleNamespace(env=SimpleNamespace(_observables={}))
    with pytest.raises(RuntimeError, match='No image'):
        with suspend_cameras(env):
            pass


def test_continue_skips_images_but_inflight_still_renders_and_restores_flags():
    from perceptbudget.continue_camera import ContinueCapture
    image, state = Observable('image'), Observable('proprio')
    calls = []
    def step(action):
        calls.append((image.enabled, state.enabled, action))
        return ({'state': 4, **({'agentview_image': 9} if image.enabled else {})}, 0., False, {})
    env = SimpleNamespace(env=SimpleNamespace(_observables=dict(image=image,state=state)),step=step)
    capture = ContinueCapture()
    assert capture.step(env, 2, control_step=7, inference_active=False)[0] == {'state': 4}
    assert image.enabled and state.enabled
    assert 'agentview_image' in capture.step(env, 3, control_step=8, inference_active=True)[0]
    assert calls == [(False, True, 2), (True, True, 3)]
    assert capture.summary()['skipped_control_steps'] == [7]


def test_request_after_continue_forces_live_images_without_using_old_frame(monkeypatch):
    from perceptbudget.continue_camera import ContinueCapture
    from perceptbudget import continue_camera
    fresh = dict(state=4, agentview_image='new external', robot0_eye_in_hand_image='new wrist')
    calls = []
    def acquire(env, old):
        calls.append(env)
        return fresh, dict(physics_advanced=False, force_update=True)
    monkeypatch.setattr(continue_camera, 'fresh_camera_observation', acquire)
    capture = ContinueCapture()
    current, detail = capture.for_request('env', dict(state=4), 30)
    assert current is fresh and not detail['physics_advanced']
    capture.for_request('env', fresh, 31)
    assert calls == ['env'] and capture.frame_steps == [30, 31]
    assert capture.acquisitions[0]['step'] == 30


def test_request_only_mode_advances_physics_and_state_while_inference_images_are_suspended(monkeypatch):
    from perceptbudget.continue_camera import ContinueCapture
    from perceptbudget import continue_camera
    image, state = Observable('image'), Observable('proprio')
    position = [0]
    rendered = []
    def step(action):
        position[0] += action
        assert state.is_enabled()
        result = dict(position=position[0])
        if image.is_enabled():
            rendered.append(position[0])
            result.update(agentview_image=position[0],robot0_eye_in_hand_image=-position[0])
        return result, 0., False, {}
    env = SimpleNamespace(env=SimpleNamespace(_observables=dict(image=image,state=state)),step=step)
    capture = ContinueCapture(request_only=True)
    observation = capture.step(env,2,control_step=0,inference_active=True)[0]
    observation = capture.step(env,3,control_step=1,inference_active=False)[0]
    assert observation == {'position':5} and rendered == []
    assert image.is_enabled() and state.is_enabled()
    def acquire(current_env, old):
        assert current_env is env and image.is_enabled()
        return dict(old,agentview_image=position[0],robot0_eye_in_hand_image=-position[0]),dict(physics_advanced=False)
    monkeypatch.setattr(continue_camera,'fresh_camera_observation',acquire)
    fresh, detail = capture.for_request(env,observation,2)
    assert fresh == dict(position=5,agentview_image=5,robot0_eye_in_hand_image=-5)
    assert position == [5] and not detail['physics_advanced']
    assert capture.summary()['skipped_control_steps'] == [0,1]
    assert capture.summary()['skipped_inference_steps'] == [0]
    assert capture.summary()['policy'] == 'request_only_camera_suspend_v2'


def test_request_capture_pair_changes_only_unused_image_sampling():
    from perceptbudget.device_quality import frequency_pair_views, configure_frequency_variant, camera_view_for_condition
    cfg=dict(m4_request_capture_pair=True,m4_request_only_images=True,m4_lean_continue=True,
             m3_three_frequency=True,m3_start_floor_ms=25,m3_idle_min_ms=50,
             m3_low_full=True,H=50,N=5,state_enabled=True)
    assert frequency_pair_views(cfg,0) == ('inflight_images','request_images')
    assert frequency_pair_views(cfg,1) == ('request_images','inflight_images')
    for label in frequency_pair_views(cfg,0):
        child=dict(cfg)
        configure_frequency_variant(child,label)
        assert camera_view_for_condition(label) == 'full'
        assert child['m4_lean_continue']
        assert child['m4_request_only_images'] == (label=='request_images')
        assert all(child[k]==cfg[k] for k in cfg if k!='m4_request_only_images')


@pytest.mark.parametrize('group',['m3','q30'])
def test_request_only_cli_preserves_current_baseline_and_builds_one_load_pair(monkeypatch,capsys,group):
    import json
    import sys
    from scripts import run_m4
    args=['run_m4.py','--acceptance-group',group,'--task-indices','0','--dry-run']
    monkeypatch.setattr(sys,'argv',args)
    assert run_m4.main()==0
    old=json.loads(capsys.readouterr().out)
    assert old['lean_continue'] and not old['request_only_images'] and old['episodes']==1
    monkeypatch.setattr(sys,'argv',[*args,'--request-only-images','--request-capture-pair'])
    assert run_m4.main()==0
    new=json.loads(capsys.readouterr().out)
    assert new['request_only_images'] and new['request_capture_pair']
    assert new['episodes']==2 and new['model_loads']==1 and new['Fast']=='disabled'
    assert new['H']==old['H'] and new['N']==old['N'] and new['compute_profiles']==old['compute_profiles']


def test_request_only_pair_is_forwarded_to_the_real_worker(tmp_path):
    import json
    import sys
    from unittest.mock import patch
    from scripts import run_m4
    cfg=run_m4.load_config()
    from perceptbudget.three_frequency import transition_costs
    transitions=transition_costs(cfg['output_root'])
    cfg['output_root']=str(tmp_path/'results')
    commands=[]
    def stop_before_model(command,log,**kwargs):
        commands.append(command)
        if '--prepare-environments' in command:return 0
        raise RuntimeError('stop before model')
    args=['run_m4.py','--acceptance-group','m3','--task-indices','0',
          '--low-full','--three-frequency','--idle-min-ms','50','--start-floor-ms','25',
          '--request-only-images','--request-capture-pair']
    with patch.object(sys,'argv',args),patch.object(run_m4,'load_config',return_value=cfg), \
         patch.object(run_m4,'validate'),patch('perceptbudget.clock_access.authorize'), \
         patch('perceptbudget.three_frequency.transition_costs',return_value=transitions), \
         patch('perceptbudget.three_frequency.preflight'), \
         patch('scripts.profile_clock_screen.monitored_worker',side_effect=stop_before_model):
        with pytest.raises(RuntimeError,match='stop before model'):
            run_m4.main()
    command=next(c for c in commands if '--worker' in c)
    assert '--request-only-images' in command and '--request-capture-pair' in command
    assert command[command.index('--idle-min-ms')+1]=='50'
    run=next((tmp_path/'results'/'M4').iterdir())
    saved=json.loads((run/'run.json').read_text())['config_snapshot']
    assert saved['m4_request_only_images'] and saved['m4_request_capture_pair']


def test_camera_refresh_does_not_resample_or_replace_state():
    import numpy as np
    from perceptbudget.continue_camera import fresh_camera_observation
    calls = []
    image, state = Observable('image'), Observable('proprio')
    image.update = lambda **kw: calls.append(('image',kw['force']))
    state.update = lambda **kw: calls.append(('state',kw['force']))
    inner = SimpleNamespace(_observables=dict(image=image,state=state),model_timestep=.002,_obs_cache={},
        sim=SimpleNamespace(get_state=lambda: np.array([1.,2.])))
    inner._get_observations = lambda **kw: dict(joints='other state',agentview_image=3,robot0_eye_in_hand_image=4)
    obs, details = fresh_camera_observation(SimpleNamespace(env=inner),dict(joints='original state'))
    assert obs == dict(joints='original state',agentview_image=3,robot0_eye_in_hand_image=4)
    assert calls == [('image',True)]
    assert not details['proprioception_force_updated'] and not details['physics_advanced']


def test_capture_pair_keeps_clock_state_and_model_configuration_identical():
    from perceptbudget.device_quality import frequency_pair_views, configure_frequency_variant, camera_view_for_condition
    cfg = dict(m4_continue_overhead_pair=True,m3_three_frequency=True,m3_start_floor_ms=25,
               m3_low_full=True,H=50,N=5,state_enabled=True)
    assert frequency_pair_views(cfg,0) == ('full_capture','lean_continue')
    assert frequency_pair_views(cfg,1) == ('lean_continue','full_capture')
    for label in frequency_pair_views(cfg,0):
        child = dict(cfg)
        configure_frequency_variant(child,label)
        assert camera_view_for_condition(label) == 'full'
        assert child['m4_lean_continue'] == (label == 'lean_continue')
        assert all(child[k] == cfg[k] for k in cfg)


def test_pair_cli_has_two_episodes_one_model(monkeypatch,capsys):
    import json
    import sys
    from scripts import run_m4
    common=['run_m4.py','--acceptance-group','m3','--low-full','--three-frequency',
            '--start-floor-ms','25','--continue-overhead-pair','--task-indices','0','--dry-run']
    monkeypatch.setattr(sys,'argv',common)
    assert run_m4.main() == 0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes'] == 2 and plan['model_loads'] == 1
    assert plan['continue_overhead_pair'] and plan['Fast'] == 'disabled'


@pytest.mark.parametrize('capture_flags,enabled',[([],True),(['--lean-continue'],True),(['--full-capture'],False)])
@pytest.mark.parametrize('group',['m3','q30'])
def test_capture_is_enabled_by_default_and_old_behavior_can_be_reproduced(monkeypatch,capsys,capture_flags,enabled,group):
    import json
    import sys
    from scripts import run_m4
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group',group,
        '--task-indices','0','--continue-overhead-pair','--dry-run',*capture_flags])
    assert run_m4.main() == 0
    plan=json.loads(capsys.readouterr().out)
    assert plan['lean_continue'] is enabled
    assert plan['tasks']==[['libero_spatial',0]] and plan['episodes']==2
    assert plan['compute_profiles']==['baseline'] and plan['Full'] and plan['Fast']=='disabled'
    assert not plan['low_full'] and not plan['three_frequency']


@pytest.mark.parametrize('capture_flag',['--lean-continue','--full-capture'])
def test_capture_option_can_be_forwarded_to_frequency_pair_worker(monkeypatch,capsys,capture_flag):
    import json
    import sys
    from scripts import run_m4
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','m3','--low-full',
        '--three-frequency','--start-floor-ms','25','--full-frequency-pair','--task-indices','0',
        capture_flag,'--dry-run'])
    assert run_m4.main() == 0
    plan=json.loads(capsys.readouterr().out)
    assert plan['lean_continue'] == (capture_flag=='--lean-continue')
    assert plan['full_frequency_pair'] and plan['episodes']==2 and plan['model_loads']==1
