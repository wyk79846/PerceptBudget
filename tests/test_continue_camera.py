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
def test_q30_capture_is_enabled_by_default_and_old_behavior_can_be_reproduced(monkeypatch,capsys,capture_flags,enabled):
    import json
    import sys
    from scripts import run_m4
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','q30',
        '--task-indices','0','--continue-overhead-pair','--dry-run',*capture_flags])
    assert run_m4.main() == 0
    plan=json.loads(capsys.readouterr().out)
    assert plan['lean_continue'] is enabled
    assert plan['tasks']==[['libero_spatial',0]] and plan['episodes']==2
    assert plan['compute_profiles']==['baseline'] and plan['Full'] and plan['Fast']=='disabled'
    assert not plan['low_full'] and not plan['three_frequency']
