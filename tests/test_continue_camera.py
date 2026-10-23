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
