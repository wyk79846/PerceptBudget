import numpy as np
import pytest

from perceptbudget.camera_feature_context import changed_camera


def test_camera_change_uses_floating_difference_and_preserves_camera_order():
    old = [np.full((224, 224, 3), 255, np.uint8), np.zeros((224, 224, 3), np.uint8)]
    fresh = [np.zeros_like(old[0]), np.ones_like(old[1])]
    camera, differences = changed_camera(fresh, old)
    assert camera == 0 and differences == [255., 1.]
    camera, differences = changed_camera(fresh[::-1], old[::-1])
    assert camera == 1 and differences == [1., 255.]
    assert changed_camera(old, old)[0] == 0


def test_camera_change_rejects_shape_or_nonfinite_inputs():
    image = np.zeros((224, 224, 3), np.float32)
    with pytest.raises(ValueError, match='matched'):
        changed_camera([image[:10], image], [image, image])
    with pytest.raises(ValueError, match='Nonfinite'):
        changed_camera([image+np.nan, image], [image, image])
