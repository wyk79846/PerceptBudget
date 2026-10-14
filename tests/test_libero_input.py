import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from perceptbudget.libero_input import (
    load_libero_norm_stats,
    prepare_libero_observation,
    quantile_normalize_state,
    raw_libero_state,
)


class LiberoInputTests(unittest.TestCase):
    def setUp(self):
        self.observation = {
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.asarray(
                [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float64
            ),
            "robot0_gripper_qpos": np.asarray([0.04, -0.04], dtype=np.float32),
            "agentview_image": np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3),
            "robot0_eye_in_hand_image": np.full((4, 4, 3), 17, dtype=np.uint8),
        }
        self.norm_stats = {
            "state": {"q01": [-1.0] * 8, "q99": [1.0] * 8},
            "actions": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
        }

    def test_raw_state_is_eef_position_axis_angle_and_two_gripper_values(self):
        state = raw_libero_state(self.observation)
        np.testing.assert_allclose(
            state,
            [0.1, 0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.04, -0.04],
            atol=1e-6,
        )

    def test_quantile_normalization_uses_checkpoint_formula_without_padding(self):
        state = raw_libero_state(self.observation)
        q01 = np.asarray([-1.0] * 8, dtype=np.float32)
        q99 = np.asarray([1.0] * 8, dtype=np.float32)
        expected = (state - q01) / (
            q99 - q01 + np.float32(1e-6)
        ) * 2.0 - 1.0
        np.testing.assert_allclose(
            quantile_normalize_state(state, self.norm_stats), expected,
            rtol=1e-6, atol=1e-6,
        )
        bad_stats = {"state": {"q01": [0.0] * 7, "q99": [1.0] * 7}}
        with self.assertRaisesRegex(ValueError, "不自动补维"):
            quantile_normalize_state(state, bad_stats)

    def test_prepared_images_are_rotated_and_resized_and_disabled_state_is_omitted(self):
        prepared = prepare_libero_observation(
            self.observation, self.norm_stats, state_enabled=False
        )
        self.assertEqual([image.shape for image in prepared["images"]], [
            (224, 224, 3), (224, 224, 3)
        ])
        self.assertNotIn("state", prepared)
        np.testing.assert_array_equal(
            prepared["images"][0][0, 0], self.observation["agentview_image"][-1, -1]
        )
        self.assertEqual(prepared["raw_state"].shape, (8,))

    def test_non_square_images_keep_orientation_and_receive_symmetric_zero_padding(self):
        base = np.zeros((2, 4, 3), dtype=np.uint8)
        base[0, 0] = [201, 2, 3]
        base[1, 3] = [4, 202, 6]
        observation = dict(self.observation)
        observation["agentview_image"] = base
        observation["robot0_eye_in_hand_image"] = base.copy()

        prepared = prepare_libero_observation(
            observation, self.norm_stats, state_enabled=False
        )
        image = prepared["images"][0]
        self.assertEqual(image.shape, (224, 224, 3))
        np.testing.assert_array_equal(image[:56], 0)
        np.testing.assert_array_equal(image[168:], 0)
        np.testing.assert_array_equal(image[56, 0], base[1, 3])
        np.testing.assert_array_equal(image[56, -1], base[1, 0])

    def test_enabled_state_is_added_as_an_eight_dimensional_model_input(self):
        prepared = prepare_libero_observation(
            self.observation, self.norm_stats, state_enabled=True
        )
        self.assertEqual(prepared["state"].shape, (8,))
        self.assertEqual(prepared["raw_state"].shape, (8,))

    def test_norm_stats_wrapper_is_unwrapped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "norm_stats.json"
            path.write_text(
                json.dumps({"norm_stats": self.norm_stats}), encoding="utf-8"
            )
            self.assertEqual(load_libero_norm_stats(path), self.norm_stats)

    def test_import_does_not_load_torch_or_flashrt(self):
        importlib.import_module("scripts.eval_q")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("flash_rt", sys.modules)
        self.assertNotIn("libero", sys.modules)


if __name__ == "__main__":
    unittest.main()
