import math
import unittest

import numpy as np

from scripts.probe_delay_buffer import commit_blended


class ArmOnlyBlendTest(unittest.TestCase):
    def test_ndarray_bootstrap(self):
        actions = np.arange(140, dtype=np.float32).reshape(20, 7) / 137
        queue = {}
        expired, overwritten, info = commit_blended(queue, actions, 0, 0, 0, arm_only=True)
        self.assertEqual((expired, overwritten), (0, 0))
        self.assertEqual(info['overlap_actions'], 0)
        self.assertEqual(info['blended_dimensions'], 6)
        self.assertEqual(info['gripper_policy'], 'latest_prediction')
        for i in range(20):
            np.testing.assert_array_equal(queue[i][0], actions[i])

    def test_five_step_formula_endpoints_and_all_latest_grippers(self):
        old = np.arange(140, dtype=np.float64).reshape(20, 7) / 151
        new = -np.arange(140, dtype=np.float64).reshape(20, 7) / 157 - 0.123
        queue = {}
        commit_blended(queue, old, 0, 0, 0, arm_only=True)
        for step in range(15):
            queue.pop(step)
        expired, overwritten, info = commit_blended(queue, new, 8, 15, 1, arm_only=True)
        self.assertEqual((expired, overwritten), (7, 5))
        self.assertEqual(info['global_interval_steps'], [15, 20])
        self.assertEqual(info['new_offset_interval'], [7, 12])
        self.assertEqual(sorted(queue), list(range(15, 28)))
        weights = [(math.exp(-1.2 * i / 4) - math.exp(-1.2)) /
                   (1 - math.exp(-1.2)) for i in range(5)]
        for i, weight in enumerate(weights):
            self.assertAlmostEqual(info['weights_old'][i], weight, places=14)
            np.testing.assert_allclose(queue[15 + i][0][:6],
                                       weight * old[15 + i, :6] + (1 - weight) * new[7 + i, :6],
                                       rtol=0, atol=1e-14)
            self.assertEqual(info['merged_actions'][i][6], new[7 + i, 6])
        self.assertEqual(info['weights_old'][0], 1.0)
        self.assertEqual(info['weights_old'][-1], 0.0)
        np.testing.assert_array_equal(queue[15][0][:6], old[15, :6])
        np.testing.assert_array_equal(queue[19][0][:6], new[11, :6])
        for target, item in queue.items():
            self.assertEqual(item[0][6], new[target - 8, 6])
            self.assertEqual(item[1:4], (1, 8, target - 8))

    def test_single_overlap_uses_new_action(self):
        queue = {}
        commit_blended(queue, [[0.1] * 7], 0, 0, 0, arm_only=True)
        latest = [[0.73] * 6 + [-0.23456789]]
        _, _, info = commit_blended(queue, latest, 0, 0, 1, arm_only=True)
        self.assertEqual(info['weights_old'], [0.0])
        self.assertEqual(info['merged_actions'], latest)
        self.assertEqual(queue[0][0], latest[0])

    def test_default_seven_dimensions_still_blend_gripper(self):
        queue = {}
        old = [[0.2] * 7 for _ in range(5)]
        new = [[0.8] * 7 for _ in range(5)]
        commit_blended(queue, old, 0, 0, 0)
        _, _, info = commit_blended(queue, new, 0, 0, 1)
        self.assertEqual(info['blended_dimensions'], 7)
        self.assertEqual(info['gripper_policy'], 'same_blend_as_arm')
        for i, weight in enumerate(info['weights_old']):
            self.assertEqual(queue[i][0], [weight * 0.2 + (1 - weight) * 0.8] * 7)
        self.assertNotEqual(queue[2][0][6], new[2][6])


if __name__ == '__main__':
    unittest.main()
