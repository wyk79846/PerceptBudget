import math
import unittest

from perceptbudget.action_blend import blend_actions


class ActionBlendTest(unittest.TestCase):
    def test_first_and_last_weights_are_exact(self):
        old = [[float(i)] for i in range(4)]
        new = [[10.0 + i] for i in range(4)]

        merged, weights = blend_actions(old, new)

        self.assertEqual(weights[0], 1.0)
        self.assertEqual(weights[-1], 0.0)
        self.assertEqual(merged[0], old[0])
        self.assertEqual(merged[-1], new[-1])

    def test_intermediate_weights_follow_the_paper_formula(self):
        beta = 1.2
        old = [[0.0] for _ in range(5)]
        new = [[1.0] for _ in range(5)]

        _, weights = blend_actions(old, new, beta)

        expected = [
            (math.exp(-beta * k / 4) - math.exp(-beta))
            / (1.0 - math.exp(-beta))
            for k in range(5)
        ]
        for actual, wanted in zip(weights, expected):
            self.assertTrue(math.isclose(actual, wanted, rel_tol=1e-14, abs_tol=1e-15))

    def test_empty_and_single_step_chunks(self):
        self.assertEqual(blend_actions([], []), ([], []))

        old = [[1.0, 2.0, 3.0]]
        new = [[4.0, 5.0, 0.37]]
        merged, weights = blend_actions(old, new)
        self.assertEqual(merged, new)
        self.assertEqual(weights, [0.0])

    def test_all_seven_dimensions_share_weight_without_gripper_binarization(self):
        old = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2] for _ in range(3)]
        new = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8] for _ in range(3)]

        merged, weights = blend_actions(old, new)

        for index, weight in enumerate(weights):
            expected = [weight * 0.0 + (1.0 - weight) * 1.0] * 6
            expected.append(weight * 0.2 + (1.0 - weight) * 0.8)
            self.assertEqual(len(merged[index]), 7)
            for actual, wanted in zip(merged[index], expected):
                self.assertAlmostEqual(actual, wanted)
        self.assertNotIn(merged[1][-1], (0.0, 1.0))

    def test_inputs_are_not_modified(self):
        old = [[0, 1], [2, 3], [4, 5]]
        new = [[6, 7], [8, 9], [10, 11]]
        old_before = [row[:] for row in old]
        new_before = [row[:] for row in new]

        blend_actions(old, new)

        self.assertEqual(old, old_before)
        self.assertEqual(new, new_before)

    def test_invalid_beta_is_rejected(self):
        for beta in (0, -1, float("nan"), float("inf"), -float("inf"), None, "1.2"):
            with self.subTest(beta=beta):
                with self.assertRaises(ValueError):
                    blend_actions([[0.0]], [[1.0]], beta)

    def test_invalid_chunk_shapes_are_rejected(self):
        invalid_pairs = (
            ([[0.0], [1.0]], [[1.0]]),
            ([[0.0, 1.0]], [[1.0]]),
            ([[0.0], [1.0, 2.0]], [[1.0], [2.0]]),
            ([[0.0], [1.0]], [[1.0], [2.0, 3.0]]),
            ([0.0, 1.0], [[1.0], [2.0]]),
            ([[]], [[]]),
        )
        for old, new in invalid_pairs:
            with self.subTest(old=old, new=new):
                with self.assertRaises(ValueError):
                    blend_actions(old, new)

    def test_nonfinite_and_nonnumeric_values_are_rejected(self):
        invalid_pairs = (
            ([[float("nan")]], [[1.0]]),
            ([[0.0]], [[float("inf")]]),
            ([["0.0"]], [[1.0]]),
        )
        for old, new in invalid_pairs:
            with self.subTest(old=old, new=new):
                with self.assertRaises(ValueError):
                    blend_actions(old, new)


if __name__ == "__main__":
    unittest.main()
