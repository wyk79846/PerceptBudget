import unittest

import numpy as np

from perceptbudget.model import validate_actions


class ActionShapeTests(unittest.TestCase):
    def test_h20_output_is_checked_and_copied(self):
        source = np.zeros((20, 7), dtype=np.float32)
        result = validate_actions(source, 20)
        self.assertEqual(result.shape, (20, 7))
        source[0, 0] = 1
        self.assertEqual(result[0, 0], 0)

    def test_rejects_default_h10_when_h20_requested(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid action output"):
            validate_actions(np.zeros((10, 7)), 20)

    def test_rejects_nan(self):
        source = np.zeros((20, 7))
        source[1, 1] = np.nan
        with self.assertRaises(RuntimeError):
            validate_actions(source, 20)


if __name__ == "__main__":
    unittest.main()
