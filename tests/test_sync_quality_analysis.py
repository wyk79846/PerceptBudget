import unittest

from scripts.analyze_sync_quality import path_length, resample, switches


class QualityAnalysisTest(unittest.TestCase):
    def test_path_and_sampling(self):
        points = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]]
        self.assertEqual(path_length(points), 2)
        self.assertEqual(resample(points, 3), [[0, 0, 0], [1, 0, 0], [2, 0, 0]])

    def test_stationary(self):
        self.assertEqual(resample([[1, 2, 3], [1, 2, 3]], 3), [[1, 2, 3]] * 3)

    def test_zero_command_does_not_create_switch(self):
        rows = [{'control_step': i, 'executed_action': [0] * 6 + [value]}
                for i, value in enumerate([-1, 0, 1, 1, -1])]
        self.assertEqual([x['step'] for x in switches(rows)], [2, 4])
