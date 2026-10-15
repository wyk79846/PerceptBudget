import unittest

from scripts.analyze_delay_trace import target_contact, describe


class DelayTraceAnalysisTest(unittest.TestCase):
    def test_contact_requires_target_finger_and_nonpositive_distance(self):
        def scene(a, b, distance):
            return {'contacts': [{'geom1': a, 'geom2': b, 'distance': distance}]}
        self.assertTrue(target_contact(scene('gripper0_finger1_collision', 'akita_black_bowl_1_g0', -0.001)))
        self.assertTrue(target_contact(scene('akita_black_bowl_1_g0', 'gripper0_finger1_collision', 0)))
        self.assertFalse(target_contact(scene('gripper0_finger1_collision', 'akita_black_bowl_1_g0', 0.001)))
        self.assertFalse(target_contact(scene('table_collision', 'akita_black_bowl_1_g0', -0.001)))
        self.assertFalse(target_contact(scene(None, 'akita_black_bowl_1_g0', -0.001)))

    def test_describe(self):
        self.assertEqual(describe([1, 3]), {'count': 2, 'mean': 2, 'median': 2.0, 'max': 3})
        self.assertEqual(describe([]), {'count': 0, 'mean': None, 'median': None, 'max': None})
