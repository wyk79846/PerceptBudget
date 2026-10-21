import unittest

from scripts.analyze_m1_h50_stages import counts, PHASES, TARGET, GOAL
from perceptbudget.stage_evidence import annotate_episode
from scripts.analyze_m1_h50_stages import window_duty


class StageSupplementTests(unittest.TestCase):
    def test_commands_do_not_imply_grasp(self):
        rows = [dict(step=i, action=[0]*6+[1], before={}, after={}) for i in range(5)]
        annotation = annotate_episode(rows, TARGET, GOAL)
        self.assertEqual(counts(annotation), {p: 5 if p == 'unknown' else 0 for p in PHASES})
        self.assertTrue(all(s['evidence']['finger_target_contact'] is None
                            for s in annotation['steps']))

    def test_contact_is_independent_of_stage(self):
        scene = dict(contacts=[dict(geom1=TARGET+'_g1', geom2='gripper0_finger1', distance=0)])
        annotation = annotate_episode([dict(step=0, after=scene)], TARGET, GOAL)
        self.assertEqual(counts(annotation)['unknown'], 1)
        self.assertEqual(counts(annotation)['contact_candidate'], 0)
        self.assertTrue(annotation['steps'][0]['evidence']['finger_target_contact'])

    def test_counts_reject_unrecognized_phase(self):
        with self.assertRaises(ValueError):
            counts(dict(steps=[dict(phase='grasped')]))

    def test_overlapping_inference_union_and_task_clipping(self):
        result = window_duty([(1, 5)], [(-2, 2), (1.5, 3), (4, 8)])
        self.assertEqual(result['T_seconds'], 4)
        self.assertEqual(result['busy_seconds'], 3)
        self.assertEqual(result['d'], .75)
        self.assertEqual(result['idle_gaps_seconds'], [1])

    def test_gaps_never_bridge_other_phase(self):
        result = window_duty([(0, 1), (1, 2), (4, 5)], [])
        self.assertEqual(result['windows'], [[0, 2], [4, 5]])
        self.assertEqual(result['idle_gaps_seconds'], [2, 1])
        self.assertEqual(result['d'], 0)

    def test_empty_phase_is_not_zero_duty(self):
        result = window_duty([], [(0, 10)])
        self.assertIsNone(result['d'])
        self.assertEqual(result['T_seconds'], 0)
        self.assertEqual(result['idle_gaps_seconds'], [])

    def test_tick_window_includes_sync_wait_and_last_to_stop(self):
        # A wait [1,4] between tick 0 and tick 5 belongs to that entire window.
        first = window_duty([(0, 5)], [(1, 4), (6, 10)])
        last = window_duty([(5, 7)], [(1, 4), (6, 10)])
        self.assertEqual(first['busy_seconds'], 3)
        self.assertEqual(first['idle_gaps_seconds'], [1, 1])
        self.assertEqual(last['busy_seconds'], 1)
        self.assertEqual(last['idle_gaps_seconds'], [1])
