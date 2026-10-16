import unittest

from perceptbudget.stage_evidence import TASK_OBJECTS, annotate_episode


TARGET, GOAL = TASK_OBJECTS[0]


def scene(x=0, z=1, contact='table_collision', observed=True):
    obs = {TARGET + '_pos': [x, 0, z], GOAL + '_pos': [.1, 0, 1]}
    return dict(observations=obs if observed else {}, raw_state=[x, 0, z+.1],
                contacts=[] if contact is None else [dict(
                    geom1=TARGET+'_g1', geom2=contact, distance=0)])


def row(step, after, before=None):
    return dict(step=step, before=before or scene(), after=after)


class StageEvidenceTests(unittest.TestCase):
    def transport_prefix(self):
        return [row(i, scene(.03+i*.01, 1.04, 'gripper0_finger1')) for i in range(3)]

    def annotate(self, rows, **kwargs):
        return annotate_episode(rows, TARGET, GOAL, **kwargs)

    def test_mapping(self):
        self.assertEqual(TASK_OBJECTS, {0: (TARGET, 'plate_1'), 3: (TARGET, 'plate_1')})
        self.assertEqual(TARGET, 'akita_black_bowl_1')

    def test_table_is_not_grasp(self):
        result = self.annotate([row(i, scene()) for i in range(5)])
        self.assertTrue(all(s['phase'] == 'unknown' for s in result['steps']))
        self.assertFalse(result['steps'][0]['evidence']['finger_target_contact'])

    def test_unknown_and_contact_independent(self):
        s = self.annotate([row(0, scene(contact='gripper0_finger1', observed=False))])['steps'][0]
        self.assertEqual(s['phase'], 'unknown')
        self.assertTrue(s['evidence']['finger_target_contact'])
        self.assertFalse(s['evidence']['object_observed'])

    def test_transport_down_from_cookiebox(self):
        rows = [row(i, scene(.03+i*.01, .98, 'gripper0_finger1'),
                    scene(contact='cookies_1_g1')) for i in range(3)]
        result = self.annotate(rows)
        self.assertEqual([s['phase'] for s in result['steps']],
                         ['unknown', 'unknown', 'transport_candidate'])
        self.assertEqual(result['segments'][-1], dict(
            phase='transport_candidate', start_step=2, end_step=2))

    def test_placement_candidate_requires_support_release_and_window(self):
        for contact, expected in [('plate_1_g2', 'placement_candidate'),
                                  ('table_collision', 'unknown'),
                                  ('gripper0_finger1', 'unknown'), (None, 'unknown')]:
            with self.subTest(contact=contact):
                result = self.annotate(self.transport_prefix() +
                                       [row(i, scene(.1, contact=contact)) for i in range(3, 6)])
                self.assertEqual(result['steps'][-1]['phase'], expected)
                self.assertNotIn('placement', result['steps'][3]['phase'])

    def test_gaps_break_windows_and_segments(self):
        result = self.annotate([row(i, scene(.1, contact='plate_1_g2')) for i in [0, 1, 3]])
        self.assertEqual(len(result['segments']), 2)
        self.assertEqual(result['steps'][-1]['phase'], 'unknown')

    def test_missing_contacts_and_empty(self):
        after = scene(observed=False)
        del after['contacts']
        self.assertIsNone(self.annotate([row(0, after)])['steps'][0]['evidence']['finger_target_contact'])
        self.assertEqual(self.annotate([])['segments'], [])

    def test_validation(self):
        for thresholds in [{'window_steps': 1}, {'motion_m': float('nan')}, {'typo': 1}]:
            with self.assertRaises(ValueError):
                self.annotate([], thresholds=thresholds)
        with self.assertRaises(ValueError):
            self.annotate([row(1, scene()), row(1, scene())])

    def test_other_bowl_and_positive_distance_are_not_contact(self):
        for name, distance in [('akita_black_bowl_10_g1', 0),
                               ('akita_black_bowl_2_g1', 0), (TARGET+'_g1', .001)]:
            after = scene()
            after['contacts'] = [dict(geom1=name, geom2='gripper0_finger1',
                                      distance=distance)]
            evidence = self.annotate([row(0, after)])['steps'][0]['evidence']
            self.assertFalse(evidence['finger_target_contact'])

    def test_transient_contact_and_relative_slip_remain_unknown(self):
        for slip in [False, True]:
            rows = [row(i, scene(.03+i*.01, contact='gripper0_finger1'))
                    for i in range(3)]
            if slip:
                rows[-1]['after']['raw_state'][0] += .02
            else:
                rows[0]['after']['contacts'] = []
            self.assertEqual(self.annotate(rows)['steps'][-1]['phase'], 'unknown')

    def test_descent_on_terminal_frame_after_transport(self):
        before = scene(.1, 1.03, 'gripper0_finger1')
        after = scene(.1, 1.02, 'gripper0_finger1')
        result = self.annotate(self.transport_prefix() + [row(3, after, before)])
        last = result['steps'][-1]
        self.assertEqual(last['phase'], 'placement_approach_candidate')
        self.assertEqual(last['evidence']['placement_approach_reasons'],
                         ['target_descending_toward_goal'])
        self.assertEqual(result['segments'][-1]['start_step'], 3)

    def test_small_adjustments_require_continuous_bounded_motion(self):
        for delta, expected in [(0, False), (.0001, False), (.002, True), (.01, False)]:
            with self.subTest(delta=delta):
                tail = [row(i+3, scene(.09+i*delta, 1.02, 'gripper0_finger1'))
                        for i in range(3)]
                last = self.annotate(self.transport_prefix()+tail)['steps'][-1]
                self.assertEqual(last['phase'] == 'placement_approach_candidate', expected)
                if expected:
                    self.assertIn('continuous_small_adjustments_near_goal',
                                  last['evidence']['placement_approach_reasons'])

    def test_no_transport_never_labels_placement(self):
        for contact in ['plate_1_g2', 'gripper0_finger1']:
            result = self.annotate([row(i, scene(.09+i*.002, contact=contact))
                                    for i in range(3)])
            self.assertTrue(all('placement' not in s['phase'] for s in result['steps']))
        result = self.annotate([row(0, scene(.1, 1.02, 'gripper0_finger1'),
                                    scene(.1, 1.03, 'gripper0_finger1'))])
        self.assertEqual(result['steps'][0]['phase'], 'unknown')

    def test_descent_requires_goal_contact_context_and_held_motion(self):
        for x, observed, contact in [(.3, True, 'gripper0_finger1'),
                                      (.1, False, 'gripper0_finger1'), (.1, True, None)]:
            tail = row(3, scene(x, 1.02, contact, observed),
                       scene(x, 1.03, 'gripper0_finger1'))
            self.assertNotIn('placement',
                             self.annotate(self.transport_prefix()+[tail])['steps'][-1]['phase'])
