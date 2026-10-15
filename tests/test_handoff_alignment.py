import unittest

from scripts.probe_delay_buffer import buffer_count, commit_aligned, refresh_due


class HandoffAlignmentTests(unittest.TestCase):
    def test_overdue_request_uses_first_free_current_frame(self):
        self.assertTrue(refresh_due(12, 5, 5, 'latest_due'))
        self.assertFalse(refresh_due(12, 5, 5, 'boundary'))
        self.assertFalse(refresh_due(16, 12, 5, 'latest_due'))
        self.assertTrue(refresh_due(17, 12, 5, 'latest_due'))
        self.assertFalse(refresh_due(0, 0, 5, 'latest_due'))

    def actions(self):
        return [[1., 0., 0., 0., 0., 0., -1.] for _ in range(20)]

    def test_measured_progress_is_not_elapsed_control_steps(self):
        queue = {}
        expired, overwritten, info = commit_aligned(
            queue, self.actions(), 0, 7, 1, [0, 0, 0], [.03, 0, 0], [], 'progress')
        self.assertEqual(expired, 3)
        self.assertEqual(overwritten, 0)
        self.assertEqual(info['selected_prediction_offset'], 3)
        self.assertEqual(info['discarded_tail_actions'], 4)
        self.assertEqual(queue[7][3], 3)
        self.assertEqual(buffer_count(queue, 7), 13)
        self.assertEqual(max(queue), 19)  # Do not extend the original H20 horizon.

    def test_unmatched_state_falls_back_to_fresh_near_term_command(self):
        _, _, info = commit_aligned({}, self.actions(), 0, 7, 1,
                                   [0, 0, 0], [1, 1, 1], [], 'progress')
        self.assertEqual(info['selected_prediction_offset'], 0)
        self.assertGreater(info['best_waypoint_distance_m'], .02)

    def test_head_keeps_first_fresh_actions_but_trims_equal_tail_budget(self):
        queue = {}
        _, _, info = commit_aligned(queue, self.actions(), 8, 15, 2,
                                   [0, 0, 0], [.03, 0, 0], [], 'head')
        self.assertEqual(info['selected_prediction_offset'], 0)
        self.assertEqual(info['remaining_actions'], 13)
        self.assertEqual(info['discarded_tail_actions'], 7)
        self.assertEqual(min(queue), 15)
        self.assertEqual(max(queue), 27)

    def test_unperformed_gripper_change_is_not_skipped(self):
        actions = self.actions()
        for action in actions[3:]:
            action[6] = 1
        history = [{'step': i, 'action': [1, 0, 0, 0, 0, 0, -1],
                    'before': {'raw_state': [0, 0, 0]},
                    'after': {'raw_state': [0, 0, 0]}} for i in range(7)]
        queue = {}
        _, _, info = commit_aligned(queue, actions, 0, 7, 1,
                                   [0, 0, 0], [.07, 0, 0], history, 'progress')
        self.assertEqual(info['selected_prediction_offset'], 3)
        self.assertEqual(queue[7][0][6], 1)

    def test_expired_chunk_cannot_create_new_buffer(self):
        queue = {}
        _, _, info = commit_aligned(queue, self.actions(), 0, 21, 1,
                                   [0, 0, 0], [0, 0, 0], [], 'head')
        self.assertEqual(info['remaining_actions'], 0)
        self.assertEqual(queue, {})

    def test_latest_executed_gripper_change_counts_as_performed(self):
        actions = self.actions()
        for action in actions[3:]:
            action[6] = 1
        history = [{'step': i, 'action': [1, 0, 0, 0, 0, 0, 1 if i == 6 else -1],
                    'before': {'raw_state': [0, 0, 0]},
                    'after': {'raw_state': [0, 0, 0]}} for i in range(7)]
        _, _, info = commit_aligned({}, actions, 0, 7, 1,
                                   [0, 0, 0], [.07, 0, 0], history, 'progress')
        self.assertEqual(info['selected_prediction_offset'], 7)


if __name__ == '__main__':
    unittest.main()
