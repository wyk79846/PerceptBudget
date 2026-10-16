import unittest

from scripts.probe_delay_buffer import (
    buffer_count, commit, commit_blended, commit_rtc_fused,
    reference_prefix_count,
    commit_prefix_hold, prefix_hold_due,
)


class DelayBufferTest(unittest.TestCase):
    def test_blend_accepts_real_numpy_model_outputs(self):
        import numpy as np
        queue = {}
        actions = np.zeros((20, 7), dtype=np.float32)
        _, _, info = commit_blended(queue, actions, 0, 0, 0)
        self.assertEqual(info['blended_dimensions'], 7)
        _, _, info = commit_blended(queue, actions, 8, 14, 1)
        self.assertEqual(info['global_interval_steps'], [14, 20])
        self.assertEqual(info['weights_old'][0], 1.)
        self.assertEqual(info['weights_old'][-1], 0.)

    def test_hold_starts_only_at_locked_boundary(self):
        request = {'scheduled_commit_step': 12}
        self.assertFalse(prefix_hold_due(request, 11, 'hold'))
        self.assertTrue(prefix_hold_due(request, 12, 'hold'))
        self.assertTrue(prefix_hold_due(request, 13, 'hold'))
        self.assertFalse(prefix_hold_due(request, 13, 'discard'))
        self.assertFalse(prefix_hold_due(None, 13, 'hold'))

    def test_late_hold_resumes_prefix_and_crops_tail_with_raw32(self):
        actions = [[float(i)] * 7 for i in range(20)]
        raw = [[float(i)] * 32 for i in range(20)]
        queue = {i: ('old',) for i in range(14, 40)}
        expired, overwritten, info = commit_prefix_hold(queue, actions, 5, 14, 1, raw, 7)
        self.assertEqual(expired, 7)
        self.assertEqual(overwritten, 11)
        self.assertEqual(info['discarded_tail_actions'], 2)
        self.assertEqual(info['original_deadline_step'], 25)
        self.assertEqual(sorted(queue), list(range(14, 25)))
        self.assertEqual(queue[14], (actions[7], 1, 5, 7, raw[7]))
        self.assertEqual(queue[24][3], 17)
        # The next request can lock seven raw32 rows without inverse normalization.
        self.assertEqual([queue[14+i][4] for i in range(7)], raw[7:14])
        self.assertEqual(buffer_count(queue, 19), 6)  # insufficient: never extend H

    def test_on_time_hold_has_no_tail_crop(self):
        queue = {}
        _, _, info = commit_prefix_hold(queue, list(range(20)), 5, 12, 1,
                                         [[i]*32 for i in range(20)], 7)
        self.assertEqual(queue[12][0], 7)
        self.assertEqual(info['discarded_tail_actions'], 0)
        self.assertEqual(buffer_count(queue, 12), 13)

    def test_fully_expired_hold_errors_without_mutating_queue(self):
        for step in (25, 26):
            queue = {25: ('old',)}
            with self.assertRaisesRegex(RuntimeError, 'fully expired'):
                commit_prefix_hold(queue, list(range(20)), 5, step, 1, [], 7)
            self.assertEqual(queue, {25: ('old',)})

    def test_h20_q8_old_queue_is12(self):
        queue = {}
        self.assertEqual(commit(queue, list(range(20)), 0, 0, 0), (0, 0))
        for step in range(8):
            queue.pop(step)
        self.assertEqual(buffer_count(queue, 8), 12)
        self.assertEqual(reference_prefix_count(queue, 8, 8), 0)
        for step in range(8, 16):
            queue.pop(step)
        self.assertEqual(buffer_count(queue, 16), 4)
        self.assertEqual(commit(queue, list(range(20)), 8, 16, 1), (8, 4))
        self.assertEqual(queue[16], (8, 1, 8, 8))
        self.assertEqual(buffer_count(queue, 16), 12)

    def test_gap_and_reference_prefix(self):
        queue = {0: (0, 0, 0, 0), 2: (2, 0, 0, 2)}
        self.assertEqual(buffer_count(queue, 0), 1)
        self.assertEqual(reference_prefix_count(queue, 0, 8), 1)
        self.assertEqual(buffer_count(queue, 1), 0)

    def test_raw_actions_kept_for_prefix_without_inverse_normalization(self):
        queue = {}
        raw = [[float(i)] * 32 for i in range(20)]
        commit(queue, list(range(20)), 0, 0, 1, raw)
        self.assertEqual(queue[8][4], [8.0] * 32)
        commit(queue, list(range(20)), 8, 16, 2, raw)
        self.assertEqual(queue[16][4], raw[8])

    def test_blend_uses_same_absolute_steps_and_only_future_overlap(self):
        queue = {}
        old = [[float(i)] * 7 for i in range(20)]
        new = [[float(100 + i)] * 7 for i in range(20)]
        commit(queue, old, 0, 0, 0)
        for step in range(13):
            queue.pop(step)
        expired, overwritten, info = commit_blended(queue, new, 8, 13, 1)
        self.assertEqual((expired, overwritten), (5, 7))
        self.assertEqual(info['global_interval_steps'], [13, 20])
        self.assertEqual(info['new_offset_interval'], [5, 12])
        self.assertEqual(queue[13][0], old[13])
        self.assertEqual(queue[19][0], new[11])
        self.assertEqual(queue[20][0], new[12])
        self.assertEqual(buffer_count(queue, 13), 15)
        self.assertNotIn(12, queue)
        self.assertEqual(queue[13][4]['weight_old'], 1)
        self.assertEqual(new[5], [105.0] * 7)

    def test_blend_bootstrap_and_single_overlap(self):
        queue = {}
        new = [[0.1] * 7, [0.2] * 7]
        _, _, info = commit_blended(queue, new, 0, 0, 0)
        self.assertEqual(info['overlap_actions'], 0)
        queue.pop(0)
        _, _, info = commit_blended(queue, [[0.9] * 7], 1, 1, 1)
        self.assertEqual(info['weights_old'], [0.0])
        self.assertEqual(queue[1][0], [0.9] * 7)

    def test_rtc_fusion_uses_temporal_weights_on_arm_and_latest_gripper(self):
        import math
        queue, chunks = {}, []
        old = [[float(i)] * 7 for i in range(20)]
        new = [[float(100 + i)] * 7 for i in range(20)]
        commit_rtc_fused(queue, chunks, old, 0, 0, 0)
        for step in range(8):
            queue.pop(step)
        _, _, info = commit_rtc_fused(queue, chunks, new, 8, 8, 1,
                                     decay=0.1, max_chunks=3)
        old_weight = math.exp(-0.1 * 8)
        expected = (old_weight * 8 + 100) / (old_weight + 1)
        self.assertAlmostEqual(queue[8][0][0], expected)
        self.assertEqual(queue[8][0][6], new[0][6])
        self.assertEqual(queue[8][4]['source_prediction_ids'], [1, 0])
        self.assertEqual(info['arm_dimensions'], 6)
        self.assertEqual(info['gripper_policy'], 'latest_prediction')

    def test_rtc_fusion_accepts_numpy_action_batch(self):
        import numpy as np
        queue, chunks = {}, []
        actions = np.ones((2, 7), dtype=np.float32)
        commit_rtc_fused(queue, chunks, actions, 0, 0, 0)
        self.assertEqual(queue[0][0][6], 1.0)

    def test_rtc_fusion_does_not_reblend_single_source_steps(self):
        queue, chunks = {}, []
        first = [[1.0] * 7 for _ in range(4)]
        second = [[3.0] * 7 for _ in range(4)]
        commit_rtc_fused(queue, chunks, first, 0, 0, 0, max_chunks=3)
        queue.pop(0)
        commit_rtc_fused(queue, chunks, second, 4, 4, 1, max_chunks=3)
        self.assertEqual(queue[4][0], second[0])
        self.assertEqual(queue[4][4]['source_count'], 1)
