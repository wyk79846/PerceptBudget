import unittest

from scripts.probe_delay_buffer import buffer_count, commit, commit_blended, reference_prefix_count


class DelayBufferTest(unittest.TestCase):
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

    def test_six_dimensional_blend_uses_new_gripper(self):
        queue = {}
        old = [[float(i)] * 7 for i in range(20)]
        new = [[float(100 + i)] * 7 for i in range(20)]
        commit(queue, old, 0, 0, 0)
        for step in range(8):
            queue.pop(step)
        _, _, info = commit_blended(queue, new, 8, 8, 1, blend_dims=6)
        merged = queue[9][0]
        self.assertNotEqual(merged[:6], old[9][:6])
        self.assertEqual(merged[6], new[1][6])
        self.assertEqual(info['blended_dimensions'], 6)
        self.assertEqual(info['gripper_policy'], 'new_prediction')
