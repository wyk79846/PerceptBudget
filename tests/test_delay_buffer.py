import unittest

from scripts.probe_delay_buffer import buffer_count, commit, reference_prefix_count


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
