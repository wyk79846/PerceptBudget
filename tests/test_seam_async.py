import unittest

import numpy as np

from perceptbudget.seam_rollout import queued_raw_tail
from scripts.probe_delay_buffer import commit


class AsyncSeamAlignmentTest(unittest.TestCase):
    def test_tail_uses_queue_origin_and_commit_consumes_actual_delay(self):
        raw = np.arange(20 * 32, dtype=np.float32).reshape(20, 32)
        queue = {}
        commit(queue, np.zeros((20, 7)), 40, 40, 0, raw)
        for step in range(40, 50):
            queue.pop(step)
        tail, executed, origin = queued_raw_tail(queue, 50, 20)
        self.assertEqual((executed, origin), (10, 40))
        np.testing.assert_array_equal(tail, raw[10:])
        # Prediction observed at 50 completes at 57: new rows 0..6 expired.
        for step in range(50, 57):
            queue.pop(step)
        expired, _ = commit(queue, np.ones((20, 7)), 50, 57, 1, raw)
        self.assertEqual(expired, 7)
        self.assertEqual(queue[57][3], 7)
        self.assertEqual(max(queue), 69)
        tail, executed, origin = queued_raw_tail(queue, 60, 20)
        self.assertEqual((executed, origin), (10, 50))
        np.testing.assert_array_equal(tail, raw[10:])

    def test_missing_queue_has_no_fabricated_prior(self):
        self.assertEqual(queued_raw_tail({}, 20, 20), (None, 0, None))

    def test_rejects_incomplete_or_mixed_origin_tail(self):
        raw = np.zeros((20, 32), dtype=np.float32)
        for broken in ('missing', 'origin', 'raw'):
            queue = {}
            commit(queue, np.zeros((20, 7)), 0, 10, 0, raw)
            if broken == 'missing':
                del queue[15]
            elif broken == 'origin':
                queue[15] = (queue[15][0], 0, 1, 15, raw[15])
            else:
                queue[15] = queue[15][:4]
            with self.assertRaises(ValueError):
                queued_raw_tail(queue, 10, 20)
