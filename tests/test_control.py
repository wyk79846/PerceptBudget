import unittest

from perceptbudget.control import AsyncActionController, SUPPORTED_Q


class FakeFuture:
    """CPU-only Future stand-in whose result can be completed by a test."""

    def __init__(self):
        self._done = False
        self._value = None
        self._error = None

    def done(self):
        return self._done

    def result(self):
        if not self._done:
            raise RuntimeError("result requested before completion")
        if self._error is not None:
            raise self._error
        return self._value

    def set_result(self, value):
        self._value = value
        self._done = True

    def set_exception(self, error):
        self._error = error
        self._done = True


class AsyncActionControllerTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.futures = []

        def submit(snapshot, observation_step):
            future = FakeFuture()
            self.calls.append((snapshot, observation_step))
            self.futures.append(future)
            return future

        self.controller = AsyncActionController(
            submit, q=5, horizon=20, control_period_s=0.05
        )

    def _complete(self, future, label, count=20):
        future.set_result(["{}{}".format(label, index) for index in range(count)])

    def test_bootstrap_alignment_and_consumes_old_actions_during_inference(self):
        self.assertEqual(SUPPORTED_Q, (5, 8, 10, 12, 15))
        self.controller.bootstrap("obs-0")
        self.assertEqual(self.calls, [("obs-0", 0)])
        self._complete(self.futures[0], "old-")
        self.assertTrue(self.controller.poll())

        self.assertEqual(self.controller.control_step, 0)
        self.assertEqual(self.controller.action, "old-0")
        self.assertEqual([self.controller.pop() for _ in range(5)], [
            "old-0", "old-1", "old-2", "old-3", "old-4"
        ])
        self.assertEqual(self.controller.control_step, 5)

        self.assertTrue(self.controller.request("obs-5"))
        self.assertEqual(self.calls[-1], ("obs-5", 5))
        self.assertEqual([self.controller.pop() for _ in range(3)], [
            "old-5", "old-6", "old-7"
        ])
        self.assertEqual(self.controller.control_step, 8)
        self.assertTrue(self.controller.inflight)

    def test_completed_chunk_trims_expired_prefix_and_never_replays(self):
        self.controller.bootstrap("obs-0")
        self._complete(self.futures[0], "old-")
        self.controller.poll()
        for _ in range(5):
            self.controller.pop()

        self.assertTrue(self.controller.request("obs-5"))
        for _ in range(3):
            self.assertEqual(self.controller.pop(), "old-{}".format(5 + _))
        self._complete(self.futures[1], "new-")
        self.assertTrue(self.controller.poll())

        # Three predicted actions are already past; the newer chunk also
        # supersedes the twelve still-buffered old actions at steps 8..19.
        self.assertEqual(self.controller.dropped_actions, 15)
        self.assertEqual(self.controller.action, "new-3")
        self.assertEqual(self.controller.pop(), "new-3")
        self.assertEqual(self.controller.pop(), "new-4")
        self.assertEqual(self.controller.control_step, 10)
        self.assertEqual(self.controller.action, "new-5")
        self.assertNotIn("old-5", [self.controller.pop(), self.controller.pop()])

    def test_new_chunk_replaces_overlapping_future_actions(self):
        self.controller.bootstrap("obs-0")
        self._complete(self.futures[0], "old-")
        self.controller.poll()
        for _ in range(5):
            self.controller.pop()

        self.assertTrue(self.controller.request("obs-5"))
        self._complete(self.futures[1], "new-")
        self.controller.poll()
        self.assertEqual(self.controller.pop(), "new-0")
        self.assertEqual(self.controller.dropped_actions, 15)
        self.assertEqual(self.controller.submitted_actions, 40)

    def test_q_boundary_is_fixed_and_busy_trigger_is_skipped_not_queued(self):
        self.controller.bootstrap("obs-0")
        for step in range(5):
            self.assertFalse(self.controller.request("obs-{}".format(step)))
            self.assertIsNone(self.controller.pop())
        self.assertFalse(self.controller.request("obs-5"))
        self.assertEqual(self.controller.skipped_requests, 1)
        self.assertEqual(self.controller.skipped_trigger_steps, [5])
        self.assertEqual(self.controller.submitted_requests, 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.controller.pop())

        self._complete(self.futures[0], "late-")
        for step in range(6, 10):
            self.assertFalse(self.controller.request("obs-{}".format(step)))
            self.controller.pop()
        self.assertTrue(self.controller.request("obs-10"))
        self.assertEqual(self.calls[-1], ("obs-10", 10))
        self.assertEqual(self.controller.submitted_requests, 2)

    def test_future_exception_propagates_and_releases_inflight_slot(self):
        self.controller.bootstrap("obs-0")
        failure = ValueError("inference failed")
        self.futures[0].set_exception(failure)
        with self.assertRaisesRegex(ValueError, "inference failed"):
            self.controller.poll()
        self.assertFalse(self.controller.inflight)
        self.assertEqual(self.controller.failed_requests, 1)
        self.assertEqual(self.controller.pop(), None)
        self.assertEqual(self.controller.underflow_steps, [0])

    def test_effective_buffer_statistics_use_fixed_control_period(self):
        self.controller.bootstrap("obs-0")
        self._complete(self.futures[0], "a-", count=4)
        self.controller.poll()
        self.assertEqual(self.controller.buffer, 4)
        self.assertEqual(self.controller.buffer_time_s, 0.2)
        self.assertEqual([self.controller.pop() for _ in range(4)], [
            "a-0", "a-1", "a-2", "a-3"
        ])
        self.assertEqual(self.controller.buffer_samples, (
            (0, 4), (1, 3), (2, 2), (3, 1)
        ))
        self.assertEqual(self.controller.buffer_summary, {
            "count": 4, "min": 1, "median": 2.5, "max": 4
        })
        self.assertEqual(self.controller.buffer_time_summary_s, {
            "count": 4, "min": 0.05, "median": 0.125, "max": 0.2
        })

    def test_rejects_observation_step_mismatch(self):
        self.controller.bootstrap("obs-0")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.controller.request("stale-obs", observation_step=1)


if __name__ == "__main__":
    unittest.main()
