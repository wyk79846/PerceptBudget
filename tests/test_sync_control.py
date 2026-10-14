import threading
import unittest
from concurrent.futures import Future

from perceptbudget.control import SynchronousActionController


class FakeFuture:
    """CPU-only Future whose result() blocks until the test completes it."""

    def __init__(self):
        self._completed = threading.Event()
        self.result_called = threading.Event()
        self._value = None
        self._error = None

    def done(self):
        return self._completed.is_set()

    def result(self):
        self.result_called.set()
        self._completed.wait()
        if self._error is not None:
            raise self._error
        return self._value

    def set_result(self, value):
        self._value = value
        self._completed.set()

    def set_exception(self, error):
        self._error = error
        self._completed.set()


class SynchronousActionControllerTests(unittest.TestCase):
    def _pop_while_completing(self, controller, future, step, actions, old_action):
        outcome = {}

        def pop():
            try:
                outcome["action"] = controller.pop()
            except BaseException as error:
                outcome["error"] = error

        worker = threading.Thread(target=pop, daemon=True)
        worker.start()
        try:
            self.assertTrue(future.result_called.wait(2), "pop() did not wait on result()")
            self.assertEqual(controller.control_step, step)
            self.assertEqual(controller.action, old_action)
        finally:
            future.set_result(actions)
            worker.join(2)

        self.assertFalse(worker.is_alive(), "pop() did not finish after Future completion")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["action"]

    def test_q5_refresh_restarts_each_chunk_at_zero_across_30_tasks(self):
        for task_id in range(30):
            with self.subTest(task_id=task_id):
                calls = []

                def submit(snapshot, observation_step):
                    future = FakeFuture()
                    calls.append((snapshot, observation_step, future))
                    return future

                controller = SynchronousActionController(
                    submit, q=5, horizon=20, control_period_s=0.05
                )

                for step in range(30):
                    snapshot = "task-{}-obs-{}".format(task_id, step)
                    if step == 0:
                        self.assertTrue(controller.bootstrap(snapshot))
                    else:
                        submitted = controller.request(snapshot, observation_step=step)
                        self.assertEqual(submitted, step % 5 == 0)

                    chunk_id, offset = divmod(step, 5)
                    if offset == 0:
                        self.assertEqual(calls[-1][:2], (snapshot, step))
                        future = calls[-1][2]
                        chunk = [
                            "task-{}-chunk-{}-offset-{}".format(task_id, chunk_id, index)
                            for index in range(20)
                        ]
                        old_action = (
                            None
                            if chunk_id == 0
                            else "task-{}-chunk-{}-offset-5".format(task_id, chunk_id - 1)
                        )
                        action = self._pop_while_completing(
                            controller, future, step, chunk, old_action
                        )
                        self.assertEqual(
                            action,
                            "task-{}-chunk-{}-offset-0".format(task_id, chunk_id),
                        )
                    else:
                        self.assertEqual(
                            controller.pop(),
                            "task-{}-chunk-{}-offset-{}".format(
                                task_id, chunk_id, offset
                            ),
                        )

                    self.assertEqual(controller.control_step, step + 1)

                self.assertEqual(
                    [observation_step for _, observation_step, _ in calls],
                    [0, 5, 10, 15, 20, 25],
                )
                self.assertEqual(controller.submitted_requests, 6)
                self.assertEqual(controller.submitted_actions, 6 * 20)
                # Each refresh replaces 15 still-future actions. No actions
                # expire while inference is pending because pop() waits.
                self.assertEqual(controller.dropped_actions, 5 * 15)
                self.assertEqual(controller.expired_actions, 0)
                self.assertEqual(controller.overwritten_actions, 5 * 15)
                self.assertEqual(controller.underflow_steps, [])

    def test_pop_result_error_uses_poll_failure_accounting_for_done_future(self):
        calls = []

        def submit(snapshot, observation_step):
            future = Future()
            calls.append(future)
            return future

        controller = SynchronousActionController(
            submit, q=5, horizon=20, control_period_s=0.05
        )
        controller.bootstrap("obs-0")
        failure = ValueError("inference failed")
        calls[0].set_exception(failure)
        self.assertTrue(calls[0].done())

        with self.assertRaisesRegex(ValueError, "inference failed"):
            controller.pop()

        self.assertFalse(controller.inflight)
        self.assertEqual(controller.failed_requests, 1)
        self.assertEqual(controller.control_step, 0)
        self.assertEqual(controller.underflow_steps, [])


if __name__ == "__main__":
    unittest.main()
