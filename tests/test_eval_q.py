import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts import eval_q


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def advance(self, seconds):
        self.now += seconds


class FakeFuture:
    def __init__(self, function, args, kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._done = False
        self._value = None
        self._error = None

    def run(self):
        if self._done:
            return
        try:
            self._value = self.function(*self.args, **self.kwargs)
        except BaseException as exc:
            self._error = exc
        self._done = True

    def done(self):
        return self._done

    def result(self):
        if not self._done:
            raise RuntimeError("result requested before completion")
        if self._error is not None:
            raise self._error
        return self._value


class FakeExecutor:
    """Run bootstrap inline and optionally hold later predictions until released."""

    def __init__(self, defer_after_first=False):
        self.defer_after_first = defer_after_first
        self.submissions = 0
        self.pending = None

    def submit(self, function, *args, **kwargs):
        self.submissions += 1
        future = FakeFuture(function, args, kwargs)
        if self.defer_after_first and self.submissions > 1:
            self.pending = future
        else:
            future.run()
        return future

    def complete_pending(self):
        self.pending.run()

    def shutdown(self, wait=True):
        pass


class FakeEnv:
    def __init__(self, observation, success_after, on_step=None):
        self.observation = observation
        self.success_after = success_after
        self.on_step = on_step
        self.formal_steps = 0
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.formal_steps += 1
        if self.on_step is not None:
            self.on_step(self.formal_steps)
        return self.observation, 0.0, False, {}

    def check_success(self):
        return self.formal_steps >= self.success_after


class FakeModel:
    def __init__(self, clock, *, duration=0.0, error=None, on_predict=None):
        self.clock = clock
        self.duration = duration
        self.error = error
        self.on_predict = on_predict
        self.calls = []
        self.actions = np.zeros((5, 7), dtype=np.float32)
        self.actions[:, 6] = 0.6

    def predict(self, images, **kwargs):
        self.calls.append((images, kwargs))
        if self.on_predict is not None:
            self.on_predict()
        self.clock.advance(self.duration)
        if self.error is not None:
            raise self.error
        return self.actions


class EvalQEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.observation = {
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        self.norm_stats = {
            "state": {"q01": [-1.0] * 8, "q99": [1.0] * 8},
        }
        self.task = {"task_id": 2, "instruction": "fake task"}
        self.power_report = SimpleNamespace(
            energy_mj=12.5,
            coverage_fraction=0.75,
            missing_count=1,
            total_energy_mj=None,
        )
        self.monitor_instances = []
        clock = self.clock
        test_case = self

        class FakePowerMonitor:
            def __init__(self, poll_interval):
                self.readings = []
                self.started = False
                self.stopped = False
                test_case.monitor_instances.append(self)

            @staticmethod
            def mark():
                return clock.monotonic()

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

            def integrate_report(self, start, stop):
                return test_case.power_report

        self.fake_power_monitor = FakePowerMonitor
        self.fake_torch = SimpleNamespace(
            manual_seed=Mock(), cuda=SimpleNamespace(manual_seed_all=Mock())
        )

    def _run_episode(self, env, model, executor, *, max_steps, episode=None):
        if episode is None:
            episode = {"task_id": self.task["task_id"], "task_name": "fake"}
        trace = io.StringIO()
        with patch.multiple(
            eval_q,
            _prepare_env_episode=Mock(return_value=(self.observation, 20.0)),
            PowerMonitor=self.fake_power_monitor,
            ThreadPoolExecutor=Mock(return_value=executor),
            MAX_STEPS=max_steps,
        ), patch.object(eval_q, "time", self.clock):
            try:
                result = eval_q._run_episode(
                    env, self.task, 5, model, self.norm_stats, True, 5, np,
                    self.fake_torch, trace, episode,
                )
            finally:
                self.last_trace_records = [
                    json.loads(line) for line in trace.getvalue().splitlines()
                ]
        return result, self.last_trace_records

    def test_bootstrap_is_timed_without_advancing_steps_and_first_success_counts(self):
        env = FakeEnv(self.observation, success_after=1)
        model = FakeModel(
            self.clock, duration=0.025,
            on_predict=lambda: self.assertEqual(env.formal_steps, 0),
        )

        episode, records = self._run_episode(
            env, model, FakeExecutor(), max_steps=1
        )

        self.assertEqual(episode["status"], "success")
        self.assertEqual(episode["control_steps"], 1)
        self.assertEqual(episode["simulated_control_seconds"], 0.05)
        self.assertGreaterEqual(episode["real_task_seconds"], 0.025)
        self.assertEqual(episode["inference"]["requests"][0]["observation_step"], 0)
        ready = next(row for row in records if row["event"] == "bootstrap_ready")
        first_step = next(row for row in records if row["event"] == "control_step")
        self.assertFalse(ready["control_steps_started"])
        self.assertEqual(first_step["control_step"], 0)

        images, model_args = model.calls[0]
        self.assertEqual([image.shape for image in images], [(224, 224, 3)] * 2)
        self.assertEqual(model_args["state"].shape, (8,))
        np.testing.assert_array_equal(images[0], np.zeros((224, 224, 3), dtype=np.uint8))
        self.assertEqual(episode["power"]["coverage_fraction"], 0.75)
        self.assertEqual(episode["power"]["covered_energy_mj"], 12.5)
        self.assertIsNone(episode["power"]["total_energy_mj"])

    def test_wait_hold_step_advances_control_count_and_preserves_gripper(self):
        executor = FakeExecutor(defer_after_first=True)
        env = FakeEnv(
            self.observation, success_after=6,
            on_step=lambda count: executor.complete_pending() if count == 6 else None,
        )
        model = FakeModel(self.clock)

        episode, _ = self._run_episode(env, model, executor, max_steps=6)

        self.assertEqual(episode["status"], "success")
        self.assertEqual(episode["control_steps"], 6)
        self.assertEqual(episode["simulated_control_seconds"], 0.3)
        self.assertEqual(episode["inference"]["underflow_steps"], [5])
        np.testing.assert_array_equal(
            env.actions[5], np.asarray([0, 0, 0, 0, 0, 0, 0.6], dtype=np.float32)
        )
        self.assertTrue(executor.pending.done())

    def test_bootstrap_prediction_error_marks_invalid_without_success(self):
        env = FakeEnv(self.observation, success_after=1)
        model = FakeModel(self.clock, error=RuntimeError("bootstrap failed"))
        episode = {"task_id": self.task["task_id"], "task_name": "fake"}

        with self.assertRaisesRegex(RuntimeError, "bootstrap failed"):
            self._run_episode(
                env, model, FakeExecutor(), max_steps=3, episode=episode
            )

        records = self.last_trace_records
        self.assertEqual(episode["status"], "invalid")
        self.assertEqual(episode["control_steps"], 0)
        self.assertEqual(episode["inference"]["failed_requests"], 1)
        self.assertEqual(env.formal_steps, 0)
        self.assertTrue(any(row["event"] == "episode_invalid" for row in records))
        self.assertFalse(any(row["event"] == "control_step" for row in records))


if __name__ == "__main__":
    unittest.main()
