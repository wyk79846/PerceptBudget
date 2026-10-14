import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import eval_q


SUITES = ["libero_spatial", "libero_object", "libero_goal"]
Q_VALUES = [5, 8, 10, 12, 15]


class FakeSuite:
    def __init__(self, name, *, duplicate_names=False):
        self.name = name
        self.n_tasks = 10
        self.duplicate_names = duplicate_names
        self.init_state_calls = []

    def get_task(self, local_id):
        task_name = "{}-task-{}".format(self.name, local_id)
        if self.duplicate_names and self.name in SUITES[:2] and local_id == 0:
            task_name = "shared-task-name"
        return SimpleNamespace(name=task_name, language="instruction for " + task_name)

    def get_task_init_states(self, local_id):
        self.init_state_calls.append(local_id)
        return ["init:{}:{}:0".format(self.name, local_id), "unused-state"]

    def get_task_bddl_file_path(self, local_id):
        return "/fake/libero/{}/task-{}.bddl".format(self.name, local_id)


class FakeBenchmark:
    def __init__(self, *, duplicate_names=False):
        self.duplicate_names = duplicate_names
        self.instances = {}

    def get_benchmark_dict(self):
        def constructor(name):
            def make_suite(task_order_index=0):
                if task_order_index != 0:
                    raise AssertionError("expected task_order_index=0")
                suite = FakeSuite(name, duplicate_names=self.duplicate_names)
                self.instances[name] = suite
                return suite

            return make_suite

        return {name: constructor(name) for name in SUITES}


@contextmanager
def mock_libero_api(*, duplicate_names=False):
    benchmark = FakeBenchmark(duplicate_names=duplicate_names)
    with patch.object(eval_q, "_libero_api", return_value=(benchmark, object())):
        with patch.object(Path, "is_file", return_value=True):
            yield benchmark


def formal_config():
    return {
        "model_id": "pi05_libero_pytorch",
        "H": 20,
        "N": 10,
        "precision": "bfloat16",
        "state_enabled": True,
        "candidate_q": list(Q_VALUES),
        "screen": {
            "suites": list(SUITES),
            "task_ids": list(range(30)),
            "trials_per_task": 1,
        },
    }


class TaskSelectionTests(unittest.TestCase):
    def test_global_ids_map_to_suite_and_local_ids_and_keep_init_state_zero(self):
        selected_ids = [0, 9, 10, 19, 20, 29]
        expected = [
            (0, "libero_spatial", 0),
            (9, "libero_spatial", 9),
            (10, "libero_object", 0),
            (19, "libero_object", 9),
            (20, "libero_goal", 0),
            (29, "libero_goal", 9),
        ]

        with mock_libero_api() as benchmark:
            tasks = eval_q._load_tasks(selected_ids, SUITES)

        self.assertEqual(len(tasks), len(expected))
        for task, (global_id, suite_name, local_id) in zip(tasks, expected):
            self.assertEqual(task["task_id"], global_id)
            self.assertEqual(task["suite"], suite_name)
            self.assertEqual(task["suite_task_id"], local_id)
            self.assertEqual(task["init_state"], "init:{}:{}:0".format(suite_name, local_id))
            self.assertEqual(task["bddl_path"], Path(
                "/fake/libero/{}/task-{}.bddl".format(suite_name, local_id)
            ))
        for suite_name in SUITES:
            suite = benchmark.instances[suite_name]
            self.assertEqual(
                suite.init_state_calls,
                [local_id for _, name, local_id in expected if name == suite_name],
            )

    def test_global_id_outside_selected_suites_is_rejected(self):
        with mock_libero_api():
            for task_id in (-1, 30):
                with self.subTest(task_id=task_id):
                    with self.assertRaises(ValueError):
                        eval_q._load_tasks([task_id], SUITES)

    def test_duplicate_task_names_are_rejected_across_suites(self):
        with mock_libero_api(duplicate_names=True):
            with self.assertRaisesRegex(ValueError, "任务名称重复"):
                eval_q._load_tasks([0, 10], SUITES)

    def test_formal_plan_has_150_episodes_and_smoke_stays_at_task_zero_q5(self):
        config = formal_config()

        task_ids, q_values, mode = eval_q._plan(config, smoke=False)
        self.assertEqual(task_ids, list(range(30)))
        self.assertEqual(q_values, Q_VALUES)
        self.assertEqual(mode, "formal_screen")
        self.assertEqual(len(task_ids) * len(q_values), 150)

        self.assertEqual(eval_q._plan(config, smoke=True), ([0], [5], "smoke"))

    def test_historical_nine_spatial_task_plan_remains_valid(self):
        config = formal_config()
        config["screen"]["suites"] = ["libero_spatial"]
        config["screen"]["task_ids"] = list(range(9))

        self.assertEqual(
            eval_q._plan(config, smoke=False),
            (list(range(9)), Q_VALUES, "formal_screen"),
        )

    def test_formal_plan_rejects_duplicate_task_ids(self):
        config = formal_config()
        config["screen"]["task_ids"][11] = 10

        with self.assertRaises(ValueError):
            eval_q._plan(config, smoke=False)


if __name__ == "__main__":
    unittest.main()
