import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_q import write_report


class SummarizeQTests(unittest.TestCase):
    def _summary(self):
        return {
            "status": "running",
            "expected_episodes": 6,
            "task_ids": [0, 1],
            "q_values": [5, 8, 10],
            "control_freq_hz": 20,
            "episodes": [
                {
                    "task_id": 0,
                    "task_name": "task zero",
                    "q": 5,
                    "status": "success",
                    "control_steps": 40,
                    "simulated_control_seconds": 2.0,
                    "real_task_seconds": 3.0,
                    "power": {"total_energy_mj": 1000, "coverage_fraction": 1.0},
                    "inference": {
                        "underflow_steps": [4],
                        "dropped_actions": 2,
                        "inference_busy_seconds_within_task": 1.0,
                        "inference_duty_fraction": 1 / 3,
                        "buffer_summary_actions": {
                            "count": 40, "min": 1, "median": 3, "max": 8,
                        },
                    },
                },
                {
                    "task_id": 1,
                    "task_name": "task one",
                    "q": 5,
                    "status": "timeout",
                    "control_steps": 220,
                    "simulated_control_seconds": 11.0,
                    "real_task_seconds": 12.0,
                    "power": {"total_energy_mj": 5000, "coverage_fraction": 0.5},
                    "inference": {
                        "underflow_steps": [],
                        "dropped_actions": 1,
                        "inference_busy_seconds_within_task": 2.0,
                        "buffer_summary_actions": {
                            "count": 220, "min": 0, "median": 2, "max": 10,
                        },
                    },
                },
                {
                    "task_id": 0,
                    "task_name": "task zero",
                    "q": 8,
                    "status": "success",
                    "control_steps": 30,
                    "simulated_control_seconds": 1.5,
                    "real_task_seconds": 4.0,
                    "power": {"total_energy_mj": 2000, "coverage_fraction": 1.0},
                    "inference": {
                        "underflow_steps": [],
                        "dropped_actions": 3,
                        "inference_busy_seconds_within_task": 2.0,
                        "inference_duty_fraction": 0.5,
                        "buffer_summary_actions": {
                            "count": 30, "min": 2, "median": 4, "max": 9,
                        },
                    },
                },
                {
                    "task_id": 0,
                    "task_name": "task zero",
                    "q": 10,
                    "status": "failure",
                    "control_steps": 5,
                    "simulated_control_seconds": 0.25,
                    "real_task_seconds": None,
                    "power": {"total_energy_mj": 500, "coverage_fraction": 1.0},
                    "inference": {},
                },
            ],
        }

    def test_partial_aggregation_pairing_units_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            source_path = run_dir / "summary.json"
            source_path.write_text(
                json.dumps(self._summary(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            original_bytes = source_path.read_bytes()

            json_path, markdown_path = write_report(run_dir)

            self.assertEqual(source_path.read_bytes(), original_bytes)
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["observed_episodes"], 4)
            self.assertEqual([row["q"] for row in report["q_summaries"]], [5, 8, 10])

            q5, q8, q10 = report["q_summaries"]
            self.assertEqual((q5["success_count"], q5["timeout_count"]), (1, 1))
            self.assertTrue(q5["q_grid_complete"])
            self.assertEqual(q5["successful_control_time_mean_s"], 2.0)
            self.assertFalse(q5["energy_sum_complete"])
            self.assertEqual(q5["reported_attempt_energy_sum_j"], 1.0)
            self.assertIsNone(q5["energy_per_success_j"])
            self.assertTrue(q8["energy_sum_complete"])
            self.assertFalse(q8["q_grid_complete"])
            self.assertIsNone(q8["energy_per_success_j"])
            self.assertIsNone(q10["energy_per_success_j"])
            self.assertIsNone(q10["all_attempts_real_time_mean_s"])
            self.assertEqual(q10["all_attempts_real_time_sample_count"], 0)
            self.assertEqual(q5["underflow_step_count"], 1)
            self.assertEqual(q5["dropped_actions_total"], 3)

            q8_episode = next(row for row in report["episodes"] if row["q"] == 8)
            self.assertEqual(q8_episode["buffer_summary_seconds"]["median_s"], 0.2)
            timeout = next(row for row in report["episodes"] if row["status"] == "timeout")
            self.assertIsNone(timeout["simulated_control_seconds"])
            self.assertIsNone(timeout["energy_j"])

            comparison = next(
                row for row in report["q5_paired_comparisons"] if row["q"] == 8
            )
            self.assertEqual(comparison["common_success_task_ids"], [0])
            self.assertEqual(comparison["control_time"]["delta_mean_s"], -0.5)
            self.assertEqual(comparison["real_time"]["delta_mean_s"], 1.0)

            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("partial（未完成）", markdown)
            self.assertIn("不合并各 episode 的 median", markdown)
            self.assertIn("P_onidle 尚未独立测量", markdown)


if __name__ == "__main__":
    unittest.main()
