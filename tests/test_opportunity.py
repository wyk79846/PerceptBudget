import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_opportunity import analyze_run


def _readings(power_at):
    return [{"timestamp_monotonic": i / 10,
             "power_mw": power_at(i / 10) * 1000, "error": None}
            for i in range(101)]


def _episode(requests, power_at=lambda _: 2.0):
    return {
        "task_id": 0, "task_name": "task", "q": 5, "status": "success",
        "control_freq_hz": 20, "completed_at_monotonic": 10.0,
        "real_task_seconds": 10.0,
        "power": {"raw_readings": _readings(power_at)},
        "inference": {"requests": requests,
                      "buffer_summary_actions": {
                          "count": 4, "min": 2, "median": 4, "max": 6}},
    }


def _request(start, stop):
    return {"inference_started_monotonic": start,
            "inference_ended_monotonic": stop}


class OpportunityTests(unittest.TestCase):
    def _analyze(self, episodes, *, expected=None, status="completed", task_ids=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        run_dir = Path(temp.name)
        summary = {
            "status": status,
            "expected_episodes": len(episodes) if expected is None else expected,
            "task_ids": [0] if task_ids is None else task_ids,
            "q_values": [5],
            "episodes": episodes,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8")
        return run_dir, analyze_run(run_dir)

    def test_clips_and_merges_busy_windows_and_converts_buffer(self):
        _, result = self._analyze([_episode([
            _request(-2, 3), _request(2, 6), _request(8, 12)])])
        episode, q = result["episodes"][0], result["by_q"][0]

        self.assertEqual(episode["busy_intervals"], [[0.0, 6.0], [8.0, 10.0]])
        self.assertAlmostEqual(episode["d"], 0.8)
        self.assertAlmostEqual(episode["busy_power"]["total_energy_j"], 16.0)
        self.assertAlmostEqual(episode["idle_power"]["total_energy_j"], 4.0)
        self.assertEqual(episode["buffer_summary_seconds"], {
            "min": 0.1, "median": 0.2, "max": 0.3})
        self.assertEqual(q["status"], "complete")
        self.assertTrue(result["status"] == "complete")

    def test_missing_power_coverage_marks_analysis_partial(self):
        episode = _episode([_request(2, 6)])
        episode["power"]["raw_readings"] = [
            row for row in episode["power"]["raw_readings"]
            if not 3.0 <= row["timestamp_monotonic"] <= 3.5]
        _, result = self._analyze([episode])

        self.assertEqual(result["source_status"], "completed")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["episodes"][0]["status"], "partial")
        self.assertIsNone(result["episodes"][0]["busy_power"]["total_energy_j"])
        self.assertIsNone(result["by_q"][0]["busy_energy_j"])

    def test_incomplete_expected_q_is_partial_and_keeps_covered_energy(self):
        _, result = self._analyze([_episode([_request(2, 6)])],
                                  expected=2, status="running", task_ids=[0, 1])
        q = result["by_q"][0]

        self.assertEqual(result["status"], "partial")
        self.assertEqual(q["status"], "partial")
        self.assertEqual(q["episodes"], 1)
        self.assertEqual(q["expected_episodes"], 2)
        self.assertIsNone(q["busy_energy_j"])
        self.assertIsNotNone(q["busy_covered_energy_j"])

    def test_empty_inference_has_zero_busy_and_undefined_ratio(self):
        _, result = self._analyze([_episode([])])
        episode = result["episodes"][0]

        self.assertEqual(episode["d"], 0.0)
        self.assertIsNone(episode["P_on_w"])
        self.assertIsNone(episode["r"])
        self.assertIsNone(episode["S"])
        self.assertEqual(episode["idle_seconds"], 10.0)

    def test_negative_opportunity_score_is_not_clamped(self):
        _, result = self._analyze([_episode(
            [_request(0, 5)], lambda t: 1.0 if t <= 5 else 3.0)])
        episode = result["episodes"][0]

        self.assertGreater(episode["r"], 1.0)
        self.assertLess(episode["S"], 0.0)

    def test_missing_request_timestamp_does_not_invent_idle_time(self):
        _, result = self._analyze([_episode([
            _request(2, 6), {"inference_started_monotonic": 7.0}])])
        episode = result["episodes"][0]

        self.assertFalse(episode["inference_timeline_complete"])
        self.assertIsNone(episode["idle_power"])
        self.assertIsNone(episode["d"])
        self.assertIsNone(episode["r"])


if __name__ == "__main__":
    unittest.main()
