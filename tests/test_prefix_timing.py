import math
import unittest

from perceptbudget.prefix_timing import PrefixTiming


class PrefixTimingTest(unittest.TestCase):
    def test_latency_and_tick_observations_change_the_choice(self):
        timing = PrefixTiming(0.10)
        initial = timing.choose(q=8, buffer_actions=8)
        self.assertEqual(initial["required_steps"], 3)
        self.assertEqual(initial["selected"], 5)

        timing.observe_latency(0.28)
        slower_inference = timing.choose(q=8, buffer_actions=8)
        self.assertEqual(slower_inference["latency_estimate_s"], 0.28)
        self.assertEqual(slower_inference["required_steps"], 6)
        self.assertEqual(slower_inference["selected"], 6)

        timing.observe_tick(0.05)
        timing.observe_tick(0.08)
        slower_control = timing.choose(q=8, buffer_actions=8)
        self.assertEqual(slower_control["tick_estimate_s"], 0.05)
        self.assertEqual(slower_control["required_steps"], 6)
        self.assertEqual(slower_control["selected"], 6)

    def test_tick_p20_does_not_let_slow_samples_lock_prefix_at_five(self):
        timing = PrefixTiming(0.410)
        for _ in range(12):
            timing.observe_tick(0.085)
        for _ in range(4):
            timing.observe_tick(0.060)

        result = timing.choose(q=8, buffer_actions=8)
        self.assertEqual(result["tick_sample_count"], 16)
        self.assertEqual(result["tick_estimate_s"], 0.060)
        self.assertEqual(result["required_steps"], 7)
        self.assertEqual(result["selected"], 7)

    def test_buffer_shortage_is_reported_as_infeasible(self):
        result = PrefixTiming(0.28).choose(q=8, buffer_actions=5)
        self.assertIsNone(result["selected"])
        self.assertTrue(result["infeasible"])
        self.assertEqual(result["reason"], "no_candidate_fits")
        self.assertEqual(result["required_steps"], 6)

    def test_q_limits_the_largest_eligible_prefix(self):
        timing = PrefixTiming(0.10)
        self.assertEqual(timing.choose(q=5, buffer_actions=8)["selected"], 5)

        result = timing.choose(q=4, buffer_actions=8)
        self.assertIsNone(result["selected"])
        self.assertTrue(result["infeasible"])

    def test_invalid_samples_are_ignored_and_ticks_are_floored(self):
        timing = PrefixTiming(float("nan"))
        for value in (0, -0.1, float("nan"), float("inf"), None, "0.2"):
            self.assertFalse(timing.observe_latency(value))
            self.assertFalse(timing.observe_tick(value))

        missing = timing.choose(q=8, buffer_actions=8)
        self.assertIsNone(missing["selected"])
        self.assertIsNone(missing["required_steps"])
        self.assertEqual(missing["reason"], "missing_latency_samples")
        self.assertEqual(missing["latency_sample_count"], 0)
        self.assertEqual(missing["tick_sample_count"], 0)

        timing.observe_latency(0.10)
        timing.observe_tick(0.01)
        result = timing.choose(q=8, buffer_actions=8)
        self.assertEqual(result["latency_sample_count"], 1)
        self.assertEqual(result["tick_sample_count"], 1)
        self.assertEqual(result["tick_estimate_s"], 0.05)
        self.assertEqual(result["required_steps"], 3)

    def test_only_the_latest_sixteen_samples_are_kept(self):
        timing = PrefixTiming(10.0)
        for index in range(1, 21):
            timing.observe_latency(index / 100)
        result = timing.choose(q=8, buffer_actions=8)
        self.assertEqual(result["latency_sample_count"], 16)
        # The retained values are .05 through .20; nearest-rank p80 is .17.
        self.assertTrue(math.isclose(result["latency_estimate_s"], 0.17))


if __name__ == "__main__":
    unittest.main()
