import copy
import unittest

from scripts.run_async_q import configure_handoff, configure_pending_gain, handoff_options


class PendingGainTest(unittest.TestCase):
    def test_default_unchanged_and_explicit_full_gain(self):
        config = {}
        configure_handoff(config)
        self.assertEqual(configure_pending_gain(config), .5)
        configure_pending_gain(config, 1.)
        self.assertEqual(handoff_options(config), dict(condition='latency_limited_progress',
                         alignment_policy='progress', refresh_policy='latest_due', pending_arm_gain=1.))

    def test_resume_does_not_mutate_or_switch_gain(self):
        config = dict(action_handoff='latency_limited_progress', pending_arm_gain=1.)
        before = copy.deepcopy(config)
        configure_pending_gain(config, 1., resume=True)
        configure_pending_gain(config, resume=True)
        self.assertEqual(config, before)
        with self.assertRaises(ValueError):
            configure_pending_gain(config, .5, resume=True)

    def test_invalid_gain_and_other_profile_rejected(self):
        for value in (0., 2., True, float('nan')):
            with self.assertRaises(ValueError):
                configure_pending_gain({'action_handoff': 'latency_limited_progress'}, value)
        with self.assertRaises(ValueError):
            configure_pending_gain({'action_handoff': 'crop_overwrite'}, 1.)
