"""CPU-only checks for commit-anchored, seven-step advance handoffs."""
import unittest

import numpy as np

from scripts import probe_delay_buffer as probe
from tests import test_prefix_hold_episode as helpers


class AdvanceHandoffTest(unittest.TestCase):
    def run_fake(self, delay, max_steps):
        return helpers.PrefixHoldEpisodeTest().run_fake(
            delay, max_steps=max_steps, q=8, advance=True)

    def assert_prefixes_executed(self, actions, requests):
        for request in requests[1:]:
            start, end = request['committed_interval_steps']
            if end <= len(actions):
                np.testing.assert_array_equal(
                    actions[start:end], request['committed_prefix_actions'])
                self.assertEqual(request['scheduled_commit_step'], start + 7)

    def assert_deadlines(self, commits):
        for commit in commits[1:]:
            origin = commit['observation_step']
            step = commit['commit_step']
            self.assertEqual(commit['new_prediction_remaining_interval_steps'],
                             [step, origin + 20])
            self.assertEqual(commit['new_prediction_remaining_actions'],
                             origin + 20 - step)

    def test_due_boundary_and_late_poll(self):
        for commit in (0, 8, 10, 20):
            with self.subTest(commit=commit):
                self.assertFalse(probe.advance_prefix_due(commit, commit, 8, 7))
                self.assertTrue(probe.advance_prefix_due(commit + 1, commit, 8, 7))
                self.assertTrue(probe.advance_prefix_due(commit + 2, commit, 8, 7))
        for q in (0, 6, 7):
            with self.subTest(q=q), self.assertRaises(ValueError):
                probe.advance_prefix_due(1, 0, q, 7)

    def test_seven_step_inference_hands_off_every_eight_steps(self):
        result, actions, requests = self.run_fake(7, 25)
        self.assertEqual([r['observation_step'] for r in requests], [0, 1, 9, 17])
        self.assertEqual([c['commit_step'] for c in result['commits']], [0, 8, 16, 24])
        self.assertEqual(result['late_hold_count'], 0)
        self.assert_prefixes_executed(actions, requests)
        self.assert_deadlines(result['commits'])
        for step in (8, 16, 24):
            np.testing.assert_array_equal(actions[step], np.full(7, 7))
        for request in requests[1:]:
            self.assertEqual(request['refresh_anchor'], 'advance_handoff')
            self.assertEqual(request['target_handoff_step'], request['observation_step'] + 7)
        self.assertTrue(result['advance_handoff'])

    def test_early_ready_preserves_prefix_and_does_not_resubmit(self):
        result, actions, requests = self.run_fake(2, 25)
        self.assertEqual([r['observation_step'] for r in requests], [0, 1, 9, 17])
        self.assertEqual([c['commit_step'] for c in result['commits']], [0, 8, 16, 24])
        self.assertEqual([r['ready_observed_step'] for r in requests[1:]], [3, 11, 19])
        self.assert_prefixes_executed(actions, requests)

    def test_nine_step_inference_holds_and_reanchors_to_actual_commit(self):
        result, actions, requests = self.run_fake(9, 21)
        self.assertEqual([r['observation_step'] for r in requests], [0, 1, 11])
        self.assertEqual([c['commit_step'] for c in result['commits']], [0, 10, 20])
        self.assertEqual(result['late_hold_count'], 2)
        self.assertEqual(result['late_hold_steps'], 4)
        self.assertAlmostEqual(result['late_hold_wall_seconds'], .2)
        for start in (8, 18):
            np.testing.assert_array_equal(actions[start:start + 2], np.zeros((2, 7)))
            np.testing.assert_array_equal(actions[start + 2], np.full(7, 7))
        self.assert_prefixes_executed(actions, requests)
        self.assert_deadlines(result['commits'])

    def test_expired_result_cannot_extend_original_horizon(self):
        with self.assertRaisesRegex(RuntimeError, 'fully expired'):
            self.run_fake(20, 23)

    def test_late_commit_with_short_next_prefix_stops_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, 'insufficient queued actions.*no horizon extension'):
            self.run_fake(13, 17)
