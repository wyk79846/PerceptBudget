"""Deterministic control-loop checks without a model, GPU or simulator."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import probe_delay_buffer as probe


class PrefixHoldEpisodeTest(unittest.TestCase):
    def run_fake(self, delay, policy='hold', max_steps=15, q=5, advance=False, rtc_merge=False):
        clock = SimpleNamespace(now=0.)
        obs = {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in
               ('agentview_image', 'robot0_eye_in_hand_image')}
        env = SimpleNamespace(actions=[])
        def step(action):
            env.actions.append(action.copy())
            clock.now += .05
            return obs, 0., False, {}
        env.step, env.check_success = step, lambda: False
        class Future:
            def __init__(self, fn):
                self.fn, self.start, self.ready = fn, len(env.actions), False
            def done(self):
                return self.ready or len(env.actions) >= self.start + delay
            def result(self):
                if not self.ready:
                    self.value, self.ready = self.fn(), True
                return self.value
        class Executor:
            def __init__(self, **kw):
                self.pending = None
            def submit(self, fn):
                if self.pending is not None:
                    assert self.pending.ready, 'concurrent inference submitted'
                self.pending = Future(fn)
                return self.pending
            def shutdown(self, **kw): pass
        class Adapter:
            prefix_len = 7
            model = SimpleNamespace(_pipe=SimpleNamespace(norm_stats={
                'actions': {'q01': [-1.] * 7, 'q99': [1.] * 7}}))
            def predict(self, snapshot, prompt, timing, raw_prefix=None, prefix_len=None):
                timing.update(inference_started_monotonic=clock.now,
                              inference_ended_monotonic=clock.now, prefix_exact=True)
                raw = np.repeat(np.arange(20, dtype=np.float32)[:, None], 32, axis=1)
                if raw_prefix is not None:
                    raw[:prefix_len] = raw_prefix
                if rtc_merge:
                    from flash_rt.core.utils.actions import unnormalize_actions
                    return unnormalize_actions(raw, self.model._pipe.norm_stats)[:, :7], raw
                return raw[:, :7].copy(), raw
        def scene(*args):
            return {'raw_state': [0.]*8, 'sim_state': [len(env.actions)*.05],
                    'contacts': [], 'observations': {}}
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(probe.base, '_prepare_env_episode', return_value=(obs, 20)), \
             patch.object(probe.base, '_seed_process'), \
             patch.object(probe.base, '_make_snapshot', return_value={'raw_state': np.zeros(8)}), \
             patch.object(probe.base, '_power_result', return_value={}), \
             patch.object(probe, 'snapshot_scene', side_effect=scene), \
             patch.object(probe, 'PowerMonitor'), \
             patch.object(probe, 'device_state', return_value={}), \
             patch.object(probe, 'ThreadPoolExecutor', Executor), \
             patch.object(probe.time, 'monotonic', side_effect=lambda: clock.now), \
             patch.object(probe.time, 'sleep', side_effect=lambda n: setattr(clock, 'now', clock.now+n)):
            result = probe.run_episode(env, {'task_id': 0, 'task_name': 'fake',
                'instruction': 'fake', 'max_steps': max_steps}, q, None, {},
                {'state_enabled': True, 'H': 20, 'prefix_lock': {
                    'late_policy': policy, 'advance_handoff': advance, 'rtc_merge': rtc_merge}},
                np, None, Path(directory), stream, Adapter())
            requests = json.loads((Path(directory)/f'q{q}_requests.json').read_text())
        return result, env.actions, requests

    def test_rtc_merge_preserves_executed_prefix_and_raw_lineage(self):
        from flash_rt.core.utils.actions import unnormalize_actions
        result, actions, requests = self.run_fake(7, max_steps=25, q=8, advance=True, rtc_merge=True)
        self.assertTrue(result['prefix_rtc_merge'])
        self.assertEqual([c['commit_step'] for c in result['commits']], [0, 8, 16, 24])
        self.assertGreater(result['commits'][1]['rtc_action_fusion']['fused_steps'], 0)
        records = {r['target_step']: r for r in result['commits'][1]['rtc_action_fusion']['records']}
        for step in range(8, 16):
            np.testing.assert_array_equal(actions[step], records[step]['action'])
            np.testing.assert_array_equal(actions[step], unnormalize_actions(
                np.asarray(records[step]['raw'], dtype=np.float32),
                {'actions': {'q01': [-1.] * 7, 'q99': [1.] * 7}})[:7])
        for i in range(7):
            np.testing.assert_array_equal(requests[2]['committed_prefix_raw'][i], records[9 + i]['raw'])
            np.testing.assert_array_equal(requests[2]['committed_prefix_actions'][i], actions[9 + i])

    def test_late_ticks_zero_then_resumes_at_seven(self):
        result, actions, requests = self.run_fake(9)
        self.assertEqual(result['late_hold_steps'], 2)
        self.assertEqual(result['late_hold_count'], 1)
        self.assertAlmostEqual(result['late_hold_wall_seconds'], .1)
        np.testing.assert_array_equal(actions[12:14], np.zeros((2, 7)))
        np.testing.assert_array_equal(actions[14], np.full(7, 7))
        self.assertEqual(result['commits'][1]['new_prediction_remaining_interval_steps'], [14, 25])
        self.assertEqual(len(requests[1]['committed_prefix_raw'][0]), 32)

    def test_early_result_waits_until_boundary(self):
        result, actions, _ = self.run_fake(2)
        self.assertEqual(result['late_hold_count'], 0)
        self.assertEqual(result['commits'][1]['commit_step'], 12)
        np.testing.assert_array_equal(actions[5:12], np.repeat(np.arange(5, 12)[:, None], 7, axis=1))

    def test_default_discard_still_discards_late_result(self):
        result, actions, _ = self.run_fake(9, 'discard')
        self.assertEqual(result['late_discarded_predictions'], 1)
        self.assertEqual(result['late_hold_count'], 0)
        np.testing.assert_array_equal(actions[12], np.full(7, 12))

    def test_short_next_prefix_ends_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, 'insufficient queued actions.*no horizon extension'):
            self.run_fake(16, max_steps=26)

    def test_fully_expired_result_ends_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, 'fully expired'):
            self.run_fake(20, max_steps=26)
