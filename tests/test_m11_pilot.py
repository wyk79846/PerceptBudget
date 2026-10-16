import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import probe_delay_buffer as probe
from scripts.probe_m11 import make_hook, plan, validate_result


class M11PilotTest(unittest.TestCase):
    def test_plan_is_bounded_and_balanced(self):
        specs = plan()
        self.assertEqual(len(specs), 20)
        self.assertEqual(len({s['episode_id'] for s in specs}), 20)
        for task in (0, 3):
            for disturbed in (False, True):
                self.assertEqual([s['q'] for s in specs if s['task_id'] == task
                                  and s['disturbed'] == disturbed and s['mode'] == 'sync'], [5, 8, 12])

    def test_hook_refreshes_once_and_before_action(self):
        events = []
        audit = {'after_pos': [.12, .2, .3]}
        fresh = {'akita_black_bowl_1_pos': np.array(audit['after_pos'])}
        env = SimpleNamespace(env=SimpleNamespace(_get_observations=lambda **kw: fresh))
        hook = make_hook(True, events)
        old = {'old': True}
        with patch('scripts.probe_m11.nudge_object', return_value=audit) as nudge:
            self.assertEqual(hook(env, old, 29), (old, None))
            observation, event = hook(env, old, 30)
            self.assertIs(observation, fresh)
            self.assertEqual(event['step'], 30)
            self.assertEqual(len(events), 1)
            nudge.assert_called_once_with(env, 'akita_black_bowl_1', (.02, 0., 0.))
            with self.assertRaises(RuntimeError):
                hook(env, observation, 30)

    def test_sync_recorder_waits_without_consuming_and_restarts_prefix(self):
        clock = SimpleNamespace(now=0.)
        obs = {'agentview_image': np.zeros((2, 2, 3), dtype=np.uint8),
               'robot0_eye_in_hand_image': np.zeros((2, 2, 3), dtype=np.uint8)}
        env = SimpleNamespace(actions=[])
        def step(action):
            env.actions.append(action.copy())
            clock.now += .05
            return obs, 0., False, {}
        env.step = step
        env.check_success = lambda: len(env.actions) == 7
        class Future:
            def __init__(self, fn):
                self.fn, self.ready = fn, False
            def done(self):
                return self.ready
            def result(self):
                if not self.ready:
                    self.value, self.ready = self.fn(), True
                return self.value
        class Executor:
            def __init__(self, **kw):
                pass
            def submit(self, fn):
                return Future(fn)
            def shutdown(self, **kw):
                pass
        predict_steps = []
        def infer(model, snapshot, prompt, horizon, timing):
            predict_steps.append(len(env.actions))
            timing['inference_started_monotonic'] = clock.now
            clock.now += .4
            timing['inference_ended_monotonic'] = clock.now
            actions = np.zeros((horizon, 7))
            actions[:, 0] = np.arange(horizon)
            actions[:, 6] = -1
            return actions
        def scene(*args):
            return {'raw_state': [len(env.actions)*.01]+[0]*7,
                    'sim_state': [len(env.actions)*.05, 0.], 'contacts': [], 'observations': {}}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(probe.base, '_prepare_env_episode', return_value=(obs, 20)), \
             patch.object(probe.base, '_seed_process'), \
             patch.object(probe.base, '_make_snapshot', return_value={'raw_state': np.zeros(8)}), \
             patch.object(probe.base, '_infer_actions', side_effect=infer), \
             patch.object(probe.base, '_power_result', return_value={}), \
             patch.object(probe, 'snapshot_scene', side_effect=scene), \
             patch.object(probe, 'PowerMonitor'), \
             patch.object(probe, 'device_state', return_value={}), \
             patch.object(probe, 'ThreadPoolExecutor', Executor), \
             patch.object(probe.time, 'monotonic', side_effect=lambda: clock.now), \
             patch.object(probe.time, 'sleep', side_effect=lambda n: setattr(clock, 'now', clock.now+n)):
            stream = io.StringIO()
            result = probe.run_episode(env, {'task_id': 0, 'task_name': 'test',
                'instruction': 'test', 'max_steps': 7}, 3, None, {},
                {'state_enabled': True, 'H': 20, 'execution_mode': 'synchronous_quality'},
                np, None, Path(directory), stream, synchronous=True,
                condition='synchronous_quality')
            self.assertEqual(predict_steps, [0, 3, 6])
            self.assertEqual([a[0] for a in env.actions], [0, 1, 2, 0, 1, 2, 0])
            self.assertEqual(result['actual_refresh_intervals_steps'], [3, 3])
            self.assertEqual(result['underflow_steps'], [])
            self.assertTrue(all(c['control_steps_until_commit'] == 0 for c in result['commits']))
            self.assertEqual(len(json.loads((Path(directory)/'q3_requests.json').read_text())), 3)
            result.update(mode='sync', disturbed=False, disturbance_events=[])
            validate_result(result, {'H': 20})

    def test_sync_cannot_silently_enable_async_alignment(self):
        with self.assertRaisesRegex(ValueError, 'unchanged fresh prefix'):
            probe.run_episode(None, {}, 5, None, {}, {}, np, None, None, None,
                              synchronous=True, alignment_policy='progress')


if __name__ == '__main__':
    unittest.main()
