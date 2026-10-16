import copy
from pathlib import Path
import tempfile
import unittest

from scripts import run_async_q
from scripts.run_async_q import (configure_handoff, configure_model_lifetime, handoff_options,
                                 validate_episode, validate_pairing)


class AsyncRunnerTests(unittest.TestCase):
    def test_refresh_anchor_new_run_defaults_to_request(self):
        config = {'H': 20, 'N': 10, 'dtype': 'bfloat16'}
        before = copy.deepcopy(config)
        self.assertEqual(run_async_q.configure_refresh_anchor(config), 'request')
        self.assertEqual(config, dict(before, refresh_anchor='request'))

    def test_refresh_anchor_new_run_honors_config_and_explicit_request(self):
        for anchor in ('request', 'commit'):
            with self.subTest(anchor=anchor):
                config = {'refresh_anchor': anchor, 'H': 20}
                self.assertEqual(run_async_q.configure_refresh_anchor(config), anchor)
                self.assertEqual(config, {'refresh_anchor': anchor, 'H': 20})
                other = 'commit' if anchor == 'request' else 'request'
                self.assertEqual(run_async_q.configure_refresh_anchor(
                    config, requested=other), other)
                self.assertEqual(config, {'refresh_anchor': other, 'H': 20})

    def test_refresh_anchor_explicit_commit_is_recorded(self):
        config = {'H': 20}
        self.assertEqual(run_async_q.configure_refresh_anchor(
            config, requested='commit'), 'commit')
        self.assertEqual(config, {'H': 20, 'refresh_anchor': 'commit'})

    def test_refresh_anchor_rejects_invalid_values(self):
        for invalid in ('', 'COMMIT', 'post_commit', 'latest_due', 'unknown', 5):
            for resume in (False, True):
                with self.subTest(invalid=invalid, resume=resume):
                    with self.assertRaises(ValueError):
                        run_async_q.configure_refresh_anchor(
                            {}, requested=invalid, resume=resume)
                    with self.assertRaises(ValueError):
                        run_async_q.configure_refresh_anchor(
                            {'refresh_anchor': invalid}, resume=resume)

    def test_refresh_anchor_legacy_resume_uses_request_without_mutation(self):
        config = {'H': 20, 'refresh_policy': 'latest_due'}
        before = copy.deepcopy(config)
        self.assertEqual(run_async_q.configure_refresh_anchor(config, resume=True),
                         'request')
        self.assertEqual(config, before)
        self.assertEqual(run_async_q.configure_refresh_anchor(
            config, requested='request', resume=True), 'request')
        self.assertEqual(config, before)
        with self.assertRaises(ValueError):
            run_async_q.configure_refresh_anchor(config, requested='commit', resume=True)
        self.assertEqual(config, before)

    def test_refresh_anchor_resume_preserves_anchor_and_rejects_switch(self):
        for anchor, other in (('request', 'commit'), ('commit', 'request')):
            with self.subTest(anchor=anchor):
                config = {'refresh_anchor': anchor, 'H': 20}
                before = copy.deepcopy(config)
                self.assertEqual(run_async_q.configure_refresh_anchor(
                    config, resume=True), anchor)
                self.assertEqual(config, before)
                self.assertEqual(run_async_q.configure_refresh_anchor(
                    config, requested=anchor, resume=True), anchor)
                self.assertEqual(config, before)
                with self.assertRaises(ValueError):
                    run_async_q.configure_refresh_anchor(
                        config, requested=other, resume=True)
                self.assertEqual(config, before)

    def test_progress_handoff_refresh_anchor_only_changes_refresh_policy(self):
        # Commit anchoring is opt-in for the q5 nine-task experiment.
        for anchor, policy in ((None, 'latest_due'), ('request', 'latest_due'),
                               ('commit', 'post_commit')):
            with self.subTest(anchor=anchor):
                config = {}
                configure_handoff(config, 'latency_limited_progress')
                if anchor is not None:
                    config['refresh_anchor'] = anchor
                before = copy.deepcopy(config)
                self.assertEqual(handoff_options(config), {
                    'condition': 'latency_limited_progress',
                    'alignment_policy': 'progress', 'refresh_policy': policy,
                    'pending_arm_gain': 0.5})
                self.assertEqual(config, before)

    def test_model_lifetime_new_run_defaults_to_session(self):
        config = {'H': 20, 'N': 10, 'dtype': 'bfloat16'}
        self.assertEqual(configure_model_lifetime(config), 'session')
        self.assertEqual(config['model_lifetime'], 'session')
        self.assertIsInstance(config['worker_scope'], str)
        self.assertRegex(config['worker_scope'].lower(), r'session|all tasks|across tasks')
        self.assertNotEqual(config['worker_scope'],
                            'one task, all q; fresh process per task')
        self.assertEqual((config['H'], config['N'], config['dtype']),
                         (20, 10, 'bfloat16'))

    def test_model_lifetime_new_run_honors_configured_task(self):
        config = {'model_lifetime': 'task'}
        self.assertEqual(configure_model_lifetime(config), 'task')
        self.assertEqual(config['model_lifetime'], 'task')
        self.assertEqual(config['worker_scope'],
                         'one task, all q; fresh process per task')

    def test_model_lifetime_explicit_request_overrides_new_run_config(self):
        for configured, requested in (('task', 'session'), ('session', 'task')):
            with self.subTest(configured=configured, requested=requested):
                config = {'model_lifetime': configured}
                self.assertEqual(configure_model_lifetime(config, requested=requested),
                                 requested)
                self.assertEqual(config['model_lifetime'], requested)

    def test_model_lifetime_rejects_invalid_values(self):
        for invalid in ('', 'worker', 'SESSION', 'sessions'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    configure_model_lifetime({}, requested=invalid)
                with self.assertRaises(ValueError):
                    configure_model_lifetime({'model_lifetime': invalid})
                with self.assertRaises(ValueError):
                    configure_model_lifetime({'model_lifetime': invalid}, resume=True)
                with self.assertRaises(ValueError):
                    configure_model_lifetime({}, requested=invalid, resume=True)

    def test_model_lifetime_legacy_resume_defaults_to_task_without_mutation(self):
        config = {'H': 20, 'worker_scope': 'historical worker description'}
        before = copy.deepcopy(config)
        self.assertEqual(configure_model_lifetime(config, resume=True), 'task')
        self.assertEqual(config, before)
        self.assertEqual(configure_model_lifetime(config, requested='task', resume=True),
                         'task')
        self.assertEqual(config, before)
        with self.assertRaises(ValueError):
            configure_model_lifetime(config, requested='session', resume=True)
        self.assertEqual(config, before)

    def test_model_lifetime_resume_preserves_mode_and_rejects_switch(self):
        for lifetime, other in (('session', 'task'), ('task', 'session')):
            with self.subTest(lifetime=lifetime):
                config = {'model_lifetime': lifetime,
                          'worker_scope': 'saved worker description', 'H': 20}
                before = copy.deepcopy(config)
                self.assertEqual(configure_model_lifetime(config, resume=True), lifetime)
                self.assertEqual(config, before)
                self.assertEqual(configure_model_lifetime(
                    config, requested=lifetime, resume=True), lifetime)
                self.assertEqual(config, before)
                with self.assertRaises(ValueError):
                    configure_model_lifetime(config, requested=other, resume=True)
                self.assertEqual(config, before)

    def progress_episode(self, elapsed=7, offset=2):
        episode = self.episode()
        episode['condition'] = 'latency_limited_progress'
        episode['commits'][1]['commit_step'] = 8+elapsed
        for item, k in zip(episode['commits'], (0, offset)):
            d = min(20, item['commit_step']-item['observation_step'])
            remaining = 20-d
            scheduled = [item['commit_step'], item['commit_step']+remaining]
            item.update(expired_new_actions=k, new_prediction_remaining_actions=remaining,
                        new_prediction_remaining_interval_steps=scheduled,
                        action_alignment={
                            'policy': 'progress',
                            'elapsed_control_steps': item['commit_step']-item['observation_step'],
                            'selected_prediction_offset': k, 'discarded_prefix_actions': k,
                            'discarded_tail_actions': d-k, 'remaining_actions': remaining,
                            'retained_prediction_offset_interval': [k, k+remaining],
                            'scheduled_interval_steps': scheduled,
                            'original_horizon_end_step': item['observation_step']+20})
        return episode

    def test_profile_defaults_and_worker_options(self):
        config = {'H': 20, 'N': 10, 'dtype': 'bfloat16'}
        configure_handoff(config)
        self.assertEqual(config['action_handoff'], 'latency_limited_progress')
        configure_handoff(config, 'crop_overwrite')
        self.assertEqual(handoff_options(config), {'condition': 'crop_overwrite'})
        configure_handoff(config, 'latency_limited_progress')
        self.assertEqual(handoff_options(config), {
            'condition': 'latency_limited_progress', 'alignment_policy': 'progress',
            'refresh_policy': 'latest_due', 'pending_arm_gain': 0.5})
        self.assertEqual((config['H'], config['N'], config['dtype']), (20, 10, 'bfloat16'))

    def test_resume_preserves_profile_and_rejects_switch(self):
        for profile in ('crop_overwrite', 'latency_limited_progress'):
            config = {}
            configure_handoff(config, profile)
            before = copy.deepcopy(config)
            self.assertEqual(configure_handoff(config, resume=True), profile)
            configure_handoff(config, profile, resume=True)
            self.assertEqual(config, before)
            other = 'crop_overwrite' if profile != 'crop_overwrite' else 'latency_limited_progress'
            with self.assertRaises(ValueError):
                configure_handoff(config, other, resume=True)
        self.assertEqual(configure_handoff({}, resume=True), 'crop_overwrite')

    def test_new_run_uses_configured_profile_unless_explicitly_overridden(self):
        config = {'action_handoff': 'crop_overwrite'}
        self.assertEqual(configure_handoff(config), 'crop_overwrite')
        self.assertEqual(configure_handoff(config, 'latency_limited_progress'),
                         'latency_limited_progress')
        self.assertEqual(config['pending_arm_gain'], 0.5)

    def test_retime_keeps_thirteen_steps(self):
        import numpy as np
        from scripts.probe_delay_buffer import commit_aligned
        episode = self.progress_episode()
        actions = np.zeros((20, 7))
        actions[:, 0] = 1
        queue = {}
        expired, _, alignment = commit_aligned(
            queue, actions, 8, 15, 1, [0, 0, 0], [0.02, 0, 0], [], 'progress')
        episode['commits'][1].update(expired_new_actions=expired, action_alignment=alignment)
        validate_episode(episode, 20)
        self.assertEqual(expired, 2)
        self.assertEqual(sorted(queue), list(range(15, 28)))
        self.assertEqual(episode['commits'][1]['new_prediction_remaining_actions'], 13)

    def test_retime_rejects_extended_deadline_or_buffer(self):
        for field in ('scheduled_interval_steps', 'original_horizon_end_step',
                      'new_prediction_remaining_interval_steps', 'new_prediction_remaining_actions'):
            with self.subTest(field=field):
                episode = self.progress_episode()
                item = episode['commits'][1]
                target = item if field.startswith('new_') else item['action_alignment']
                if isinstance(target[field], list):
                    target[field] = [15, 35]
                else:
                    target[field] += 7
                item['new_queue_actions'] = 40
                with self.assertRaises(RuntimeError):
                    validate_episode(episode, 20)

    def test_retime_rejects_incorrect_prefix_tail_and_offsets(self):
        for field, value in (('discarded_prefix_actions', 7), ('discarded_tail_actions', 0),
                             ('selected_prediction_offset', -1), ('selected_prediction_offset', 8),
                             ('retained_prediction_offset_interval', [2, 21])):
            with self.subTest(field=field, value=value):
                episode = self.progress_episode()
                episode['commits'][1]['action_alignment'][field] = value
                with self.assertRaises(RuntimeError):
                    validate_episode(episode, 20)
        episode = self.progress_episode()
        episode['commits'][1]['expired_new_actions'] = 7
        with self.assertRaises(RuntimeError):
            validate_episode(episode, 20)

    def test_retime_expired_horizon_has_empty_schedule(self):
        for elapsed in (20, 23):
            validate_episode(self.progress_episode(elapsed=elapsed), 20)

    def episode(self):
        return {'condition': 'crop_overwrite', 'status': 'timeout',
                'field_checks': {'full_predictions': True, 'frames': True},
                'initial_sim_state_sha256': 'same-state',
                'initial_frames_sha256': 'same-frame',
                'commits': [
                    {'observation_step': 0, 'commit_step': 0,
                     'new_prediction_remaining_actions': 20, 'expired_new_actions': 0},
                    {'observation_step': 8, 'commit_step': 15,
                     'new_prediction_remaining_actions': 13, 'expired_new_actions': 7}]}

    def test_elapsed_control_steps_define_remaining_horizon(self):
        # Bootstrap can take wall-clock time yet expires no simulator steps.
        # Later H20 observation at 8 committed at 15 leaves 13 steps, not 20.
        validate_episode(self.episode(), 20)

    def test_rejects_failure_to_discard_elapsed_prefix(self):
        episode = self.episode()
        episode['commits'][1]['new_prediction_remaining_actions'] = 20
        with self.assertRaises(RuntimeError):
            validate_episode(episode, 20)

    def test_rejects_missing_records_not_genuine_task_timeout(self):
        episode = self.episode()
        validate_episode(episode, 20)
        episode['field_checks']['frames'] = False
        with self.assertRaises(RuntimeError):
            validate_episode(episode, 20)

    def test_rejects_initial_state_and_image_mismatch(self):
        first = self.episode()
        second = copy.deepcopy(first)
        validate_pairing([first, second])
        second['initial_frames_sha256'] = 'different'
        with self.assertRaises(RuntimeError):
            validate_pairing([first, second])

    def test_measured_render_roundoff_is_distinguished_from_scene_change(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = self.episode(), self.episode()
            first.update(q=5, artifact_dir='a')
            second.update(q=12, artifact_dir='b', initial_frames_sha256='different')
            (root/'a').mkdir()
            (root/'b').mkdir()
            image = np.zeros((1, 256, 512, 3), dtype=np.uint8)
            np.savez(root/'a/q5_frames.npz', frames=image)
            image.flat[:4] = 1
            np.savez(root/'b/q12_frames.npz', frames=image)
            validate_pairing([first, second], root)
            self.assertTrue(second['initial_frame_comparison']['accepted'])
            image.flat[0] = 2
            np.savez(root/'b/q12_frames.npz', frames=image)
            with self.assertRaises(RuntimeError):
                validate_pairing([first, second], root)


if __name__ == '__main__':
    unittest.main()
