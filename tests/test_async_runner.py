import copy
import unittest

from scripts.run_async_q import validate_episode, validate_pairing


class AsyncRunnerTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
