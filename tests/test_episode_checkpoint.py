import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import run_async_q as runner


class EpisodeCheckpointTest(unittest.TestCase):
    def test_validation_failure_preserves_measurements_and_resume_recovers_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'screen': {'suites': []}, 'state_enabled': True, 'H': 20,
                      'candidate_q': [12], 'execution_mode': 'asynchronous_q5_paired',
                      'norm_stats_path': 'test-only'}
            (root/'run.json').write_text(json.dumps({'config_snapshot': config}))
            task = {'task_id': 4, 'suite': 'libero_spatial', 'suite_task_id': 4,
                    'instruction': 'test'}
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
            env = SimpleNamespace(env=SimpleNamespace(parsed_problem={}), close=Mock())
            episode = {'task_id': 4, 'q': 12, 'status': 'success', 'power': {'test': 123}}
            session = {}
            callback = Mock()
            with patch.object(runner.base, '_check_cuda_and_native', return_value=(torch, {})), \
                 patch.object(runner.base, '_load_and_check_model', return_value=object()), \
                 patch.object(runner.base, '_load_tasks', return_value=[task]), \
                 patch('perceptbudget.libero_input.load_libero_norm_stats', return_value={}), \
                 patch.object(runner.base, '_libero_api', return_value=(None, None)), \
                 patch.object(runner.base, '_make_env', return_value=env), \
                 patch.object(runner.base, '_prepare_env_episode', return_value=({}, 20)), \
                 patch.object(runner.base, '_make_snapshot', return_value={}), \
                 patch.object(runner.base, '_infer_actions'), \
                 patch.object(runner, 'memory_snapshot', return_value={'MemAvailable_kib': 2**22}), \
                 patch.object(runner, 'validate_episode'), \
                 patch.object(runner, 'validate_pairing') as check, \
                 patch.object(runner.probe, 'run_episode', return_value=copy.deepcopy(episode)) as infer:
                check.side_effect = RuntimeError('validation failed')
                with self.assertRaisesRegex(RuntimeError, 'validation failed'):
                    runner.worker(root, 4, session, callback)
                pending = list(root.glob('tasks/task04/q12-*/pending_episode.json'))
                self.assertEqual(len(pending), 1)
                saved = json.loads(pending[0].read_text())
                self.assertEqual(saved['power'], {'test': 123})
                callback.assert_not_called()
                check.side_effect = None
                infer.side_effect = AssertionError('A completed pending episode must not be simulated twice')
                runner.worker(root, 4, session, callback)
                summary = json.loads((root/'tasks/task04/summary.json').read_text())
                self.assertEqual(summary['status'], 'completed')
                self.assertEqual(len(summary['episodes']), 1)
                self.assertEqual(summary['episodes'][0]['power'], {'test': 123})
                self.assertEqual(infer.call_count, 1)
                self.assertEqual(len(list(root.glob('tasks/task04/q12-*'))), 1)
                self.assertEqual(callback.call_count, 1)
