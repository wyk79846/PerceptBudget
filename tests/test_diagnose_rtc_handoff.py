import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import diagnose_rtc_handoff as diag


class DiagnosticBatchTests(unittest.TestCase):
    def test_worker_forwards_q8_independently_of_delay(self):
        argv=['diagnostic','--reference-run','/tmp/reference','--worker','controlled_d5',
              '--parent','/tmp/results','--task-id','0','--q','8']
        with patch('sys.argv',argv),patch.object(diag,'worker',return_value=0) as worker:
            self.assertEqual(diag.main(),0)
        worker.assert_called_once_with(Path('/tmp/reference'),'controlled_d5',Path('/tmp/results'),0,5.,8,20)

    def test_h10_q5_d5_is_explicitly_forwarded(self):
        argv=['diagnostic','--reference-run','/tmp/reference','--worker','controlled_d5',
              '--parent','/tmp/results','--task-id','0','--q','5','--horizon','10']
        with patch('sys.argv',argv),patch.object(diag,'worker',return_value=0) as worker:
            self.assertEqual(diag.main(),0)
        worker.assert_called_once_with(Path('/tmp/reference'),'controlled_d5',Path('/tmp/results'),0,5.,5,10)

    def test_fixed_eight_is_not_actual_async(self):
        c = diag.configuration({}, 'controlled_d8')
        self.assertEqual(c['controlled_delay_steps'], 8)
        self.assertEqual(c['rtc_fixed_delay_estimate'], 8)
        self.assertTrue(c['rtc_paper_schedule'])
        self.assertFalse(c['explicit_actual_delay_experiment'])

    def test_ten_tasks_are_distinct_and_keep_durable_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / 'reference'
            parent = root / 'results'
            reference.mkdir()
            parent.mkdir()
            (reference / 'run.json').write_text(json.dumps({'config_snapshot':
                {'H': 20, 'N': 5, 'screen': {'task_ids': [2]}}}))
            seen = []

            def child(command, **kwargs):
                task = int(command[command.index('--task-id') + 1])
                mode = command[command.index('--worker') + 1]
                directory = Path(command[command.index('--parent') + 1])
                self.assertEqual(mode, 'controlled_d8')
                seen.append(task)
                (directory / (mode + '.json')).write_text(json.dumps({
                    'status': 'completed', 'episode': {'status': 'success'}}))
                return 0

            argv = ['diagnostic', '--reference-run', str(reference), '--conditions',
                    'controlled_d8', '--task-ids', *map(str, range(10))]
            with patch('sys.argv', argv), patch.object(diag, 'create_run', return_value=parent), \
                    patch.object(diag.subprocess, 'call', side_effect=child), \
                    patch.object(diag.runner.base, '_save_run_status'), patch('builtins.print'):
                self.assertEqual(diag.main(), 0)
            self.assertEqual(seen, list(range(10)))
            summary = json.loads((parent / 'summary.json').read_text())
            self.assertEqual(summary['expected'], 10)
            self.assertEqual(summary['successes'], 10)
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(len(list(parent.glob('task*/controlled_d8.json'))), 10)


if __name__ == '__main__':
    unittest.main()
