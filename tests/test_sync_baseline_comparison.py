import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.compare_sync_baseline import compare, load_source


class SyncBaselineComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = dict(H=20, N=10, state_enabled=True, precision='bfloat16',
                        model_id='model', flashrt_revision='rev', seed=7,
                        init_state_index=0, output_root=str(self.root/'output'),
                        checkpoint_path='/historical/checkpoint',
                        flashrt_loading_patch_sha256='recorded-patch')
        self.sync = [self.source('sync-a', True, range(27)),
                     self.source('sync-b', True, range(27, 30))]
        self.async_run = self.source('async', False, range(30))

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def source(self, name, sync, tasks):
        root = self.root/name
        root.mkdir()
        cfg = dict(self.cfg, execution_mode='synchronous_quality' if sync else 'asynchronous_q5_paired')
        self.write(root/'run.json', {'config_snapshot': cfg})
        eps, trace = [], []
        for task in tasks:
            for q in ((5,) if sync else (5, 8, 10, 12, 15)):
                ep = dict(task_id=task, q=q, suite=('spatial', 'object', 'goal')[task//10],
                          task_name=f'task{task}', status='success')
                rows = []
                for step in range(3):
                    state = [step*.01, 0, 0, 0, 0, 0, 0, 0]
                    action = [1 if step == 0 else 0, 0, 0, 0, 0, 0, 1 if step == 0 else -1]
                    row = dict(event='control_step', task_id=task, q=q)
                    if sync:
                        row.update(control_step=step, raw_state=state, executed_action=action)
                    else:
                        row.update(step=step, action=action, before={'raw_state': state},
                                   after={'raw_state': state, 'contacts': []})
                    rows.append(row)
                if sync:
                    ep.update(control_steps=3, real_task_seconds=2, simulated_control_seconds=.15)
                    trace.extend(rows)
                else:
                    artifact = root/f'task{task}-q{q}'
                    artifact.mkdir()
                    ep.update(steps=3, real_seconds=1, simulated_seconds=.15, artifact_dir=str(artifact))
                    (artifact/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                eps.append(ep)
        if sync:
            (root/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
        self.write(root/'summary.json', dict(episodes=eps, seed=7, init_state_index=0,
                                           **({} if sync else {'config': cfg})))
        return root

    def change(self, path, mutation):
        value = json.loads(path.read_text())
        mutation(value)
        self.write(path, value)

    def test_success_limited_evidence_read_only_and_reference(self):
        self.write(self.async_run/'analysis.json', {'q': {'5': {'buffer': 'original'}}})
        originals = {p: p.read_bytes() for root in [*self.sync, self.async_run]
                     for p in root.rglob('*') if p.is_file()}
        output = compare(self.sync, self.async_run)
        result = json.loads((output/'analysis.json').read_text())
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['comparison_baseline'], 'sync_q5')
        self.assertFalse(result['full_scene_initialization_verified'])
        self.assertEqual(result['async_analysis_reference']['byq']['5']['buffer'], 'original')
        for row in result['byq'].values():
            self.assertEqual(row['success_difference']['n'], 30)
            self.assertIsNotNone(row['success_difference']['ci95'])
            self.assertEqual(row['common_success_time_difference']['real_seconds_difference']['mean'], -1)
            task = row['tasks'][0]
            self.assertEqual(task['initial_robot_state_max_abs_delta'], 0)
            self.assertEqual(set(task['trajectory']['baseline_stage_eef_distance_m']), {'unknown'})
            self.assertEqual(task['command_events']['sync_q5']['gripper_switch']['steps'], [1])
        self.assertIn('historical_checkpoint_sha256', result['sources'][0]['missing_evidence'])
        self.assertEqual(len(result['sources'][0]['trace_sha256']), 1)
        md = (output/'analysis.md').read_text()
        self.assertLess(md.index('同步未存checkpoint SHA'), md.index('|q / suite|'))
        self.assertLess(md.index('不能凭30任务CI声称非劣'), md.index('|q / suite|'))
        for p, raw in originals.items():
            self.assertEqual(p.read_bytes(), raw)
        # Other evaluator tests may already import torch; check a fresh process.
        subprocess.run([sys.executable, '-c',
                        'import sys; import scripts.compare_sync_baseline; '
                        'assert "torch" not in sys.modules'],
                       cwd=Path(__file__).resolve().parents[1],
                       capture_output=True, text=True, check=True)

    def test_final_summary_quality_and_supply(self):
        path = self.async_run/'summary.json'
        summary = json.loads(path.read_text())
        losses = {5: 10, 8: 5, 10: 3, 12: 2, 15: 6}
        for ep in summary['episodes']:
            if ep['task_id'] < losses[ep['q']]:
                ep['status'] = 'timeout'
            if ep['task_id'] == 0 and ep['q'] == 12:
                ep['underflow_steps'] = [1]
        summary['config']['baseline_q'] = 5
        self.write(path, summary)
        output = compare(self.sync, self.async_run)
        md = (output/'analysis.md').read_text()
        for q, drop in [(5, 33.3), (8, 16.7), (10, 10), (12, 6.7), (15, 20)]:
            self.assertIn(f'q{q}点估计下降{drop:.1f}pp', md)
        self.assertIn('目标：无人满足', md)
        self.assertIn('上限：q10、q12', md)
        self.assertIn('q12=1步', md)
        self.assertEqual(json.loads((output/'run.json').read_text())['config_snapshot']['baseline_q'], 5)

    def test_missing_task(self):
        self.change(self.sync[1]/'summary.json', lambda d: d['episodes'].pop())
        with self.assertRaisesRegex(ValueError, 'missing task'):
            compare(self.sync, self.async_run)

    def test_cfg_seed_init_and_patch_mismatch(self):
        path = self.async_run/'summary.json'
        original = json.loads(path.read_text())
        for key in ('H', 'N', 'state_enabled', 'precision', 'model_id', 'flashrt_revision',
                    'checkpoint_path', 'flashrt_loading_patch_sha256', 'seed', 'init_state_index'):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed['config'][key] = 'different'
                self.write(path, changed)
                with self.assertRaisesRegex(ValueError, 'mismatch|inconsistent'):
                    compare(self.sync, self.async_run)
        self.write(path, original)

    def test_state_mismatch_and_tolerance(self):
        path = self.async_run/'task0-q5/trace.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['before']['raw_state'][7] = 2e-8
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'robot state mismatch'):
            compare(self.sync, self.async_run)
        rows[0]['before']['raw_state'][7] = 1e-9
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.assertTrue(compare(self.sync, self.async_run).exists())

    def test_duplicate_missing_steps_and_truncated_json(self):
        path = self.sync[0]/'trace.jsonl'
        raw = path.read_text()
        lines = raw.splitlines(keepends=True)
        for bad in (lines[0]+raw, ''.join(lines[1:]), raw+'{'):
            path.write_text(bad)
            with self.assertRaises(ValueError):
                load_source(self.sync[0], True)
        path.write_text(raw)
        self.change(self.sync[0]/'summary.json', lambda d: d['episodes'][0].update(control_steps=4))
        with self.assertRaisesRegex(ValueError, 'missing control steps'):
            load_source(self.sync[0], True)

    def test_suite_name_duplicate_and_cli(self):
        path = self.async_run/'summary.json'
        original = json.loads(path.read_text())
        for key in ('suite', 'task_name'):
            changed = copy.deepcopy(original)
            changed['episodes'][0][key] = 'wrong'
            self.write(path, changed)
            with self.assertRaisesRegex(ValueError, key+' mismatch'):
                compare(self.sync, self.async_run)
        self.write(path, original)
        with self.assertRaisesRegex(ValueError, 'duplicate sync task'):
            compare([self.sync[0], self.sync[0]], self.async_run)
        process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/compare_sync_baseline.py'),
                                  '--sync-runs', *map(str, self.sync), '--async-run', str(self.async_run),
                                  '--output-root', str(self.root/'cli-output')],
                                 text=True, capture_output=True, check=True)
        self.assertTrue((Path(process.stdout.strip())/'analysis.md').exists())


if __name__ == '__main__':
    unittest.main()
