import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.compare_async_runs import LIMITATION, compare, identity
from scripts.analyze_async_q import analyze


class CompareAsyncRunsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, name, qs=(5,), tasks=(0,), session=False):
        root = self.root/name
        root.mkdir()
        cfg = dict(H=20, N=10, precision='bfloat16', state_enabled=True,
                   model_id='model', flashrt_revision='revision',
                   alignment_policy='progress', pending_arm_gain=.5,
                   action_handoff='latency_limited_progress',
                   execution_mode='asynchronous_q5_paired',
                   output_root=str(self.root/'output'))
        if session:
            cfg.update(model_lifetime='session', refresh_anchor='request')
        episodes = []
        for task in tasks:
            artifact = root/f'tasks/task{task:02d}'
            artifact.mkdir(parents=True)
            (artifact/'trace.jsonl').write_text('')
            for q in qs:
                (artifact/f'q{q}_requests.json').write_text('[]')
                episodes.append(dict(task_id=task, q=q, suite='spatial',
                                     status='success', steps=1, real_seconds=1,
                                     initial_sim_state_sha256=f'sim{task}',
                                     initial_frames_sha256=f'frames{task}',
                                     initial_raw_state=[0]*8, commits=[],
                                     artifact_dir=str(artifact.relative_to(root)),
                                     source={'original': name}))
        summary = dict(status='completed', config=cfg, episodes=episodes,
                       checkpoint_sha256='checkpoint', norm_stats_sha256='norm',
                       seed=7, init_state_index=0)
        self.save(root, summary)
        return root, summary

    def save(self, root, summary):
        (root/'summary.json').write_text(json.dumps(summary))

    def test_merge_provenance_defaults_limitations_and_read_only(self):
        a, sa = self.source('a')
        b, sb = self.source('b', (8,), session=True)
        originals = {p: p.read_bytes() for root in (a, b) for p in root.rglob('*') if p.is_file()}
        output = compare([a, b])
        summary = json.loads((output/'summary.json').read_text())
        result = json.loads((output/'analysis.json').read_text())
        self.assertEqual(output.parent.name, 'M1')
        self.assertIn('-async-final-', output.name)
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['sources'][0]['model_lifetime'], 'task')
        self.assertEqual(summary['sources'][1]['model_lifetime'], 'session')
        self.assertEqual(summary['sources'][0]['config'], sa['config'])
        self.assertEqual(summary['sources'][0]['summary_sha256'],
                         hashlib.sha256(originals[a/'summary.json']).hexdigest())
        self.assertEqual(summary['config_differences']['model_lifetime'], ['task', 'session'])
        self.assertNotIn('refresh_anchor', summary['config_differences'])
        self.assertTrue(any('生命周期变化' in w for w in result['warnings']))
        self.assertEqual(result['paired_vs_q5']['8']['success_difference']['n'], 1)
        self.assertEqual(result['limitations'], [LIMITATION]+analyze(output)['limitations'])
        self.assertTrue((output/'analysis.md').read_text().startswith('> **跨run'))
        for ep in summary['episodes']:
            self.assertTrue(Path(ep['artifact_dir']).is_absolute())
            self.assertEqual(ep['source'], {'original': Path(ep['source_run']).name})
        self.assertFalse((output/'tasks').exists())
        for path, raw in originals.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_reject_fixed_mismatches_and_missing_metadata(self):
        a, sa = self.source('a')
        b, sb = self.source('b', (8,))
        for location, key in [('config', k) for k in
                              ('H', 'N', 'precision', 'state_enabled', 'model_id',
                               'flashrt_revision', 'action_handoff', 'alignment_policy',
                               'pending_arm_gain')]+[(None, k) for k in
                              ('checkpoint_sha256', 'norm_stats_sha256', 'seed', 'init_state_index')]:
            with self.subTest(key=key):
                changed = copy.deepcopy(sb)
                (changed['config'] if location else changed)[key] = 'different'
                self.save(b, changed)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    compare([a, b])
        del sa['config']['model_id']
        with self.assertRaisesRegex(ValueError, 'model_id'):
            identity(sa)
        self.assertFalse((self.root/'output').exists())

    def test_reject_duplicate_even_if_suite_differs(self):
        a, _ = self.source('a')
        b, sb = self.source('b')
        sb['episodes'][0]['suite'] = 'goal'
        self.save(b, sb)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            compare([a, b])

    def test_reject_sync_and_postcommit_in_metadata_or_trace(self):
        a, original = self.source('a')
        for location, key, value in [('config', 'refresh_anchor', 'commit'),
                                     ('episode', 'refresh_policy', 'post_commit'),
                                     ('config', 'execution_mode', 'synchronous_quality'),
                                     ('episode', 'execution_mode', 'synchronous_quality')]:
            with self.subTest(location=location, key=key):
                summary = copy.deepcopy(original)
                target = summary['config'] if location == 'config' else summary['episodes'][0]
                target[key] = value
                self.save(a, summary)
                with self.assertRaises(ValueError):
                    compare([a])
        self.save(a, original)
        (a/'tasks/task00/trace.jsonl').write_text(json.dumps(
            dict(event='episode_start', execution_mode='synchronous_quality'))+'\n')
        with self.assertRaises(ValueError):
            compare([a])

    def test_complete_cli_and_reject_partial_or_unfinished(self):
        a, sa = self.source('a', (5, 8), range(30))
        b, sb = self.source('b', (10, 12, 15), range(30), session=True)
        for status in ('running', 'failed'):
            sb['status'] = status
            self.save(b, sb)
            with self.assertRaisesRegex(ValueError, 'not completed'):
                compare([a, b], True)
        sb['status'] = 'completed'
        self.save(b, sb)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            compare([a], True)
        sb['episodes'][0]['status'] = 'running'
        self.save(b, sb)
        with self.assertRaisesRegex(ValueError, 'incomplete episode'):
            compare([a, b], True)
        sb['episodes'][0]['status'] = 'success'
        self.save(b, sb)
        script = Path(__file__).resolve().parents[1]/'scripts/compare_async_runs.py'
        process = subprocess.run([sys.executable, str(script), '--require-complete', str(a), str(b)],
                                 capture_output=True, text=True, check=True)
        output = Path(process.stdout.strip())
        self.assertEqual(json.loads((output/'summary.json').read_text())['status'], 'completed')
        self.assertEqual(len(json.loads((output/'analysis.json').read_text())['episodes']), 150)

    def test_rendering_tolerance_rechecks_only_copies(self):
        a, sa = self.source('a')
        b, sb = self.source('b', (8,))
        sb['episodes'][0]['initial_frames_sha256'] = 'different'
        self.save(b, sb)
        frames = np.zeros((1, 100, 100, 3), dtype=np.uint8)
        np.savez(a/'tasks/task00/q5_frames.npz', frames=frames)
        frames[0, 0, 0, 0] = 1
        np.savez(b/'tasks/task00/q8_frames.npz', frames=frames)
        original = (b/'summary.json').read_bytes()
        output = compare([a, b])
        result = json.loads((output/'analysis.json').read_text())
        self.assertEqual(result['paired_vs_q5']['8']['success_difference']['n'], 1)
        comparison = result['episodes'][1]['initial_frame_comparison']
        self.assertTrue(comparison['accepted'])
        self.assertEqual(comparison['baseline_sha256'], 'frames0')
        self.assertEqual((b/'summary.json').read_bytes(), original)
        frames[0, 0, 0, 0] = 2
        np.savez(b/'tasks/task00/q8_frames.npz', frames=frames)
        with self.assertRaisesRegex(RuntimeError, 'tolerance'):
            compare([a, b])
        sb['episodes'][0]['initial_sim_state_sha256'] = 'other'
        self.save(b, sb)
        with self.assertRaisesRegex(RuntimeError, 'initial_sim_state'):
            compare([a, b])


if __name__ == '__main__':
    unittest.main()
