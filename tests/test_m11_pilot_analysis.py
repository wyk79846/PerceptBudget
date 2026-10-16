import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_m11_pilot import analyze


class M11AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.episodes = []
        self.raw_paths = []

    def add_episode(self, task=0, mode='sync', q=5, disturbed=False, status='success'):
        eid = f'task{task:02d}/{mode}-q{q}-'+('nudge' if disturbed else 'clean')
        artifact = self.root/eid
        artifact.mkdir(parents=True)
        rows = [dict(event='control_step', step=s, action=[0, 0, 0, 0, 0, 0, 1],
                     before={'raw_state': [s*.01, 0, 0]}, after={'raw_state': [(s+1)*.01, 0, 0]},
                     source=dict(prediction_id=0, origin_step=0, offset=s),
                     queue_before_pop_actions=20-s, applied_arm_gain=.5) for s in range(3)]
        trace = artifact/'trace.jsonl'
        trace.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        requests = artifact/f'q{q}_requests.json'
        requests.write_text(json.dumps([dict(prediction_id=0, observation_step=0,
                                            actions=[[0, 0, 0, 0, 0, 0, 1]]*20)]))
        self.raw_paths.extend([trace, requests])
        ep = dict(episode_id=eid, mode=mode, disturbed=disturbed, q=q, task_id=task,
                  status=status, steps=3, real_seconds=q*.1, simulated_seconds=.15,
                  artifact_dir=eid, commits=[dict(prediction_id=0, observation_step=0,
                  commit_step=0, control_steps_until_commit=0, new_prediction_remaining_actions=20)],
                  disturbance_events=[{'step': 30}] if disturbed else [])
        self.episodes.append(ep)
        return ep

    def run_analysis(self):
        path = self.root/'summary.json'
        path.write_text(json.dumps(dict(status='running', episodes=self.episodes)))
        originals = {p: p.read_bytes() for p in [path, *self.raw_paths]}
        result = analyze(self.root)
        for p, data in originals.items():
            self.assertEqual(p.read_bytes(), data)
        return result

    def test_partial_single_baseline_unknown_and_frame_boundaries(self):
        ep = self.add_episode()
        result = self.run_analysis()
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['missing_episodes']), 19)
        stages = json.loads((self.root/'stage_proxy.json').read_text())[ep['episode_id']]
        self.assertTrue(all(s['phase'] == 'unknown' for s in stages['steps']))
        self.assertTrue(all(s['evidence']['finger_target_contact'] is None for s in stages['steps']))
        split = stages['split']['phase:unknown']
        self.assertIn('after', split['note'])
        self.assertEqual(split['records'][0]['original_H_expiry_step'], 20)
        self.assertEqual(split['records'][2]['prediction_offset'], 2)
        frames = json.loads((self.root/'review_frames.json').read_text())
        self.assertTrue(all(f['post_action_frame_index'] == f['before_frame_index']+1 for f in frames))
        self.assertEqual(max(f['post_action_frame_index'] for f in frames), 3)
        self.assertIsNone(result['B_safe'])
        self.assertEqual(len(result['episodes'][0]['raw_prediction_events']), 1)
        self.run_analysis()  # derived artifacts may be overwritten incrementally

    def test_missing_baseline_explicit_and_failed_q_breaks_exploration(self):
        self.add_episode(mode='async', q=8)
        result = self.run_analysis()
        self.assertTrue(result['paired_vs_syncq5']['async-q8-clean']['missing_pairs'][0]['missing_baseline'])
        self.add_episode()
        self.add_episode(q=8, status='timeout')
        self.add_episode(q=12)
        result = self.run_analysis()
        exploration = next(e for e in result['quality_exploration'] if e['task_id']==0
                           and e['mode']=='sync' and not e['disturbed'])
        self.assertEqual(exploration['largest_tested_successful_q'], 5)
        self.assertEqual(exploration['local_exploratory_seconds'], .25)

    def test_two_tasks_pairs_ci_and_disturbance_report(self):
        for task in (0, 3):
            self.add_episode(task=task)
            self.add_episode(task=task, disturbed=True, status='timeout' if task == 3 else 'success')
        result = self.run_analysis()
        pair = result['clean_vs_nudge']['sync-q5']
        self.assertEqual(pair['success_difference']['n'], 2)
        self.assertEqual(pair['success_difference']['loss'], 1)
        self.assertEqual(pair['time']['real_seconds_mutual_success_difference']['n'], 1)
        self.assertEqual(pair['tasks'][0]['trajectory']['baseline_coverage'], 1)
        report = (self.root/'analysis.md').read_text()
        self.assertIn('仅2个任务', report)
        self.assertIn('nudge−clean', report)
        self.assertIn('timeout/3/', report)

    def test_duplicate_or_missing_trace_never_silently_dropped(self):
        ep = self.add_episode()
        self.episodes.append(dict(ep))
        with self.assertRaises(ValueError):
            self.run_analysis()
        self.episodes.pop()
        (self.root/ep['artifact_dir']/'trace.jsonl').unlink()
        self.raw_paths = []
        with self.assertRaises(FileNotFoundError):
            self.run_analysis()

    def test_missing_requests_and_step_mismatch(self):
        ep = self.add_episode()
        request = self.root/ep['artifact_dir']/'q5_requests.json'
        request.unlink()
        self.raw_paths.remove(request)
        self.assertIn('missing requests', self.run_analysis()['warnings'][0])
        ep['steps'] = 4
        with self.assertRaises(ValueError):
            self.run_analysis()
