import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_async_q import (analyze, bootstrap_mean, buffer_statistics,
    clopper_pearson, command_events, compare_rows, paired_success,
    prediction_events, render, stage_proxy)


def row(step, action=None, position=None):
    scene = {'raw_state': (position or [step*.01, 0, 0])+[0]*5,
             'contacts': [], 'observations': {}}
    return {'step': step, 'event': 'control_step', 'action': action or [.2, 0, 0, 0, 0, 0, -1],
            'before': scene, 'after': scene}


class AsyncAnalysisTest(unittest.TestCase):
    def test_bootstrap_commit_does_not_expire_wall_time(self):
        result = buffer_statistics([{'prediction_id': 0, 'commit_step': 0,
            'control_steps_until_commit': 0, 'expired_new_actions': 0,
            'new_queue_actions': 20, 'old_queue_remaining_actions': 0}])
        self.assertEqual(result['new_valid_seconds']['mean'], 1)
        self.assertEqual(result['old_queue_seconds']['mean'], 0)

    def test_h20_origin8_commit15_leaves13(self):
        result = buffer_statistics([{'prediction_id': 1, 'observation_step': 8,
            'commit_step': 15, 'control_steps_until_commit': 7,
            'expired_new_actions': 7, 'new_queue_actions': 13,
            'new_prediction_remaining_actions': 13, 'old_queue_remaining_actions': 5}])
        self.assertEqual(result['records'][0]['new_valid_actions'], 13)
        self.assertEqual(result['new_valid_seconds']['mean'], .65)
        self.assertEqual(result['old_queue_seconds']['mean'], .25)

    def test_steady_buffer_excludes_bootstrap_but_preserves_all(self):
        commits = [
            {'prediction_id': 0, 'observation_step': 0, 'expired_new_actions': 0,
             'new_queue_actions': 20, 'old_queue_remaining_actions': 0},
            {'prediction_id': 1, 'observation_step': 8, 'expired_new_actions': 7,
             'new_queue_actions': 13, 'old_queue_remaining_actions': 5},
            {'prediction_id': 0, 'observation_step': 16, 'expired_new_actions': 8,
             'new_queue_actions': 12, 'old_queue_remaining_actions': 4}]
        result = buffer_statistics(commits)
        self.assertEqual(result['new_valid_seconds']['n'], 3)
        self.assertEqual(result['steady_state_new_valid_seconds']['n'], 2)
        self.assertAlmostEqual(result['steady_state_new_valid_seconds']['mean'], .625)
        self.assertAlmostEqual(result['steady_state_old_queue_seconds']['mean'], .225)
        self.assertEqual(result['bootstrap_new_valid_seconds']['mean'], 1)
        self.assertEqual(result['bootstrap_old_queue_seconds']['mean'], 0)

    def test_success_intervals_not_degenerate(self):
        self.assertLess(clopper_pearson(30, 30)[0], 1)
        interval = paired_success([True]*30, [True]*30)['ci95']
        self.assertLess(interval[0], 0)
        self.assertGreater(interval[1], 0)
        self.assertGreater(interval[1]-interval[0], .1)

    def test_discordant_pair_direction(self):
        result = paired_success([True, False, True], [False, True, False])
        self.assertEqual(result['loss'], 2)
        self.assertEqual(result['gain'], 1)
        self.assertAlmostEqual(result['difference'], -1/3)

    def test_task_bootstrap_reproducible(self):
        self.assertEqual(bootstrap_mean([1, 2, 4]), bootstrap_mean([1, 2, 4]))
        self.assertTrue(bootstrap_mean([0, 0])['constant_sample'])
        single = bootstrap_mean([42])
        self.assertEqual(single['n'], 1)
        self.assertEqual(single['mean'], 42)
        self.assertIsNone(single['ci95'])
        self.assertIn('fewer than 2', single['reason'])
        self.assertEqual(bootstrap_mean([0, 0])['ci95'], [0, 0])

    def test_events_and_zero_gripper(self):
        actions = [[.2, 0, 0, 0, 0, 0, -1], [.05, 0, 0, 0, 0, 0, 0],
                   [0, .05, 0, 0, 0, 0, 1]]
        events = command_events([row(i, a) for i, a in enumerate(actions)])
        self.assertIn({'type': 'deceleration', 'step': 1}, events)
        self.assertIn({'type': 'turn', 'step': 2}, events)
        self.assertIn({'type': 'gripper_switch', 'step': 2}, events)

    def test_forecast_tail_right_censored_not_false_positive(self):
        actions = [[.2, 0, 0, 0, 0, 0, -1] for _ in range(20)]
        for i in range(15, 20):
            actions[i] = [.2, 0, 0, 0, 0, 0, 1]
        result = prediction_events([row(i) for i in range(10)],
            [{'prediction_id': 1, 'observation_step': 0, 'actions': actions}],
            [{'prediction_id': 1, 'commit_step': 0}])[0]
        self.assertEqual(result['unmatched_predictions'], [])
        self.assertEqual(result['unmatched_actual_commands'], [])
        self.assertEqual(result['right_censored_forecast_events'], [{'type': 'gripper_switch', 'step': 15}])

    def test_matching_is_one_to_one(self):
        actions = [[.2, 0, 0, 0, 0, 0, -1] for _ in range(20)]
        for i in range(5, 20):
            actions[i] = [.2, 0, 0, 0, 0, 0, 1]
        rows = [row(i, a) for i, a in enumerate(actions)]
        result = prediction_events(rows,
            [{'prediction_id': 1, 'observation_step': 0, 'actions': actions}],
            [{'prediction_id': 1, 'commit_step': 2}])[0]
        self.assertEqual(len(result['matches']), 1)
        self.assertEqual(result['matches'][0]['error_steps_actual_minus_predicted'], 0)

    def test_retimed_chunk_events_follow_actual_scheduled_offsets(self):
        actions = [[.2, 0, 0, 0, 0, 0, -1] for _ in range(20)]
        for i in range(5, 20):
            actions[i][6] = 1
        # observation5, arrival12, retained model offsets3..15: offset5 at14.
        actual = [row(i, [.2, 0, 0, 0, 0, 0, 1 if i >= 14 else -1])
                  for i in range(25)]
        commit = {'prediction_id': 1, 'commit_step': 12,
                  'action_alignment': {'retained_prediction_offset_interval': [3, 16],
                                       'remaining_actions': 13}}
        result = prediction_events(actual,
            [{'prediction_id': 1, 'observation_step': 5, 'actions': actions}], [commit])[0]
        match = next(m for m in result['matches'] if m['type'] == 'gripper_switch')
        self.assertEqual(match['predicted_step'], 14)
        self.assertEqual(match['error_steps_actual_minus_predicted'], 0)

    def test_retiming_does_not_turn_discarded_prefix_count_into_extra_buffer(self):
        result = buffer_statistics([{'prediction_id': 1, 'control_steps_until_commit': 7,
            'expired_new_actions': 3, 'new_prediction_remaining_actions': 13,
            'action_alignment': {'discarded_prefix_actions': 3, 'discarded_tail_actions': 4}}])
        self.assertTrue(result['records'][0]['h20_consistent'])

    def test_retimed_zero_gripper_retains_full_sign_history(self):
        actions = [[.2, 0, 0, 0, 0, 0, 0] for _ in range(20)]
        actions[0][6] = -1
        for i in range(5, 20):
            actions[i][6] = 1
        actual = [row(i, [.2, 0, 0, 0, 0, 0, 1 if i >= 14 else -1])
                  for i in range(25)]
        result = prediction_events(actual,
            [{'prediction_id': 1, 'observation_step': 5, 'actions': actions}],
            [{'prediction_id': 1, 'commit_step': 12,
              'action_alignment': {'retained_prediction_offset_interval': [3, 16],
                                   'remaining_actions': 13}}])[0]
        self.assertTrue(any(m['type'] == 'gripper_switch' and m['predicted_step'] == 14
                            for m in result['matches']))

    def test_compare_common_prefix_and_eef_only_xyz(self):
        base = [row(i) for i in range(4)]
        candidate = [row(i, position=[i*.01, .1, 0]) for i in range(2)]
        result = compare_rows(base, candidate)
        self.assertEqual(result['common_prefix_steps'], 2)
        self.assertEqual(result['baseline_coverage'], .5)
        self.assertAlmostEqual(result['eef_xyz_distance_m']['mean'], .1)
        self.assertEqual(result['translation_command_l2']['mean'], 0)

    def test_proxy_contact_not_grasp_truth(self):
        r = row(0)
        r['after']['contacts'] = [{'geom1': 'gripper0_finger1', 'geom2': 'bowl_visual'}]
        self.assertEqual(stage_proxy(r), 'finger_external_contact_proxy')
        r['after']['contacts'] = [{'geom1': 'gripper0_finger1', 'geom2': 'robot0_link'}]
        self.assertEqual(stage_proxy(r), 'noncontact_low_motion_proxy')

    def make_run(self, root, episodes, mode='asynchronous_q5_paired'):
        (root/'summary.json').write_text(json.dumps({'status': 'running',
            'config': {'execution_mode': mode}, 'episodes': episodes}))
        for ep in episodes:
            artifact = root/ep['artifact_dir']; artifact.mkdir(parents=True, exist_ok=True)
            with (artifact/'trace.jsonl').open('a') as stream:
                stream.write(json.dumps({'event': 'episode_start', 'q': ep['q'], 'execution_mode': mode})+'\n')
                for i in range(3):
                    stream.write(json.dumps(dict(row(i), q=ep['q']))+'\n')
            (artifact/f"q{ep['q']}_requests.json").write_text('[]')

    def test_missing_pair_not_imputed_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = [{'task_id': t, 'suite': 'spatial', 'q': q, 'status': 'success',
                         'steps': 3, 'real_seconds': 1, 'commits': [], 'artifact_dir': f'tasks/task{t:02d}'}
                        for t, q in [(0, 5), (0, 8), (1, 8)]]
            self.make_run(root, episodes)
            result = analyze(root)
            pair = result['paired_vs_q5']['8']
            self.assertEqual(pair['success_difference']['n'], 1)
            self.assertEqual(pair['missing_pairs'][0]['missing_q'], 5)
            self.assertIn('q=5', render(result))
            json.dumps(result, allow_nan=False)

    def test_sync_baseline_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, [], mode='synchronous_quality')
            with self.assertRaisesRegex(ValueError, 'synchronous'):
                analyze(root)

    def test_sync_episode_excluded_in_async_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ep = {'task_id': 0, 'q': 5, 'status': 'success', 'execution_mode': 'synchronous_quality',
                  'artifact_dir': 'tasks/task00'}
            self.make_run(root, [ep])
            self.assertEqual(analyze(root)['q']['5']['n'], 0)

    def test_cli_incremental_outputs_preserve_raw_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = [{'task_id': 0, 'suite': 'spatial', 'q': q, 'status': 'success',
                         'steps': 3, 'real_seconds': 1, 'commits': [], 'artifact_dir': 'tasks/task00'}
                        for q in (5, 8)]
            self.make_run(root, episodes)
            raw = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            script = Path(__file__).resolve().parents[1]/'scripts/analyze_async_q.py'
            for _ in range(2):
                subprocess.run([sys.executable, str(script), str(root)], check=True, capture_output=True)
            for p, content in raw.items():
                self.assertEqual(p.read_bytes(), content)
            result = json.loads((root/'analysis.json').read_text())
            self.assertEqual(result['paired_vs_q5']['8']['success_difference']['n'], 1)
            self.assertTrue((root/'analysis.md').exists())

    def test_initial_conditions_mismatch_excludes_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = [{'task_id': 0, 'suite': 'spatial', 'q': q, 'status': 'success',
                         'initial_sim_state_sha256': str(q), 'steps': 3, 'real_seconds': 1,
                         'commits': [], 'artifact_dir': 'tasks/task00'} for q in (5, 8)]
            self.make_run(root, episodes)
            result = analyze(root)
            self.assertEqual(result['paired_vs_q5']['8']['success_difference']['n'], 0)
            self.assertEqual(result['q']['8']['n'], 1)

    def test_completion_times_exclude_timeouts_and_require_mutual_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [(0, 5, 'success', 10), (0, 8, 'success', 12),
                     (1, 5, 'success', 20), (1, 8, 'timeout', 100),
                     (2, 5, 'timeout', 90), (2, 8, 'success', 30)]
            episodes = [{'task_id': t, 'suite': 'spatial', 'q': q, 'status': status,
                         'steps': 3, 'real_seconds': seconds, 'commits': [],
                         'artifact_dir': f'tasks/task{t:02d}'} for t, q, status, seconds in cases]
            self.make_run(root, episodes)
            result = analyze(root)
            self.assertEqual(result['q']['8']['attempt_real_seconds']['n'], 3)
            self.assertEqual(result['q']['8']['successful_completion_real_seconds']['mean'], 21)
            self.assertEqual(result['q']['5']['successful_completion_real_seconds']['mean'], 15)
            metrics = result['paired_vs_q5']['8']['continuous_metrics']
            self.assertEqual(metrics['real_seconds_difference']['n'], 1)
            self.assertEqual(metrics['real_seconds_difference']['mean'], 2)
            self.assertIsNone(metrics['real_seconds_difference']['ci95'])
            self.assertEqual(metrics['attempt_elapsed_difference']['n'], 3)
            self.assertAlmostEqual(metrics['attempt_elapsed_difference']['mean'], 22/3)
            self.assertNotIn('real_seconds', result['q']['8'])
            self.assertIn('只计算双方共同成功', render(result))

    def test_all_complete_raw_forecasts_include_cropped_and_uncommitted_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ep = {'task_id': 0, 'suite': 'spatial', 'q': 5, 'status': 'success',
                  'steps': 20, 'real_seconds': 1, 'artifact_dir': 'tasks/task00',
                  'commits': [{'prediction_id': 1, 'observation_step': 0, 'commit_step': 8}]}
            self.make_run(root, [ep])
            artifact = root/'tasks/task00'
            def actions(switch):
                return [[.2, 0, 0, 0, 0, 0, -1 if i < switch else 1] for i in range(20)]
            records = [dict(row(i, action), q=5) for i, action in enumerate(actions(5))]
            (artifact/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            requests = [
                {'prediction_id': 1, 'observation_step': 0, 'actions': actions(4)},
                {'prediction_id': 2, 'observation_step': 10, 'actions': actions(5)},
                {'prediction_id': 3, 'observation_step': 10, 'actions': actions(18)},
                {'prediction_id': 4, 'observation_step': 0, 'actions': actions(4)[:5]}]
            (artifact/'q5_requests.json').write_text(json.dumps(requests))
            metadata = {'goal_state': [], 'obj_of_interest': ['bowl'], 'regions': {}}
            (artifact/'summary.json').write_text(json.dumps({'problem_metadata': metadata}))
            result = analyze(root)
            episode = result['episodes'][0]
            self.assertEqual(episode['problem_metadata'], metadata)
            self.assertEqual(len(episode['raw_prediction_event_comparisons']), 3)
            raw, actionable = result['q']['5']['raw_events_by_type'], result['q']['5']['actionable_events_by_type']
            self.assertEqual(raw['gripper_switch']['matched'], 1)
            self.assertEqual(raw['gripper_switch']['error_steps']['mean'], 1)
            self.assertEqual(raw['gripper_switch']['unmatched_predicted'], 1)
            self.assertEqual(raw['gripper_switch']['right_censored_forecast'], 1)
            self.assertEqual(actionable['gripper_switch']['matched'], 0)
            self.assertEqual(actionable['gripper_switch']['unmatched_predicted'], 0)
            self.assertIn('自我一致性偏差', render(result))


if __name__ == '__main__':
    unittest.main()
