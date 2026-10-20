import json
from pathlib import Path
import tempfile
import unittest

from scripts import analyze_m1_h50 as analysis


class AnalyzeM1H50Test(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def worker(self, root, task, q, mode, outcome, *, missing=False):
        suite = 'libero_spatial'
        worker = root/f'worker-{task}-{q}-{mode}'
        cfg = dict(H=50, N=5, candidate_q=[q], rtc_suite=suite, state_enabled=True,
                   rtc_exact_flash_attention=True, rtc_fused_norm=True,
                   native_gemm_policy='autotune', diagnostic_condition=mode,
                   checkpoint_profile='lerobot_h50_quantiles_v044',
                   checkpoint_provenance_revision='b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd')
        ep = dict(task_id=task, q=q, status=outcome, steps=3, real_seconds=2.,
                  simulated_seconds=.15, start_monotonic=10., stop_monotonic=12.,
                  power=dict(total_energy_mj=20000.), drain_seconds=.5,
                  drain_power=dict(total_energy_mj=5000.),
                  including_drain_power=dict(total_energy_mj=25000.))
        if missing:
            for k in ('power', 'drain_power', 'including_drain_power', 'drain_seconds'):
                ep.pop(k)
        self.write(worker/'run.json', dict(config_snapshot=cfg))
        self.write(worker/'summary.json', dict(episode=ep))
        rows = [dict(event='control_step', step=i, action=[0.]*6+[-1.],
                     source=dict(prediction_id=0), before=dict(raw_state=[i*.01, 0., 0.]),
                     action_start_monotonic=10.+i*.1, action_end_monotonic=10.05+i*.1)
                for i in range(3)]
        commit = dict(event='prediction_committed', prediction_id=0,
                      observation_step=0, commit_step=0,
                      new_prediction_remaining_actions=50, old_queue_remaining_actions=0)
        (worker/'rtc').mkdir()
        (worker/'rtc/trace.jsonl').write_text('\n'.join(map(json.dumps, [commit, *rows]))+'\n')
        requests = [dict(prediction_id=0, observation_step=0, actions=[[0.]*6+[-1.]]*50,
                         inference_started_monotonic=10., inference_ended_monotonic=10.5)]
        self.write(worker/f'rtc/q{q}_requests.json', requests)
        return dict(cell=[suite, task, q, mode], status='completed',
                    result=dict(run=worker.name, initial_hashes=dict(sim=f'task{task}')))

    def group(self, result, q=40, mode='actual_history'):
        return next(g for g in result['groups'] if (g['q'], g['mode'], g['suite']) ==
                    (q, mode, 'libero_spatial'))

    def test_partial_index_pairing_energy_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [self.worker(root, 0, 5, 'sync_no_rtc', 'success'),
                       self.worker(root, 1, 5, 'sync_no_rtc', 'failure'),
                       self.worker(root, 0, 40, 'actual_history', 'failure'),
                       self.worker(root, 1, 40, 'actual_history', 'success', missing=True)]
            self.write(root/'index.json', dict(schema=1, status='running', cells={str(i): r for i, r in enumerate(records)}))
            result = analysis.analyze(root)
            self.assertEqual((result['status'], result['valid'], result['missing']), ('partial', 4, 296))
            self.assertFalse(result['final_delivery'])
            group = self.group(result)
            self.assertEqual(group['success_rate'], .5)
            pair = group['paired_sync_q5']
            self.assertEqual(pair['success_difference']['n'], 2)
            self.assertEqual(pair['success_difference']['mean'], 0.)
            self.assertEqual(pair['success_difference']['ci95'], [-1., 1.])
            self.assertAlmostEqual(pair['trajectory_progress_deviation_m']['mean'], 0.)
            self.assertIsNone(group['per_success_task_energy_j'])
            self.assertIsNone(group['total_attempt_drain_energy_j'])
            baseline = self.group(result, 5, 'sync_no_rtc')
            self.assertEqual(baseline['per_success_task_energy_j'], 40.)
            self.assertEqual(baseline['per_success_drain_energy_j'], 10.)
            self.assertEqual(baseline['per_success_including_drain_energy_j'], 50.)
            self.assertEqual(group['underflow_steps'], 0)
            self.assertIn('NA', analysis.markdown(result))
            json.dumps(result, allow_nan=False)

    def test_missing_frozen_backend_is_not_a_valid_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            record=self.worker(root,0,5,'sync_no_rtc','success')
            worker=root/record['result']['run']
            path=worker/'run.json'
            value=json.loads(path.read_text())
            value['config_snapshot'].pop('rtc_fused_norm')
            self.write(path,value)
            with self.assertRaisesRegex(ValueError,'missing frozen config'):
                analysis.load_cell(record,root)

    def test_busy_union_clips_task_and_preserves_gaps(self):
        requests = [dict(inference_started_monotonic=a, inference_ended_monotonic=b)
                    for a, b in [(9., 11.), (10.5, 12.), (13., 15.)]]
        result = analysis.timeline(requests, 10., 14.)
        self.assertEqual(result['busy_seconds'], 3.)
        self.assertEqual(result['duty'], .75)
        self.assertEqual(result['idle_gaps_seconds'], [1.])
        self.assertIsNone(analysis.timeline([{}], 10., 14.)['duty'])
        self.assertIsNone(analysis.timeline(None, 10., 14.)['busy_seconds'])

    def test_events_absolute_steps_and_actual_h50_buffer(self):
        actions = [[.2, 0., 0., 0., 0., 0., -1.], [.2, 0., 0., 0., 0., 0., 1.]]
        requests = [dict(prediction_id=1, observation_step=40, actions=actions)]
        rows = [dict(step=40+i, action=a) for i, a in enumerate(actions)]
        result = analysis.prediction_events(rows, requests, [dict(prediction_id=1, commit_step=40)])
        self.assertEqual(result[0]['matches'][0]['predicted_step'], 41)
        self.assertEqual(result[0]['matches'][0]['error_steps_actual_minus_predicted'], 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self.worker(root, 0, 40, 'actual_history', 'success')
            worker = root/record['result']['run']
            path = worker/'rtc/trace.jsonl'
            events = [json.loads(line) for line in path.read_text().splitlines()]
            events[0].update(observation_step=0, commit_step=12, new_prediction_remaining_actions=38)
            path.write_text('\n'.join(map(json.dumps, events))+'\n')
            ep = analysis.load_cell(record, root)
            self.assertEqual(ep['buffer'][0]['actual_elapsed_steps'], 12)
            self.assertEqual(ep['buffer'][0]['remaining_steps'], 38)
            self.assertAlmostEqual(ep['buffer'][0]['remaining_nominal_seconds'], 1.9)
            self.assertTrue(ep['buffer'][0]['recorded_remaining_matches'])

    def test_trajectory_uses_metres_and_progress_not_step_alignment(self):
        def rows(points):
            return [dict(before=dict(raw_state=p)) for p in points]
        a = rows([[0., 0., 0.], [.25, 0., 0.], [1., 0., 0.]])
        b = rows([[0., .1, 0.], [1., .1, 0.]])
        self.assertAlmostEqual(analysis.trajectory(a, b), .1)
        self.assertIsNone(analysis.trajectory([], b))

    def test_invalid_and_duplicate_cells_cannot_make_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(cell=['libero_spatial', 0, 40, 'actual_history'], status='invalid')
            self.write(root/'index.json', dict(schema=1, cells={'a': record}))
            result = analysis.analyze(root)
            self.assertEqual(result['invalid'], 1)
            self.assertIsNone(self.group(result)['success_rate'])
            self.write(root/'index.json', dict(schema=1, cells={'a': record, 'b': record}))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                analysis.analyze(root)
            record['status'] = 'running'
            self.write(root/'index.json', dict(schema=1, cells={'a': record}))
            result = analysis.analyze(root)
            self.assertEqual(result['invalid'], 0)
            self.assertEqual(result['pending'], 1)

    def test_absent_evidence_remains_na_and_identity_mismatch_excludes_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self.worker(root, 0, 40, 'actual_history', 'success')
            worker = root/record['result']['run']
            (worker/'rtc/trace.jsonl').unlink()
            self.write(worker/'rtc/q40_requests.json', [])
            item = analysis.load_cell(record, root)
            self.assertIsNone(item['events'])
            self.assertIsNone(item['buffer'])
            self.assertIsNone(item['underflow_steps'])
            self.assertIsNone(item['timeline']['busy_seconds'])
            baseline = dict(item, initial_hashes=dict(sim='other'))
            result = analysis.paired([item], {('libero_spatial', 0): baseline})
            self.assertEqual(result['success_difference']['n'], 0)
            self.assertIsNone(result['success_difference']['ci95'])
            self.assertEqual(result['excluded'][0]['reason'], 'initial hashes mismatch')

    def test_single_pair_has_no_bootstrap_interval(self):
        item = dict(suite='libero_spatial', task_id=0, outcome='success',
                    initial_hashes={'sim': 'same'}, _rows=None)
        result = analysis.paired([item], {('libero_spatial', 0): item})
        self.assertEqual(result['success_difference']['mean'], 0.)
        self.assertIsNone(result['success_difference']['ci95'])


if __name__ == '__main__':
    unittest.main()
