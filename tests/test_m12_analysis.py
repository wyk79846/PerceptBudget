import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_m12 import aggregate, analyze, episode

ASYNC_RUN = Path(os.environ['M12_ASYNC_RUN']) if os.environ.get('M12_ASYNC_RUN') else None


class M12AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ep = dict(task_id=0, suite='spatial', q=5, status='success',
                       start_monotonic=0, stop_monotonic=1, real_seconds=1,
                       inference_busy_seconds=.8, underflow_steps=[], artifact_dir='.',
                       power={'raw_readings': [dict(timestamp_monotonic=i/10, power_mw=2000)
                                               for i in range(11)]})

    def requests(self, intervals):
        (self.root/'q5_requests.json').write_text(json.dumps([
            dict(inference_started_monotonic=a, inference_ended_monotonic=b) for a, b in intervals]))

    def test_clip_union_deduplicate_and_energy(self):
        self.requests([(-1, .3), (.2, .6), (.2, .6), (.8, 2)])
        e = episode(self.ep, self.root)
        self.assertEqual(e['busy_intervals'], [[0., .6], [.8, 1.]])
        self.assertAlmostEqual(e['d'], .8)
        self.assertAlmostEqual(e['energy_j'], 2)
        self.assertAlmostEqual(e['busy_power']['total_energy_j'], 1.6)
        self.assertAlmostEqual(e['idle_power']['total_energy_j'], .4)
        row = aggregate(5, [e, dict(e, source_status='timeout')])
        self.assertEqual(row['successes'], 1)
        self.assertAlmostEqual(row['energy_per_success_j'], 4)

    def test_short_gap_is_preserved(self):
        self.requests([(0, .499), (.501, 1)])
        e = episode(self.ep, self.root)
        self.assertAlmostEqual(e['idle_seconds'], .002)
        self.assertAlmostEqual(e['idle_power']['total_energy_j'], .004)

    def test_missing_requests_and_power_are_null(self):
        e = episode(self.ep, self.root)
        self.assertIsNone(e['d'])
        self.assertIsNone(e['busy_power'])
        self.assertAlmostEqual(e['energy_j'], 2)
        self.assertTrue(e['warnings'])
        self.requests([(0, .5)])
        self.ep.pop('power')
        e = episode(self.ep, self.root)
        self.assertIsNone(e['energy_j'])
        self.assertIsNone(e['P_on_w'])
        self.assertIsNone(aggregate(5, [e])['energy_per_success_j'])

    def test_missing_request_end_is_not_idle(self):
        (self.root/'q5_requests.json').write_text('[{"inference_started_monotonic": 0}]')
        e = episode(self.ep, self.root)
        self.assertIsNone(e['idle_power'])
        self.assertIsNone(e['S'])

    def test_calibration_is_independent_reference(self):
        self.requests([(0, .5)])
        source = self.root/'source'
        source.mkdir()
        (source/'summary.json').write_text(json.dumps(dict(
            execution_mode='asynchronous_q5_paired', config={}, episodes=[dict(
                self.ep, artifact_dir=str(self.root))])))
        calibration = self.root/'calibration'
        calibration.mkdir()
        (calibration/'summary.json').write_text(json.dumps(dict(
            status='completed', P_on_w=10, P_idle_w=2, r=.2)))
        output = analyze(source, self.root/'derived', calibration)
        result = json.loads((output/'analysis.json').read_text())
        row = result['by_q'][0]
        self.assertAlmostEqual(row['P_on_w'], 2)
        self.assertAlmostEqual(row['S'], 0)
        self.assertAlmostEqual(row['S_longwindow'], .4)
        self.assertEqual(result['calibration']['summary']['r'], .2)

    @unittest.skipUnless(ASYNC_RUN and (ASYNC_RUN/'summary.json').is_file(),
                         '设置M12_ASYNC_RUN以启用只读真实日志集成测试')
    def test_actual_async_input_independent_output_and_no_overwrite(self):
        summary_path = ASYNC_RUN/'summary.json'
        summary = json.loads(summary_path.read_text())
        paths = [summary_path, ASYNC_RUN/'analysis.json', *[
            Path(e['artifact_dir'])/f"q{e['q']}_requests.json" for e in summary['episodes']]]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        output = analyze(ASYNC_RUN, self.root/'derived')
        result = json.loads((output/'analysis.json').read_text())
        self.assertNotEqual(output, ASYNC_RUN)
        self.assertEqual(len(result['episodes']), 150)
        self.assertEqual([r['successes'] for r in result['by_q']], [20, 25, 27, 28, 24])
        self.assertEqual(sum(r['episodes'] for r in result['by_q']), 150)
        self.assertGreater(result['by_q'][-1]['underflow_steps'], 0)
        for row in result['by_q']:
            self.assertAlmostEqual(row['energy_j'], row['busy_energy_j']+row['idle_energy_j'])
            self.assertAlmostEqual(row['energy_per_success_j'], row['energy_j']/row['successes'])
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == '__main__':
    unittest.main()
