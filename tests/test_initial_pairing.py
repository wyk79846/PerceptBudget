"""CPU-only contracts for audited initial simulator state pairing."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from perceptbudget.pairing import initial_state_mismatches, validate_initial_state_pair


def episode(q, delta=0.0):
    state = [0.0, 0.25, -0.5, 1.0, 0.0, 0.125]
    state[-1] += delta
    return {'q': q, 'initial_sim_state_sha256': str(q) * 64,
            'initial_sim_state': state, 'initial_raw_state': [0.0] * 8,
            'artifact_dir': 'tasks/task00'}


class InitialPairingTest(unittest.TestCase):
    def assert_rejected(self, baseline, candidate, run_dir=None):
        with self.assertRaises((RuntimeError, ValueError)):
            validate_initial_state_pair(baseline, candidate, run_dir=run_dir)
        self.assertTrue(initial_state_mismatches(baseline, candidate))
        self.assertFalse(candidate.get('initial_state_comparison', {}).get('accepted', False))

    def test_equal_hash_fast_path_needs_no_trace(self):
        base, cand = episode(5), episode(8)
        cand['initial_sim_state_sha256'] = base['initial_sim_state_sha256']
        for ep in (base, cand):
            del ep['initial_sim_state']
        with patch.object(Path, 'open', side_effect=AssertionError('trace read')):
            validate_initial_state_pair(base, cand)
            self.assertEqual(initial_state_mismatches(base, cand), [])

    def test_roundoff_accepted_and_evidence_preserves_original_hashes(self):
        base, cand = episode(5), episode(8, 7e-9)
        cand['initial_raw_state'][3] = 7e-9
        original_base, original_hash = deepcopy(base), cand['initial_sim_state_sha256']
        validate_initial_state_pair(base, cand)
        self.assertEqual(base, original_base)
        self.assertEqual(cand['initial_sim_state_sha256'], original_hash)
        evidence = cand['initial_state_comparison']
        self.assertTrue(evidence['accepted'])
        self.assertEqual(evidence['baseline_sha256'], base['initial_sim_state_sha256'])
        self.assertEqual(evidence['baseline_raw_state'], base['initial_raw_state'])
        self.assertAlmostEqual(evidence['max_abs_difference'], 7e-9, delta=1e-15)
        self.assertAlmostEqual(evidence['raw_max_abs_difference'], 7e-9, delta=1e-15)
        self.assertEqual(evidence['sim_time_abs_difference'], 0)
        self.assertEqual(evidence['atol'], 1e-7)
        self.assertEqual(evidence['raw_atol'], 1e-8)
        self.assertEqual(evidence['time_atol'], 1e-12)
        self.assertEqual(evidence['rtol'], 0.0)
        with patch.object(Path, 'open', side_effect=AssertionError('trace reread')):
            self.assertEqual(initial_state_mismatches(base, cand), [])
        json.dumps(cand, allow_nan=False)

    def test_large_change_in_full_state_rejected_even_with_equal_raw_state(self):
        self.assert_rejected(episode(5), episode(8, 1e-4))

    def test_large_magnitude_state_does_not_enable_relative_tolerance(self):
        base, cand = episode(5), episode(8)
        base['initial_sim_state'][-1] = 1e6
        cand['initial_sim_state'][-1] = 1e6 + 1e-4
        self.assert_rejected(base, cand)

    def test_equal_hash_does_not_bypass_raw_state_check(self):
        for difference, accepted in ((7e-9, True), (1e-4, False)):
            with self.subTest(difference=difference):
                base, cand = episode(5), episode(8)
                cand['initial_sim_state_sha256'] = base['initial_sim_state_sha256']
                cand['initial_raw_state'][0] = difference
                if accepted:
                    validate_initial_state_pair(base, cand)
                    self.assertEqual(initial_state_mismatches(base, cand), [])
                else:
                    self.assert_rejected(base, cand)

    def test_simulator_time_uses_stricter_tolerance(self):
        for difference, accepted in ((5e-13, True), (2e-12, False)):
            with self.subTest(difference=difference):
                base, cand = episode(5), episode(8)
                cand['initial_sim_state'][0] = difference
                if accepted:
                    validate_initial_state_pair(base, cand)
                    self.assertEqual(initial_state_mismatches(base, cand), [])
                else:
                    self.assert_rejected(base, cand)

    def test_missing_shape_and_nonfinite_full_state_rejected(self):
        for state in (None, [], [0.0], [[0.0] * 6],
                      [0.0, .25, -.5, 1.0, 0.0, float('nan')],
                      [0.0, .25, -.5, 1.0, 0.0, float('inf')]):
            with self.subTest(state=state):
                base, cand = episode(5), episode(8)
                if state is None:
                    del cand['initial_sim_state']
                else:
                    cand['initial_sim_state'] = state
                self.assert_rejected(base, cand)

    def test_raw_state_must_be_finite_eight_dimensions_and_within_tolerance(self):
        for raw in (None, [0.0] * 7, [0.0] * 9, [[0.0] * 8],
                    [float('nan')] + [0.0] * 7,
                    [float('inf')] + [0.0] * 7, [2e-8] + [0.0] * 7):
            with self.subTest(raw=raw):
                base, cand = episode(5), episode(8, 7e-9)
                if raw is None:
                    del cand['initial_raw_state']
                else:
                    cand['initial_raw_state'] = raw
                self.assert_rejected(base, cand)

    def test_trace_fallback_uses_first_control_step_for_each_q(self):
        base, cand = episode(5), episode(8, 7e-9)
        rows = [{'event': 'episode_start', 'q': 8},
                {'event': 'control_step', 'q': 15, 'before': {'sim_state': [999]}}]
        for ep in (base, cand):
            rows.append({'event': 'control_step', 'q': ep['q'], 'step': 0,
                         'before': {'sim_state': ep.pop('initial_sim_state')}})
            rows.append({'event': 'control_step', 'q': ep['q'], 'step': 1,
                         'before': {'sim_state': [999]}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / base['artifact_dir']
            artifact.mkdir(parents=True)
            trace = artifact / 'trace.jsonl'
            trace.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            original = trace.read_bytes()
            validate_initial_state_pair(base, cand, run_dir=root)
            self.assertEqual(trace.read_bytes(), original)
            self.assertTrue(cand['initial_state_comparison']['accepted'])

    def test_stale_baseline_hash_or_raw_state_invalidates_evidence(self):
        for field, value in (('initial_sim_state_sha256', 'stale' * 13),
                             ('initial_raw_state', [1e-4] + [0.0] * 7)):
            with self.subTest(field=field):
                base, cand = episode(5), episode(8, 7e-9)
                validate_initial_state_pair(base, cand)
                base[field] = value
                self.assertIn('initial_sim_state_sha256', initial_state_mismatches(base, cand))

    def test_evidence_cannot_override_fixed_numeric_limits(self):
        for field, value in (('max_abs_difference', 1e-4),
                             ('raw_max_abs_difference', 2e-8),
                             ('sim_time_abs_difference', 2e-12),
                             ('max_abs_difference', float('nan')),
                             ('raw_max_abs_difference', float('inf')),
                             ('max_abs_difference', -1.0), ('accepted', False)):
            with self.subTest(field=field, value=value):
                base, cand = episode(5), episode(8, 7e-9)
                validate_initial_state_pair(base, cand)
                cand['initial_state_comparison'].update(
                    atol=1.0, raw_atol=1.0, time_atol=1.0)
                cand['initial_state_comparison'][field] = value
                self.assertIn('initial_sim_state_sha256', initial_state_mismatches(base, cand))

    def test_claimed_tolerances_cannot_exceed_caps_even_for_small_deltas(self):
        for field, value in (('atol', 1.0), ('raw_atol', 1.0), ('time_atol', 1.0),
                             ('rtol', 1e-5), ('atol', float('nan')),
                             ('raw_atol', float('inf')), ('time_atol', -1.0)):
            with self.subTest(field=field, value=value):
                base, cand = episode(5), episode(8, 7e-9)
                validate_initial_state_pair(base, cand)
                cand['initial_state_comparison'][field] = value
                self.assertIn('initial_sim_state_sha256', initial_state_mismatches(base, cand))

    def test_missing_audit_does_not_accept_different_hashes(self):
        self.assertIn('initial_sim_state_sha256',
                      initial_state_mismatches(episode(5), episode(8, 7e-9)))


if __name__ == '__main__':
    unittest.main()
