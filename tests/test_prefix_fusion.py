import unittest

import numpy as np

from perceptbudget.prefix_fusion import fuse_prefix_tail, round_to_bf16


def item(value, offset, origin=0, prediction_id=0):
    return ([999.] * 7, prediction_id, origin, offset,
            np.full(32, value, dtype=np.float32))


class PrefixFusionTest(unittest.TestCase):
    def test_known_weight_raw_domain_and_latest_non_arm(self):
        old, new = item(0., 15), item(1., 7, 8, 1)
        new[4][6:] = np.arange(26) / 32
        # Deliberately insert targets out of order: only sorted overlap rank
        # determines the schedule, not target spacing or source offsets.
        targets = [30, 16, 40, 19, 23]
        results = fuse_prefix_tail(dict.fromkeys(targets, old), dict.fromkeys(targets, new))
        weights = [1., .629108, .354344, .150794, 0.]
        for index, (target, weight) in enumerate(zip(sorted(targets), weights)):
            result = results[target]
            meta = result['metadata']
            self.assertAlmostEqual(meta['weight_old'], weight, places=6)
            self.assertAlmostEqual(meta['weight_new'], 1 - weight, places=6)
            self.assertEqual(meta['method'], 'paper_eq8_9_arm6')
            self.assertEqual(meta['beta'], 1.2)
            self.assertEqual(meta['overlap_index'], index)
            self.assertEqual(meta['overlap_length'], 5)
            np.testing.assert_array_equal(result['raw'][:6], round_to_bf16([1 - weight] * 6))
            np.testing.assert_array_equal(result['raw'][6:], new[4][6:])
        np.testing.assert_array_equal(results[16]['raw'][:6], old[4][:6])
        np.testing.assert_array_equal(results[40]['raw'][:6], new[4][:6])
        meta = results[23]['metadata']
        self.assertEqual((meta['old_offset'], meta['new_offset']), (15, 7))
        self.assertEqual((meta['old_origin'], meta['new_origin']), (0, 8))
        self.assertEqual(meta['old_raw'], old[4].tolist())
        self.assertEqual(meta['new_raw'], new[4].tolist())

    def test_after_targets_only_and_no_mutation_or_history(self):
        old, new = item(0, 18), item(1, 7, 10, 2)
        before = {17: item(4, 17), 18: old, 30: item(4, 30)}
        after = {18: new, 19: item(2, 8, 10, 2)}
        result = fuse_prefix_tail(before, after)
        self.assertEqual(set(result), {18, 19})
        np.testing.assert_array_equal(old[4], np.zeros(32))
        np.testing.assert_array_equal(new[4], np.ones(32))
        self.assertEqual(result[19]['metadata']['source_count'], 1)
        self.assertIsNone(result[19]['metadata']['old_raw'])
        result[19]['raw'][0] = 8
        self.assertEqual(after[19][4][0], 2)
        self.assertEqual(fuse_prefix_tail(before, {}), {})

    def test_offset_difference_does_not_change_weights(self):
        before = {t: item(-.5, t + 20) for t in range(5)}
        after = {t: item(.5, t) for t in range(5)}
        reference = fuse_prefix_tail(before, after)
        shifted = fuse_prefix_tail({t: item(-.5, 100 - t) for t in range(5)}, after)
        for target in after:
            self.assertEqual(reference[target]['metadata']['weight_old'],
                             shifted[target]['metadata']['weight_old'])
            np.testing.assert_array_equal(reference[target]['raw'], shifted[target]['raw'])

    def test_clip_sources_before_blending_and_bf16_prefix_idempotence(self):
        before = {t: item(2, t) for t in range(5)}
        after = {t: item(0, t) for t in range(5)}
        result = fuse_prefix_tail(before, after)
        fused = result[2]['raw']
        weight = result[2]['metadata']['weight_old']
        np.testing.assert_array_equal(fused[:6], round_to_bf16([weight] * 6))
        self.assertFalse(np.array_equal(fused[:6], round_to_bf16([2 * weight] * 6)))
        negative = fuse_prefix_tail({t: item(-2, t) for t in range(5)}, after)
        np.testing.assert_array_equal(negative[2]['raw'][:6], round_to_bf16([-weight] * 6))
        clipped_new = fuse_prefix_tail(after, before)
        np.testing.assert_array_equal(clipped_new[2]['raw'][:6], round_to_bf16([1 - weight] * 6))
        np.testing.assert_array_equal(round_to_bf16(fused), fused)
        self.assertTrue(np.isfinite(fused).all())
        self.assertEqual(fused.shape, (32,))

    def test_single_overlap_returns_latest_raw_without_rounding_or_clipping(self):
        old, new = item(-2, 15), item(2.003, 7)
        result = fuse_prefix_tail({8: old}, {8: new})[8]
        np.testing.assert_array_equal(result['raw'], new[4])
        self.assertEqual(result['metadata']['weight_old'], 0.)
        self.assertEqual(result['metadata']['weight_new'], 1.)
        self.assertEqual(result['metadata']['overlap_index'], 0)
        self.assertEqual(result['metadata']['overlap_length'], 1)

    def test_late_hold_mapping_preserves_deadline_and_next_raw_prefix(self):
        from scripts.probe_delay_buffer import commit_prefix_hold

        before = {step: item(step / 32, step) for step in range(5, 30)}
        # Executed/locked targets have already been removed by the caller.
        after = {target: value for target, value in before.items() if target >= 14}
        raw = np.repeat((np.arange(20, dtype=np.float32) / 32)[:, None], 32, axis=1)
        commit_prefix_hold(after, raw[:, :7], 5, 14, 1, raw, 7)
        fused = fuse_prefix_tail(before, after)
        self.assertEqual(set(fused), set(range(14, 25)))
        for target, data in fused.items():
            latest = after[target]
            after[target] = (*latest[:4], data['raw'], data['metadata'])
            self.assertEqual(data['metadata']['new_offset'], target - 14 + 7)
            self.assertEqual(data['raw'][6], raw[target - 14 + 7, 6])
        next_prefix = np.array([after[t][4] for t in range(14, 21)])
        np.testing.assert_array_equal(round_to_bf16(next_prefix), next_prefix)
        self.assertEqual(before[13][4][0], 13 / 32)

    def test_bf16_known_ties_signed_zero_and_shapes(self):
        values = np.array([[1 + 1/256, 1 + 3/256],
                           [-1 - 1/256, -1 - 3/256]], dtype=np.float32)
        rounded = round_to_bf16(values)
        np.testing.assert_array_equal(rounded, [[1, 1 + 2/128], [-1, -1 - 2/128]])
        self.assertEqual(rounded.dtype, np.float32)
        self.assertTrue(np.signbit(round_to_bf16([-0.]))[0])
        self.assertEqual(round_to_bf16(1.).shape, ())
        self.assertEqual(round_to_bf16([]).shape, (0,))
        tiny = np.array([0x00008000, 0x00018000], dtype=np.uint32).view(np.float32)
        np.testing.assert_array_equal(round_to_bf16(tiny).view(np.uint32), [0, 0x20000])

    def test_invalid_inputs(self):
        for beta in (-1, 0, np.nan, np.inf, True, '1.2', None):
            with self.assertRaises(ValueError):
                fuse_prefix_tail({}, {}, beta=beta)
        for value in (np.nan, np.inf, -np.inf, np.finfo(np.float32).max):
            with self.assertRaises(ValueError):
                round_to_bf16([value])
        for raw in ([1.] * 31, [[1.] * 32], [np.nan] * 32):
            with self.assertRaises(ValueError):
                fuse_prefix_tail({}, {8: ([], 0, 0, 7, raw)})


if __name__ == '__main__':
    unittest.main()
