import unittest


class SeamGuidanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != 'torch':
                raise
            raise unittest.SkipTest('SEAM tensor tests require torch')
        cls.torch = torch
        from perceptbudget.seam_guidance import (
            aligned_prior,
            clip_prior_tail_to_executed_actions,
            seam_sample,
            vls_step,
        )
        cls.aligned_prior = staticmethod(aligned_prior)
        cls.clip_prior_tail_to_executed_actions = staticmethod(
            clip_prior_tail_to_executed_actions)
        cls.seam_sample = staticmethod(seam_sample)
        cls.vls_step = staticmethod(vls_step)

    def test_prior_pads_with_last_old_action_without_mutating_source(self):
        torch = self.torch
        tail = torch.tensor([[1., 2.], [3., 4.]])
        original = tail.clone()
        prior = self.aligned_prior(tail, 4)
        torch.testing.assert_close(prior, torch.tensor(
            [[1., 2.], [3., 4.], [3., 4.], [3., 4.]]))
        torch.testing.assert_close(tail, original)

    def test_prior_uses_actions_after_execution_clip(self):
        import numpy as np

        raw = np.array([[1.4, -1.2, 0.25, 9.0]], dtype=np.float32)
        clipped, stats = self.clip_prior_tail_to_executed_actions(raw, 2)
        np.testing.assert_array_equal(
            clipped, np.array([[1.0, -1.0, 0.25, 9.0]], dtype=np.float32))
        self.assertAlmostEqual(stats['raw_out_of_range_fraction'], 1.0)
        self.assertAlmostEqual(stats['raw_max_abs'], 1.4)
        self.assertGreater(stats['clip_delta_rms'], 0.0)
        np.testing.assert_allclose(raw, [[1.4, -1.2, 0.25, 9.0]], rtol=0, atol=1e-7)

    def test_vls_equation_and_guided_window(self):
        torch = self.torch
        candidate = torch.zeros((3, 2), dtype=torch.bfloat16)
        prior = torch.ones_like(candidate)
        result = self.vls_step(candidate, prior, step=0, num_steps=10,
                               strength=.1, guided_window=2,
                               physical_action_dim=1)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertAlmostEqual(float(result[0, 0]), .002, delta=.0001)
        self.assertEqual(float(result[1, 1]), 0.)
        self.assertEqual(float(result[2, 0]), 0.)
        self.assertEqual(float(result[0, 1]), 0.)

    def test_seam_uses_bf16_model_inputs_and_fp32_solver_state(self):
        torch = self.torch
        noise = torch.zeros((2, 32), dtype=torch.bfloat16)
        tail = torch.ones((1, 32), dtype=torch.bfloat16)
        seen_dtypes = []

        def zero_velocity(state, step):
            seen_dtypes.append(state.dtype)
            self.assertFalse(state.requires_grad)
            return torch.zeros_like(state)

        output, record = self.seam_sample(
            zero_velocity, noise, tail, num_steps=2, strength=.1,
            max_window=2, physical_action_dim=7)
        self.assertEqual(seen_dtypes, [torch.bfloat16, torch.bfloat16])
        self.assertEqual(output.dtype, torch.bfloat16)
        expected = torch.zeros_like(output)
        expected[0, :7].fill_(.24)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(record['guided_window'], 1)
        self.assertEqual(record['physical_action_dim'], 7)
        self.assertTrue(torch.equal(output[:, 7:], noise[:, 7:]))
        self.assertFalse(output.requires_grad)

    def test_fp32_solver_accumulates_updates_smaller_than_bf16_ulp(self):
        torch = self.torch
        noise = torch.zeros((2, 32), dtype=torch.bfloat16)
        noise[0, 0] = 1.0
        tail = torch.zeros((1, 32), dtype=torch.bfloat16)
        deltas = []

        def small_velocity_step(state, step):
            self.assertEqual(state.dtype, torch.bfloat16)
            delta = torch.full_like(state, 0.001)
            deltas.append(float(delta[0, 0]))
            return delta

        output, _ = self.seam_sample(
            small_velocity_step, noise, tail, num_steps=10,
            strength=.1, max_window=0, physical_action_dim=7)
        expected = torch.tensor(
            1.0 + sum(deltas), dtype=torch.float32).to(torch.bfloat16)
        self.assertEqual(output[0, 0], expected)
        self.assertGreater(float(output[0, 0]), 1.0)

    def test_bad_tail_and_window_are_rejected(self):
        torch = self.torch
        noise = torch.zeros((2, 32))
        with self.assertRaises(ValueError):
            self.seam_sample(lambda x, _: torch.zeros_like(x), noise,
                             torch.zeros((0, 32)))
        with self.assertRaises(ValueError):
            self.vls_step(torch.zeros((2, 32)), torch.zeros((2, 32)),
                          step=0, num_steps=1, guided_window=3)


if __name__ == '__main__':
    unittest.main()
