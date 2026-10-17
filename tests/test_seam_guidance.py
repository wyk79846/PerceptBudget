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
        from perceptbudget.seam_guidance import aligned_prior, seam_sample, vls_step
        cls.aligned_prior = staticmethod(aligned_prior)
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

    def test_vls_equation_and_guided_window(self):
        torch = self.torch
        candidate = torch.zeros((3, 2), dtype=torch.bfloat16)
        prior = torch.ones_like(candidate)
        result = self.vls_step(candidate, prior, step=0, num_steps=10,
                               strength=.1, guided_window=2)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertAlmostEqual(float(result[0, 0]), .002, delta=.0001)
        self.assertAlmostEqual(float(result[1, 1]), .002, delta=.0001)
        self.assertEqual(float(result[2, 0]), 0.)

    def test_seam_keeps_bf16_rounding_between_denoising_steps(self):
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
            max_window=2)
        self.assertEqual(seen_dtypes, [torch.bfloat16, torch.bfloat16])
        self.assertEqual(output.dtype, torch.bfloat16)
        expected = torch.zeros_like(output)
        expected[0].fill_(.24)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(record['guided_window'], 1)
        self.assertFalse(output.requires_grad)

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
