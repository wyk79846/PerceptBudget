import unittest

from perceptbudget.rtc_guidance import guidance_scale, soft_mask


class RTCGuidanceTest(unittest.TestCase):
    def test_paper_mask_three_regions(self):
        w = soft_mask(20, 7, 8)
        self.assertEqual(w[:7], [1.]*7)
        self.assertEqual(w[12:], [0.]*8)
        self.assertTrue(all(0 < x < 1 for x in w[7:12]))
        self.assertTrue(all(a > b for a, b in zip(w[6:12], w[7:13])))

    def test_bad_latency_is_not_silently_clamped(self):
        for d, s in [(11, 11), (7, 6), (7, 14)]:
            with self.assertRaises(ValueError):
                soft_mask(20, d, s)

    def test_scale_finite_at_zero(self):
        self.assertEqual(guidance_scale(0), 5.)
        self.assertEqual(guidance_scale(.5), 2.)

    def test_real_vjp_does_not_update_model_parameters(self):
        import torch
        from perceptbudget.rtc_guidance import guided_sample
        weight = torch.tensor(.1, requires_grad=False)
        noise = torch.ones(20, 32)
        previous = torch.zeros(12, 32)
        out, records = guided_sample(lambda x, k: x*weight/10, noise, previous,
                                     delay=7, executed=8)
        self.assertEqual(len(records), 10)
        self.assertGreater(records[0]['vjp_norm'], 0)
        self.assertIsNone(weight.grad)
        self.assertFalse(out.requires_grad)
        self.assertTrue(torch.isfinite(out).all())
