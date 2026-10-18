"""Explicit GPU gate for variable state length under one FlashAttention graph."""
import os
import unittest


@unittest.skipUnless(os.environ.get('PB_RUN_RTC_VARLEN_TEST') == '1', 'explicit GPU gate')
class VarlenAttentionTest(unittest.TestCase):
    def test_forward_vjp_and_changed_lengths_under_one_graph(self):
        import torch
        from torch.nn import functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from perceptbudget.rtc_expert import _TorchVarlenGqaAttention
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(71)
        h, cap = 20, 39
        q = torch.randn(1, 8, h, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        ak = torch.randn(h, 1, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        av = torch.randn_like(ak, requires_grad=True)
        ck = torch.randn(cap, 1, 256, device='cuda', dtype=torch.bfloat16)
        cv = torch.randn_like(ck)
        cq = torch.tensor([0, h], device='cuda', dtype=torch.int32)
        ckv = torch.tensor([0, h+7], device='cuda', dtype=torch.int32)
        go = torch.randn_like(q)
        def run():
            y = _TorchVarlenGqaAttention.apply(q, ak, av, ck, cv, cq, ckv)
            grads = torch.autograd.grad(y, (q, ak, av), grad_outputs=go)
            return (y,)+grads
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for prefix in (7, 31, 13):
            ckv[1].fill_(h+prefix)
            graph.replay()
            actual = tuple(t.detach().clone() for t in captured)
            torch.cuda.synchronize()
            with sdpa_kernel(SDPBackend.MATH):
                k = torch.cat((ak, ck[:prefix]), 0).transpose(0, 1).unsqueeze(0)
                v = torch.cat((av, cv[:prefix]), 0).transpose(0, 1).unsqueeze(0)
                ref = F.scaled_dot_product_attention(q, k, v, scale=256**-.5, enable_gqa=True)
                grads = torch.autograd.grad(ref, (q, ak, av), grad_outputs=go)
            for left, right in zip(actual, (ref,)+grads):
                self.assertTrue(bool(torch.isfinite(left).all()))
                rel = (left.float()-right.float()).norm()/right.float().norm().clamp_min(1e-8)
                self.assertLess(float(rel), .03)
            expected = tuple(t.detach().clone() for t in captured)
            graph.replay()
            torch.cuda.synchronize()
            for left, right in zip(captured, expected):
                torch.testing.assert_close(left, right, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
