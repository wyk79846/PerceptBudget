"""Opt-in GPU gates for the experimental input-only CUDA VJP kernels."""
import os
import unittest
import types


@unittest.skipUnless(os.environ.get('PB_RUN_RTC_FUSED_TEST')=='1','explicit GPU test only')
class FusedBackwardTests(unittest.TestCase):
    def setUp(self):
        import torch
        from perceptbudget.rtc_fused_backward import load
        from perceptbudget.experiment import load_config
        self.torch=torch
        load(load_config()['output_root'])

    def assert_relative(self, got, expected, tolerance=0.005):
        t=self.torch
        self.assertTrue(bool(t.isfinite(got).all()))
        error=(got.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)
        self.assertLess(float(error),tolerance)

    def test_ada_small_normal_large_inputs(self):
        from perceptbudget.rtc_fused_backward import ada_backward
        from perceptbudget.rtc_native_probe import NativeAda
        t=self.torch
        t.manual_seed(7)
        for scale in (0.,0.01,1.,10.):
            x=(scale*t.randn(20,1024,device='cuda')).bfloat16()
            style=t.randn(20,3072,device='cuda',dtype=t.bfloat16)
            dy=t.randn_like(x)
            expected=NativeAda.backward(types.SimpleNamespace(saved_tensors=(x,style)),dy)[0]
            self.assert_relative(ada_backward(x,style,dy),expected)

    def test_geglu_extremes_and_zero_seed(self):
        from perceptbudget.rtc_fused_backward import geglu_backward
        from perceptbudget.rtc_native_probe import NativeGeGLU
        t=self.torch
        t.manual_seed(8)
        g=(10*t.randn(20,4096,device='cuda')).bfloat16()
        u=t.randn_like(g)
        dy=t.randn_like(g)
        expected=NativeGeGLU.backward(types.SimpleNamespace(saved_tensors=(g,u)),dy)
        for got,want in zip(geglu_backward(g,u,dy),expected):
            self.assert_relative(got,want)
        for got in geglu_backward(g,u,t.zeros_like(dy)):
            self.assertEqual(float(got.abs().max()),0.)

    def test_captured_vjp_uses_updated_inputs(self):
        from perceptbudget.rtc_fused_backward import geglu_backward
        from perceptbudget.rtc_native_probe import NativeGeGLU
        t=self.torch
        g=t.randn(20,4096,device='cuda',dtype=t.bfloat16)
        u=t.randn_like(g)
        dy=t.randn_like(g)
        side=t.cuda.Stream()
        side.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(side):
            for _ in range(3): geglu_backward(g,u,dy)
        t.cuda.current_stream().wait_stream(side)
        graph=t.cuda.CUDAGraph()
        with t.cuda.graph(graph,stream=side):
            out=geglu_backward(g,u,dy)
        graph.replay()
        before=out[0].clone()
        g.add_(0.5)
        dy.mul_(0.3)
        graph.replay()
        expected=NativeGeGLU.backward(types.SimpleNamespace(saved_tensors=(g,u)),dy)
        for got,want in zip(out,expected): self.assert_relative(got,want)
        self.assertGreater(float((before-out[0]).abs().max()),0.)
