import types
import unittest


class NativeBackwardTests(unittest.TestCase):
    def setUp(self):
        global torch, NativeAda, NativeGeGLU, NativeLinear, DifferentiableExpert
        import torch
        from perceptbudget.rtc_native_probe import NativeAda, NativeGeGLU, NativeLinear
        from perceptbudget.rtc_expert import DifferentiableExpert

    def test_adaptive_norm_input_derivative(self):
        torch.manual_seed(7)
        x = torch.randn(5,32,requires_grad=True)
        style = torch.randn(5,96)
        dy = torch.randn_like(x)
        y,_ = DifferentiableExpert._ada(x,style)
        expected, = torch.autograd.grad(y,x,dy)
        actual = NativeAda.backward(types.SimpleNamespace(saved_tensors=(x,style)),dy)[0]
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)

    def test_geglu_both_input_derivatives(self):
        torch.manual_seed(8)
        g = torch.randn(5,32,requires_grad=True)
        u = torch.randn_like(g,requires_grad=True)
        dy = torch.randn_like(g)
        y = DifferentiableExpert._geglu(g,u)
        expected = torch.autograd.grad(y,(g,u),dy)
        actual = NativeGeGLU.backward(types.SimpleNamespace(saved_tensors=(g,u)),dy)
        for got,want in zip(actual,expected):
            torch.testing.assert_close(got,want,rtol=2e-5,atol=2e-6)

    def test_linear_input_gradient_only(self):
        weight = torch.randn(8,12)
        dy = torch.randn(5,12)
        dx,dw,backend = NativeLinear.backward(types.SimpleNamespace(saved_tensors=(weight,)),dy)
        torch.testing.assert_close(dx,dy @ weight.T)
        self.assertIsNone(dw)
        self.assertIsNone(backend)
