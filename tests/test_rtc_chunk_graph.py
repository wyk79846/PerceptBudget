"""Full-chunk guided generation contracts; CUDA requires explicit opt-in.

These tests cover frozen-context generation, not asynchronous RTC execution.
"""

import importlib.util
import inspect
import math
import os
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[2] / 'FlashRT-upstream'
          / 'flash_rt/subgraphs/pi05/rtc_action_vjp.py')
N = 10


def load_backend():
    spec = importlib.util.spec_from_file_location('_pb_test_rtc_chunk', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coefficients(step):
    # Distinct styles make accidentally using step zero ten times observable.
    return .015 + .002 * step, -.025 + .003 * step


def scalar_reference(noise, target, mask, beta, *, repeat_step_zero=False):
    """Independent Python-float recurrence, with analytic linear Jacobian.

    No backend helper, torch autograd, or project guidance implementation is
    used to compute the expected values.
    """
    state = float(noise)
    norms, errors = [], []
    for k in range(N):
        step = 0 if repeat_step_zero else k
        tau = step / N
        a, b = coefficients(step)
        delta = a * state + b
        jacobian = 1 + (1 - tau) * N * a
        estimate = state + (1 - tau) * N * delta
        error = (target - estimate) * mask
        vjp = jacobian * error
        if tau == 0:
            scale = beta
        else:
            variance = (1 - tau)**2 / (tau**2 + (1 - tau)**2)
            scale = min(beta, (1 - tau) / (tau * variance))
        state += delta + scale / N * vjp
        norms.append(abs(vjp))
        errors.append(abs(error))
    return state, norms, errors


class ChunkContractTest(unittest.TestCase):
    def test_constructor_and_inherited_runtime_ports(self):
        backend = load_backend()
        cls = backend.CapturedRtcChunk
        self.assertTrue(issubclass(cls, backend.CapturedActionVjp))
        self.assertIs(cls.replay, backend.CapturedActionVjp.replay)
        self.assertEqual(inspect.signature(cls),
                         inspect.signature(backend.CapturedActionVjp))
        # Reject before touching CUDA, even when no sample is supplied.
        with self.assertRaises(ValueError):
            cls(None, None, step=1)


class ChunkCpuTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != 'torch':
                raise
            self.skipTest('CPU autograd check requires torch')
        self.torch = torch
        self.cls = load_backend().CapturedRtcChunk

    def make_chunk(self, noise, target, mask, beta):
        torch = self.torch
        chunk = self.cls.__new__(self.cls)
        chunk.step, chunk.num_steps, chunk.tau = 0, N, 0.
        chunk.x = noise.detach().clone().requires_grad_(True)
        chunk.target = target.detach().clone()
        chunk.mask = mask.detach().clone()
        chunk.beta = torch.tensor(beta)
        weight = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        calls = []

        def delta(x, step):
            calls.append(step)
            self.assertTrue(x.is_leaf)
            self.assertTrue(x.requires_grad)
            self.assertIsNone(x.grad_fn)
            a, b = coefficients(step)
            return weight * (a * x + b)

        chunk.delta_fn = delta
        return chunk, weight, calls

    def assert_outputs(self, outputs, expected):
        torch = self.torch
        self.assertEqual(len(outputs), 3)
        self.assertEqual([tuple(t.shape) for t in outputs], [(20, 32), (N,), (N,)])
        for actual, wanted in zip(outputs, expected):
            self.assertEqual(actual.dtype, torch.float32)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)

    def test_ten_distinct_steps_against_scalar_recurrence(self):
        torch = self.torch
        for noise, target, mask, beta in [
                (.7, 1.3, 1., 5.), (-.4, .8, .35, 2.),
                (.7, -1.2, .7, 12.), (1.2, 3., 0., 5.)]:
            with self.subTest(noise=noise, target=target, mask=mask, beta=beta):
                chunk, weight, calls = self.make_chunk(
                    torch.full((20, 32), noise), torch.full((20, 32), target),
                    torch.full((20, 1), mask), beta)
                before = tuple(t.detach().clone() for t in
                               (chunk.x, chunk.target, chunk.mask, chunk.beta, weight))
                raw, norms, errors = scalar_reference(noise, target, mask, beta)
                expected = (torch.full((20, 32), raw),
                            torch.tensor(norms) * math.sqrt(640),
                            torch.tensor(errors) * math.sqrt(640))
                for _ in range(2):
                    self.assert_outputs(chunk._run(), expected)
                self.assertEqual(calls, list(range(N)) * 2)
                for value, saved in zip(
                        (chunk.x, chunk.target, chunk.mask, chunk.beta, weight), before):
                    torch.testing.assert_close(value, saved, rtol=0, atol=0)
                    self.assertIsNone(value.grad)
                self.assertFalse(weight.requires_grad)
                wrong = scalar_reference(noise, target, mask, beta,
                                         repeat_step_zero=True)[0]
                self.assertGreater(abs(raw - wrong), 1e-4)

    def test_matches_guided_sample_with_fixed_d7_s8_mask(self):
        from perceptbudget.rtc_guidance import guided_sample, soft_mask

        torch = self.torch
        noise = torch.linspace(-.8, .9, 640).reshape(20, 32)
        previous = torch.linspace(.9, -.4, 384).reshape(12, 32)
        target = torch.zeros_like(noise)
        target[:12] = previous
        mask = torch.tensor(soft_mask(20, 7, 8))[:, None]
        self.assertTrue(bool(((mask > 0) & (mask < 1)).any()))
        for beta in (2., 5., 12.):
            chunk, weight, calls = self.make_chunk(noise, target, mask, beta)
            # Use a separate callable so the leaf/call checks only see _run.
            def delta(x, step):
                a, b = coefficients(step)
                return a * x + b

            raw, diagnostics = guided_sample(delta, noise, previous, delay=7,
                                             executed=8, num_steps=N, beta=beta)
            self.assert_outputs(chunk._run(), (
                raw, torch.tensor([d['vjp_norm'] for d in diagnostics]),
                torch.tensor([d['weighted_error_norm'] for d in diagnostics])))
            self.assertEqual(calls, list(range(N)))
            self.assertIsNone(weight.grad)

    def test_zero_mask_is_unguided_ten_step_generation(self):
        torch = self.torch
        noise = torch.linspace(-1., 1., 640).reshape(20, 32)
        chunk, weight, _ = self.make_chunk(
            noise, noise + 9, torch.zeros(20, 1), 5.)
        expected = noise.clone()
        for k in range(N):
            a, b = coefficients(k)
            expected = expected + a * expected + b
        self.assert_outputs(chunk._run(), (expected, torch.zeros(N), torch.zeros(N)))
        self.assertIsNone(weight.grad)


@unittest.skipUnless(os.environ.get('PB_RUN_RTC_GRAPH_TEST') == '1',
                     'requires explicit PB_RUN_RTC_GRAPH_TEST=1; shared GPU gate')
class ChunkCudaTest(unittest.TestCase):
    def test_capture_dynamic_inputs_and_multiple_replays(self):
        import torch
        from perceptbudget.rtc_guidance import guided_sample, soft_mask

        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        cls = load_backend().CapturedRtcChunk
        noise = torch.linspace(-.8, .9, 640, device='cuda').reshape(20, 32)
        weight = torch.nn.Parameter(torch.tensor(1., device='cuda'), requires_grad=False)

        def delta(x, step):
            a, b = coefficients(step)
            return weight * (a * x + b)

        chunk = cls(delta, noise, num_steps=N, warmups=2)
        self.assertNotEqual(chunk.x.data_ptr(), noise.data_ptr())
        previous = torch.linspace(.9, -.4, 384, device='cuda').reshape(12, 32)
        target = torch.zeros_like(noise)
        target[:12] = previous
        mask = torch.tensor(soft_mask(20, 7, 8), device='cuda')[:, None]
        cases = [(noise, target, mask, 5.),
                 (noise + .3, target, mask, 5.),
                 (noise, target + .4, mask, 5.),
                 (noise, target, mask * .4, 5.),
                 (noise, target, mask, 2.),
                 (noise, target + 9, torch.zeros_like(mask), 12.),
                 (noise - .2, target - 3, torch.zeros_like(mask), 2.),
                 (noise, target, mask, 5.)]
        snapshots = []
        for sample, goal, soft, beta in cases:
            before = [v.clone() for v in (sample, goal, soft, weight)]
            sample_cpu, goal_cpu, soft_cpu = [v.cpu().tolist()
                                             for v in (sample, goal, soft)]
            expected = [scalar_reference(sample_cpu[h][d], goal_cpu[h][d],
                                         soft_cpu[h][0], beta)
                        for h in range(20) for d in range(32)]
            raw = torch.tensor([v[0] for v in expected], device='cuda').reshape(20, 32)
            norms = torch.tensor([v[1] for v in expected], device='cuda').norm(dim=0)
            errors = torch.tensor([v[2] for v in expected], device='cuda').norm(dim=0)
            wanted = (raw, norms, errors)
            outputs = tuple(v.clone() for v in chunk.replay(sample, goal, soft, beta=beta))
            self.assertEqual(len(outputs), 3)
            self.assertEqual([tuple(v.shape) for v in outputs],
                             [(20, 32), (N,), (N,)])
            for actual, reference in zip(outputs, wanted):
                self.assertEqual(actual.device, sample.device)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
            for value, saved in zip((sample, goal, soft, weight), before):
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
            torch.testing.assert_close(chunk.x, sample, rtol=0, atol=0)
            if not bool(soft.any()):
                unguided = sample.clone()
                with torch.no_grad():
                    for step in range(N):
                        unguided = unguided + delta(unguided, step)
                torch.testing.assert_close(outputs[0], unguided)
                torch.testing.assert_close(outputs[1], torch.zeros_like(norms))
                torch.testing.assert_close(outputs[2], torch.zeros_like(errors))
            snapshots.append((outputs, wanted))
        for outputs, wanted in snapshots:
            for actual, reference in zip(outputs, wanted):
                torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
        baseline, diagnostics = guided_sample(delta, noise, previous, delay=7,
                                              executed=8, num_steps=N, beta=5.)
        torch.testing.assert_close(snapshots[0][0][0], baseline)
        torch.testing.assert_close(snapshots[0][0][1], torch.tensor(
            [d['vjp_norm'] for d in diagnostics], device='cuda'))
        torch.testing.assert_close(snapshots[0][0][2], torch.tensor(
            [d['weighted_error_norm'] for d in diagnostics], device='cuda'))
        for index in range(1, 5):
            self.assertFalse(torch.allclose(snapshots[0][0][0], snapshots[index][0][0]))
        torch.testing.assert_close(snapshots[0][0][0], snapshots[-1][0][0])
        self.assertIsNone(weight.grad)
        self.assertFalse(weight.requires_grad)
        self.assertIsNone(chunk.x.grad)


if __name__ == '__main__':
    unittest.main()
