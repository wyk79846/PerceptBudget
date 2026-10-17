"""CPU contracts and explicitly opted-in CUDA tests for the upstream experiment."""

import ast
import builtins
import importlib.util
import inspect
import os
from pathlib import Path
import unittest
from unittest import mock


UPSTREAM = Path(__file__).resolve().parents[2] / 'FlashRT-upstream'
SOURCE = UPSTREAM / 'flash_rt/subgraphs/pi05/rtc_action_vjp.py'


def load_backend():
    # Load just this opt-in file, without importing model packages or CUDA code.
    spec = importlib.util.spec_from_file_location('_pb_test_rtc_action_vjp', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CapturedActionVjp


class ActionVjpImportTest(unittest.TestCase):
    def test_import_needs_no_torch_cuda_or_hooks(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'flash_rt'}:
                raise AssertionError(f'unexpected import-time dependency: {name}')
            return original_import(name, *args, **kwargs)

        with mock.patch('builtins.__import__', side_effect=guarded_import):
            cls = load_backend()
        signature = inspect.signature(cls)
        self.assertEqual(list(signature.parameters),
                         ['delta_fn', 'sample', 'step', 'num_steps', 'warmups'])
        for key, value in [('step', 0), ('num_steps', 10), ('warmups', 2)]:
            self.assertEqual(signature.parameters[key].default, value)

    def test_default_package_does_not_enable_backend(self):
        package = SOURCE.parent / '__init__.py'
        tree = ast.parse(package.read_text())
        self.assertFalse(any(isinstance(node, ast.Name)
                             and node.id == 'CapturedActionVjp'
                             for node in ast.walk(tree)))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn('rtc_action_vjp', node.module or '')
            elif isinstance(node, ast.Import):
                self.assertTrue(all('rtc_action_vjp' not in a.name for a in node.names))


class ActionVjpCpuTest(unittest.TestCase):
    def test_real_autograd_formula_and_frozen_weight_on_cpu(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != 'torch':
                raise
            self.skipTest('CPU autograd check requires torch')

        cls = load_backend()
        graph = cls.__new__(cls)
        weight = torch.tensor(.1, requires_grad=False)
        graph.x = torch.ones(4, 32, requires_grad=True)
        graph.step, graph.num_steps, graph.tau = 0, 10, 0.
        graph.delta_fn = lambda x, step: x * weight
        graph.target = torch.full_like(graph.x, 3.)
        graph.mask = torch.ones(4, 1)
        graph.beta = torch.tensor(5.)
        before = graph.x.detach().clone()
        for target, mask, beta, expected_vjp, expected_updated in [
                (3., 1., 5., 2., 2.1), (3., 1., 2., 2., 1.5),
                (3., 0., 5., 0., 1.1), (4., 1., 5., 4., 3.1)]:
            with self.subTest(target=target, mask=mask, beta=beta):
                graph.target.fill_(target)
                graph.mask.fill_(mask)
                graph.beta.fill_(beta)
                outputs = graph._run()
                for output, value in zip(outputs, (.1, 2., expected_vjp, expected_updated)):
                    self.assertEqual(output.shape, (4, 32))
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(output, torch.full_like(output, value))
                torch.testing.assert_close(graph.x, before)
                self.assertIsNone(graph.x.grad)
                self.assertIsNone(weight.grad)
                self.assertFalse(weight.requires_grad)


@unittest.skipUnless(os.environ.get('PB_RUN_RTC_GRAPH_TEST') == '1',
                     'requires explicit PB_RUN_RTC_GRAPH_TEST=1; shared GPU gate')
class ActionVjpCudaTest(unittest.TestCase):
    def test_capture_and_runtime_replays(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        cls = load_backend()
        noise = torch.ones(4, 32, device='cuda')
        original = noise.clone()
        weight = torch.tensor(.1, device='cuda', requires_grad=False)
        graph = cls(lambda x, step: x * weight, noise,
                    step=0, num_steps=10, warmups=2)
        self.assertEqual(graph.x.shape, noise.shape)
        self.assertEqual(graph.x.dtype, torch.float32)
        self.assertNotEqual(graph.x.data_ptr(), noise.data_ptr())
        self.assertEqual(graph.target.shape, noise.shape)
        self.assertEqual(graph.mask.shape, (4, 1))
        self.assertEqual(graph.beta.numel(), 1)
        snapshots = []
        for target_value, mask_value, beta, vjp, updated in [
                (3., 1., 5., 2., 2.1), (3., 1., 2., 2., 1.5),
                (3., 0., 5., 0., 1.1), (4., 1., 5., 4., 3.1)]:
            target = torch.full_like(noise, target_value)
            mask = torch.full((4, 1), mask_value, device=noise.device)
            outputs = tuple(t.detach().clone() for t in
                            graph.replay(noise, target, mask, beta=beta))
            snapshots.append((outputs, (.1, 2., vjp, updated)))
            torch.testing.assert_close(noise, original)
            torch.testing.assert_close(graph.x, original)
            torch.testing.assert_close(target, torch.full_like(target, target_value))
            torch.testing.assert_close(mask, torch.full_like(mask, mask_value))
        # Recheck saved clones after all replays to catch accidental aliasing.
        for outputs, expected in snapshots:
            self.assertEqual(len(outputs), 4)
            for output, value in zip(outputs, expected):
                self.assertEqual(output.shape, noise.shape)
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(output.device, noise.device)
                torch.testing.assert_close(output, torch.full_like(output, value))
        self.assertIsNone(weight.grad)
        self.assertFalse(weight.requires_grad)
        self.assertIsNone(graph.x.grad)

        changed_noise = noise * 2
        changed = tuple(t.detach().clone() for t in graph.replay(
            changed_noise, torch.full_like(noise, 3.), torch.ones(4, 1, device=noise.device)))
        for output, value in zip(changed, (.2, 4., -2., 1.2)):
            torch.testing.assert_close(output, torch.full_like(output, value))
        torch.testing.assert_close(changed_noise, noise*2)
        with self.assertRaises(ValueError):
            graph.replay(noise, noise, torch.ones(4, 2, device=noise.device))
        with self.assertRaises(ValueError):
            graph.replay(noise, noise, torch.ones(4, 1, device=noise.device), beta=0)


if __name__ == '__main__':
    unittest.main()
