import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from perceptbudget.prefix_model import (
    PrefixModel, _clean_t0_styles, _temporary_t0_styles,
)


class FakeBuffer:
    def __init__(self, data):
        self.data = data.copy()
        self.nbytes = data.nbytes
        self.fail_upload_once = False

    def download(self, target):
        target[...] = self.data

    def upload(self, source):
        if self.fail_upload_once:
            self.fail_upload_once = False
            raise RuntimeError('upload failed')
        self.data[...] = source


def fake_pipeline():
    # Production token/layer/width layout, fewer diffusion steps for CPU tests.
    shapes = {'time_emb': (20, 1024), 'style_attn': (18, 20, 3072),
              'style_ffn': (18, 20, 3072), 'style_final': (20, 3072)}
    rows = {name: np.full(shape, 123, np.uint16) for name, shape in shapes.items()}
    bufs = {f'decoder_{name}': FakeBuffer(
        (np.arange(2 * np.prod(shape), dtype=np.uint32) % 65536)
        .astype(np.uint16).reshape((2,) + shape)) for name, shape in shapes.items()}
    return SimpleNamespace(chunk_size=20, num_steps=2, bufs=bufs), rows


class PrefixStylesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep test discovery from loading torch for unrelated lazy-import tests.
        global torch
        import torch

    def test_every_step_layer_prefix_only_and_exact_restore(self):
        pl, rows = fake_pipeline()
        before = {k: b.data.copy() for k, b in pl.bufs.items()}
        sync = Mock()
        # Two lengths consecutively must not inherit any earlier override.
        for length in (8, 5, 8):
            timing = {}
            with _temporary_t0_styles(pl, rows, length, sync, timing):
                for name, row in rows.items():
                    data = pl.bufs['decoder_' + name].data
                    np.testing.assert_array_equal(data[..., :length, :],
                        np.broadcast_to(row[..., :length, :], data[..., :length, :].shape))
                    np.testing.assert_array_equal(data[..., length:, :],
                                                  before['decoder_' + name][..., length:, :])
            for name, buf in pl.bufs.items():
                np.testing.assert_array_equal(buf.data, before[name])
            for key in ('backup', 'switch', 'restore', 'overhead'):
                self.assertGreaterEqual(timing[f'prefix_style_{key}_seconds'], 0)
            self.assertAlmostEqual(timing['prefix_style_overhead_seconds'], sum(
                timing[f'prefix_style_{key}_seconds'] for key in ('backup', 'switch', 'restore')))
        self.assertGreaterEqual(sync.call_count, 12)

    def test_replay_exception_restores_all_buffers(self):
        pl, rows = fake_pipeline()
        before = {k: b.data.copy() for k, b in pl.bufs.items()}
        with self.assertRaisesRegex(RuntimeError, 'replay'):
            with _temporary_t0_styles(pl, rows, 8, Mock()):
                raise RuntimeError('replay failed')
        for name, buf in pl.bufs.items():
            np.testing.assert_array_equal(buf.data, before[name])

    def test_partial_switch_failure_restores_all_buffers(self):
        pl, rows = fake_pipeline()
        before = {k: b.data.copy() for k, b in pl.bufs.items()}
        pl.bufs['decoder_style_ffn'].fail_upload_once = True
        with self.assertRaisesRegex(RuntimeError, 'upload'):
            with _temporary_t0_styles(pl, rows, 8, Mock()):
                self.fail('must not replay after failed switch')
        for name, buf in pl.bufs.items():
            np.testing.assert_array_equal(buf.data, before[name])

    def test_invalid_shape_or_buffer_size_fails_before_mutation(self):
        for invalid in ('shape', 'dtype', 'size'):
            pl, rows = fake_pipeline()
            before = {k: b.data.copy() for k, b in pl.bufs.items()}
            if invalid == 'shape':
                rows['style_final'] = rows['style_final'][:8]
            elif invalid == 'dtype':
                rows['style_final'] = rows['style_final'].astype(np.float32)
            else:
                pl.bufs['decoder_style_final'].nbytes -= 2
            with self.assertRaises(ValueError):
                with _temporary_t0_styles(pl, rows, 8, Mock()):
                    self.fail('invalid layout accepted')
            for name, buf in pl.bufs.items():
                np.testing.assert_array_equal(buf.data, before[name])

    def test_t0_sinusoid_mlp_and_layer_modulations_on_cpu(self):
        # Small, nontrivial BF16 weights exercise both MLP stages and each layer.
        gen = torch.Generator().manual_seed(17)
        rand = lambda *shape: (torch.randn(*shape, generator=gen) / 8).bfloat16()
        dim, layers, horizon = 4, 3, 20
        w = {}
        for stage in ('in', 'out'):
            w[f'decoder_time_mlp_{stage}_w'] = rand(dim, dim)
            w[f'decoder_time_mlp_{stage}_b'] = rand(dim)
        for name in ('pre_attn', 'pre_ffn'):
            w[f'decoder_{name}_norm_mod_w'] = rand(layers, dim, 3 * dim)
            w[f'decoder_{name}_norm_mod_b'] = rand(layers, 3 * dim)
        w['decoder_final_norm_mod_w'] = rand(dim, 3 * dim)
        w['decoder_final_norm_mod_b'] = rand(3 * dim)
        saved = {k: v.clone() for k, v in w.items()}
        rows = _clean_t0_styles(w, horizon)
        # Independent sinusoid construction using FlashRT's period formula.
        period = 4e-3 * (4.0 / 4e-3) ** torch.linspace(0, 1, dim // 2)
        phase = torch.tensor(0.) * (1.0 / period) * 2 * np.pi
        x = torch.cat((phase.sin(), phase.cos())).bfloat16()[None]
        for stage in ('in', 'out'):
            x = x @ w[f'decoder_time_mlp_{stage}_w'] + w[f'decoder_time_mlp_{stage}_b'][None]
            x = (x.float() * x.float().sigmoid()).bfloat16()
        x = x.expand(horizon, -1).contiguous()
        expected = {'time_emb': x}
        for kind, name in [('attn', 'pre_attn'), ('ffn', 'pre_ffn')]:
            expected['style_' + kind] = torch.stack([
                x @ w[f'decoder_{name}_norm_mod_w'][i] + w[f'decoder_{name}_norm_mod_b'][i]
                for i in range(layers)])
        expected['style_final'] = x @ w['decoder_final_norm_mod_w'] + w['decoder_final_norm_mod_b']
        for name, value in expected.items():
            np.testing.assert_array_equal(rows[name], value.view(torch.uint16).numpy())
        for name in w:
            self.assertTrue(torch.equal(w[name], saved[name]))

    def test_bootstrap_does_not_touch_styles_even_when_enabled(self):
        adapter = PrefixModel.__new__(PrefixModel)
        adapter.clean_prefix_t0 = True
        raw = torch.zeros(20, 32)
        actions = np.zeros((20, 7))
        adapter.model = SimpleNamespace(predict=Mock(return_value=actions),
                                      _pipe=SimpleNamespace(_noise_out=raw))
        with patch.dict('sys.modules', {'flash_rt.core.utils.actions':
                        SimpleNamespace(unnormalize_actions=Mock())}):
            with patch('perceptbudget.prefix_model._temporary_t0_styles') as styles:
                result, actual_raw = adapter.predict({'images': [], 'state': [1]}, 'task', {})
        self.assertIs(result, actions)
        np.testing.assert_array_equal(actual_raw, raw.numpy())
        styles.assert_not_called()
        adapter.model.predict.assert_called_once_with([], prompt='task', state=[1])

    def test_t0_rejects_variants_without_importing_flashrt(self):
        adapter = PrefixModel.__new__(PrefixModel)
        adapter.clean_prefix_t0 = True
        with self.assertRaisesRegex(ValueError, 'no variants'):
            adapter.prepare_variants([5, 8])

    def test_prepare_is_cached_and_does_not_switch_mode_or_buffers(self):
        adapter = PrefixModel.__new__(PrefixModel)
        adapter.clean_prefix_t0 = False
        adapter._t0_rows = None
        pl, rows = fake_pipeline()
        pl._decode_rtc_prefix_graph = object()
        before = {k: b.data.copy() for k, b in pl.bufs.items()}
        pipe = SimpleNamespace(pipeline=pl, _force_bf16=True,
                               _graph_torch_stream=object(), _ckpt_bf16=object())
        adapter.model = SimpleNamespace(_pipe=pipe)
        with patch('torch.cuda.stream', return_value=nullcontext()), patch(
                'perceptbudget.prefix_model._clean_t0_styles', return_value=rows) as build:
            adapter.prepare_clean_prefix_styles()
            adapter.prepare_clean_prefix_styles()
        build.assert_called_once_with(pipe._ckpt_bf16, 20)
        self.assertIs(adapter._t0_rows, rows)
        self.assertFalse(adapter.clean_prefix_t0)
        for name, buf in pl.bufs.items():
            np.testing.assert_array_equal(buf.data, before[name])

    def test_prepare_requires_warmed_pipeline(self):
        adapter = PrefixModel.__new__(PrefixModel)
        adapter.model = SimpleNamespace(_pipe=SimpleNamespace(pipeline=None))
        with self.assertRaisesRegex(RuntimeError, 'warm up'):
            adapter.prepare_clean_prefix_styles()

    def test_constructor_defaults_to_legacy_and_supports_mode_toggle(self):
        enable = Mock()
        with patch.dict('sys.modules', {'flash_rt.subgraphs.pi05.rtc_prefix':
                        SimpleNamespace(enable=enable)}):
            adapter = PrefixModel(object())
        self.assertFalse(adapter.clean_prefix_t0)
        self.assertEqual(adapter.prefix_len, 8)
        enable.assert_called_once_with(adapter.model, prefix_len=8)
        adapter.clean_prefix_t0 = True
        self.assertTrue(adapter.clean_prefix_t0)

    def test_predict_legacy_t0_legacy_on_shared_pipeline(self):
        pl, rows = fake_pipeline()
        before = {k: b.data.copy() for k, b in pl.bufs.items()}
        pl._graph_stream = object()
        pl.input_rtc_prev_action_chunk_buf = object()
        pl.input_noise_buf = SimpleNamespace(ptr=0)
        pl.input_images_buf = object()
        pl._context_graph = SimpleNamespace(replay=Mock())
        stream = SimpleNamespace(cuda_stream=0, synchronize=Mock())
        pipe = SimpleNamespace(pipeline=pl, _force_bf16=True,
            _graph_torch_stream=stream, _noise_out=torch.zeros(20, 32).bfloat16(),
            _noise_buf=torch.zeros(20, 32).bfloat16(), _img_buf=object(),
            _fill_img_buf=Mock(), _cudart=SimpleNamespace(
                cudaMemcpyAsync=Mock(), cudaStreamSynchronize=Mock()), norm_stats={})
        packed = []
        def copy(source, target, handle):
            if target is pl.input_rtc_prev_action_chunk_buf:
                packed[:] = [source]
        pipe._copy_tensor_to_pipeline_buf_stream = copy
        adapter = PrefixModel.__new__(PrefixModel)
        adapter.prefix_len, adapter._variants, adapter._t0_rows = 8, {}, rows
        adapter.model = SimpleNamespace(_pipe=pipe, set_prompt=Mock())
        def replay(handle):
            for name, row in rows.items():
                data = pl.bufs['decoder_' + name].data
                if adapter.clean_prefix_t0:
                    self.assertTrue(np.all(data[..., :8, :] == 123))
                    np.testing.assert_array_equal(data[..., 8:, :],
                                                  before['decoder_' + name][..., 8:, :])
                else:
                    np.testing.assert_array_equal(data, before['decoder_' + name])
            pipe._noise_out.copy_(packed[0])
        pl._decode_rtc_prefix_graph = SimpleNamespace(replay=replay)
        real_zeros, real_as_tensor = torch.zeros, torch.as_tensor
        def cpu_zeros(*args, **kwargs):
            kwargs.pop('device', None)
            return real_zeros(*args, **kwargs)
        def cpu_tensor(*args, **kwargs):
            kwargs.pop('device', None)
            return real_as_tensor(*args, **kwargs)
        with patch.dict('sys.modules', {'flash_rt.core.utils.actions':
                SimpleNamespace(unnormalize_actions=lambda raw, stats: raw)}), patch(
                'torch.cuda.stream', side_effect=lambda _: nullcontext()), patch(
                'torch.zeros', side_effect=cpu_zeros), patch(
                'torch.as_tensor', side_effect=cpu_tensor):
            for mode in (False, True, False):
                adapter.clean_prefix_t0 = mode
                timing = {}
                adapter.predict({'images': []}, 'task', timing, np.ones((8, 32)))
                self.assertEqual(timing['clean_prefix_t0'], mode)
                self.assertEqual('prefix_style_overhead_seconds' in timing, mode)
                for name, buf in pl.bufs.items():
                    np.testing.assert_array_equal(buf.data, before[name])


if __name__ == '__main__':
    unittest.main()
