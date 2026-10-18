"""State-prompt graph replacement contracts, without allocating a GPU model."""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, mock_open

from perceptbudget.rtc_rollout import RTCGuidanceAdapter


class ContextResizeTest(unittest.TestCase):
    def test_release_drops_graph_and_records_memory_without_changing_context(self):
        adapter = self.make_adapter()
        expert = adapter.expert
        adapter.torch.cuda.memory_allocated.side_effect = [100, 10]
        adapter.torch.cuda.memory_reserved.side_effect = [200, 20]
        with patch('builtins.open', mock_open(read_data='MemAvailable: 1024 kB\n')):
            result = adapter.release_graph()
        self.assertIsNone(adapter.graph)
        self.assertIs(adapter.expert, expert)
        self.assertEqual(result['before']['cuda_allocated_bytes'], 100)
        self.assertEqual(result['after']['cuda_allocated_bytes'], 10)
        adapter.torch.cuda.synchronize.assert_called_once()
        adapter.torch.cuda.empty_cache.assert_called_once()

    def make_adapter(self):
        adapter = RTCGuidanceAdapter.__new__(RTCGuidanceAdapter)
        adapter.attention_backend = 'flash'
        adapter.expert = MagicMock()
        adapter.expert.context_capacity = 4
        adapter.pipeline = MagicMock()
        adapter.pipe = MagicMock()
        adapter.torch = MagicMock()
        adapter.graph = object()
        adapter.horizon, adapter.num_steps = 20, 5
        adapter.reference_backend = False
        adapter._audited_prefixes = set()
        adapter._reference_audited = True
        return adapter

    def modules(self, factory):
        names = ('flash_rt', 'flash_rt.subgraphs', 'flash_rt.subgraphs.pi05',
                 'flash_rt.subgraphs.pi05.rtc_action_vjp')
        result = {name: types.ModuleType(name) for name in names}
        result[names[-1]].CapturedRtcChunk = factory
        return result

    def test_same_length_refreshes_existing_graph_buffers(self):
        adapter = self.make_adapter()
        original = adapter.graph
        result = adapter._refresh_context_for_prefix(4)
        self.assertFalse(result['context_graph_rebuilt'])
        self.assertIs(adapter.graph, original)
        adapter.expert.refresh_context.assert_called_once()
        adapter.expert.snapshot_context.assert_not_called()

    def test_old_graph_released_before_context_storage_replacement(self):
        adapter = self.make_adapter()
        replacement = object()
        factory = MagicMock(return_value=replacement)
        def snapshot(**kwargs):
            self.assertIsNone(adapter.graph)
            self.assertEqual(kwargs, {'fixed_prefix_capacity': False})
            adapter.expert.context_capacity = 6
        adapter.expert.snapshot_context.side_effect = snapshot
        with patch.dict(sys.modules, self.modules(factory)):
            result = adapter._refresh_context_for_prefix(6)
        self.assertTrue(result['context_graph_rebuilt'])
        self.assertEqual(result['context_capacity'], 6)
        self.assertIs(adapter.graph, replacement)
        self.assertFalse(adapter._reference_audited)
        factory.assert_called_once()
        adapter.torch.cuda.synchronize.assert_called_once()
        adapter.expert.refresh_context.assert_not_called()

    def test_wrong_context_capacity_is_not_silently_accepted(self):
        adapter = self.make_adapter()
        factory = MagicMock()
        with patch.dict(sys.modules, self.modules(factory)):
            with self.assertRaisesRegex(RuntimeError, 'actual state prompt length'):
                adapter._refresh_context_for_prefix(6)
        factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
