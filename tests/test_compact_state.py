import types
import unittest
from perceptbudget.compact_state import frontend_type


class FakeBase:
    def __init__(self):
        self._fixed_pipeline=None
        self.pipeline=None
        self.gemm=self.fvk=self.attn_backend=None
        self.num_views=2
        self.chunk_size=20
        self._num_steps=5
        self._vision_pool_factor=1
        self._vision_num_layers=27
        self.grow_calls=[]

    def _ensure_prompt_capacity(self,n): self.grow_calls.append(n)
    def _build_pipeline_weights(self): return {'frozen':True}
    def _pipeline_precision_kwargs(self): return {}


class CompactStateTests(unittest.TestCase):
    def test_stateless_path_preserves_custom_pipeline_and_exact_capacity(self):
        f=frontend_type(FakeBase,200,pipeline_type=lambda **kw:types.SimpleNamespace(**kw),exact_no_state=True)()
        f._set_prompt_per_length(None,19)
        pipeline=f.pipeline
        self.assertEqual(pipeline.max_prompt_len,19)
        self.assertTrue(pipeline.fixed_shape)
        f._set_prompt_per_length(None,19)
        self.assertIs(f.pipeline,pipeline)
        with self.assertRaisesRegex(ValueError,'prompt changed'):
            f._set_prompt_per_length(None,20)
        with self.assertRaisesRegex(ValueError,'received state'):
            f._set_prompt_per_length([1],19)

    def make(self):
        return frontend_type(FakeBase,64,pipeline_type=lambda **kw:types.SimpleNamespace(**kw))()

    def test_full_valid_prompt_and_reuse(self):
        f=self.make()
        f._set_prompt_fixed(58)
        p=f.pipeline
        self.assertEqual(p.max_prompt_len,64)
        self.assertEqual(p.vision_num_layers,27)
        self.assertEqual(p.vision_pool_factor,1)
        self.assertEqual(f.current_prompt_len,58)
        f._set_prompt_fixed(64)
        self.assertIs(f.pipeline,p)
        self.assertEqual(f.current_prompt_len,64)

    def test_overflow_errors_before_any_allocation(self):
        f=self.make()
        with self.assertRaisesRegex(ValueError,'no truncation'): f._set_prompt_fixed(65)
        self.assertIsNone(f.pipeline)
        self.assertEqual(f.grow_calls,[])

    def test_invalid_capacity(self):
        for value in (0,201,True,1.5):
            with self.assertRaises(ValueError): frontend_type(FakeBase,value)

    def test_eighty_capacity_preserves_full_prompt_and_rejects_overflow(self):
        f=frontend_type(FakeBase,80,pipeline_type=lambda **kw:types.SimpleNamespace(**kw))()
        f._set_prompt_fixed(61)
        pipeline=f.pipeline
        self.assertEqual(pipeline.max_prompt_len,80)
        f._set_prompt_fixed(80)
        self.assertIs(f.pipeline,pipeline)
        self.assertEqual(f.current_prompt_len,80)
        with self.assertRaisesRegex(ValueError,'no truncation'): f._set_prompt_fixed(81)
        self.assertIs(f.pipeline,pipeline)
        self.assertEqual(f.current_prompt_len,80)
