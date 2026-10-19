"""Optional smaller fixed-state allocation, never shorter tokenization.

The upstream tokenizer stays at its original 200-token limit. Only the
physical fixed pipeline capacity changes. Every prompt is checked before
allocation/upload; an overflow is an error, never silent truncation.
"""


def frontend_type(base_type,capacity,*,pipeline_type=None):
    if not isinstance(capacity,int) or isinstance(capacity,bool) or not 1<=capacity<=200:
        raise ValueError('state prompt capacity must be an integer in [1,200]')

    class CompactStateFrontend(base_type):
        def _set_prompt_fixed(self,prompt_len):
            if prompt_len>capacity:
                raise ValueError(f'full state prompt has {prompt_len} tokens, compact capacity {capacity}; no truncation allowed')
            self._ensure_prompt_capacity(capacity)
            if self._fixed_pipeline is None:
                kind=pipeline_type
                if kind is None:
                    from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
                    kind=Pi05Pipeline
                self._fixed_pipeline=kind(
                    gemm=self.gemm,fvk=self.fvk,attn_backend=self.attn_backend,
                    weights=self._build_pipeline_weights(),num_views=self.num_views,
                    max_prompt_len=capacity,chunk_size=self.chunk_size,num_steps=self._num_steps,
                    vision_pool_factor=self._vision_pool_factor,
                    vision_num_layers=self._vision_num_layers,fixed_shape=True,
                    **self._pipeline_precision_kwargs())
            if self.pipeline is not self._fixed_pipeline:
                self.pipeline=self._fixed_pipeline
                self.graph_recorded=getattr(self.pipeline,'_graph',None) is not None
                self.calibrated=self.graph_recorded or bool(getattr(self.pipeline,'fp8_calibrated',False))
            self.current_prompt_len=prompt_len

    return CompactStateFrontend
