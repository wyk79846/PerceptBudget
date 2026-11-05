"""Project-side selective token pipeline with GPU redundancy selection."""


def pipeline_type(base_type, *, visual_tokens=416):
    if visual_tokens not in (320,416,464):raise ValueError('Selective pooling supports320,416 or464 tokens')
    class SelectivePipeline(base_type):
        def _allocate_buffers(self):
            if self.num_views!=2 or self.vision_pool_factor!=2:
                raise ValueError('Selective pooling requires dual views and pool storage')
            self.vision_seq_enc=visual_tokens
            self.encoder_seq_len=visual_tokens+self.max_prompt_len
            self.total_kv=self.encoder_seq_len+self.chunk_size
            return super()._allocate_buffers()

        def _prepare_selector(self):
            if hasattr(self,'_selector_lib'):return
            import torch
            from perceptbudget.selective_pool_cuda import load
            self._selector_lib=load()
            cameras=1 if visual_tokens==416 else 2
            self._selector_scores=torch.empty(64*cameras,device='cuda',dtype=torch.float32)
            self._selector_mask=torch.empty(64*cameras,device='cuda',dtype=torch.int32)
            self._selector_mapping=torch.empty(visual_tokens if cameras==2 else 160,device='cuda',dtype=torch.int32)

        def record_infer_graph(self,external_stream_int=None):
            self._prepare_selector()
            return super().record_infer_graph(external_stream_int=external_stream_int)

        def vision_encoder(self,stream=0):
            self._prepare_selector()
            # Native uniform pooling is overwritten; selector reads full,
            # fresh27-layer SigLIP features.416 preserves wrist;320/464 select both.
            super().vision_encoder(stream=stream)
            function=getattr(self._selector_lib,'selective_pool'+str(visual_tokens))
            code=function(self.bufs['vision_x'].ptr.value,
                self._selector_scores.data_ptr(),self._selector_mask.data_ptr(),
                self._selector_mapping.data_ptr(),self.bufs['vision_x_pooled'].ptr.value,stream)
            if code:raise RuntimeError('Selective pooling CUDA launch failed:'+str(code))
    return SelectivePipeline
