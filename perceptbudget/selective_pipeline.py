"""Project-side416-token pipeline using native FlashRT and GPU redundancy selection."""


def pipeline_type(base_type):
    class SelectivePipeline(base_type):
        def _allocate_buffers(self):
            if self.num_views!=2 or self.vision_pool_factor!=2:
                raise ValueError('Selective416 requires dual views and pool storage')
            self.vision_seq_enc=416
            self.encoder_seq_len=416+self.max_prompt_len
            self.total_kv=self.encoder_seq_len+self.chunk_size
            return super()._allocate_buffers()

        def _prepare_selector(self):
            if hasattr(self,'_selector_lib'):return
            import torch
            from perceptbudget.selective_pool_cuda import load
            self._selector_lib=load()
            self._selector_scores=torch.empty(64,device='cuda',dtype=torch.float32)
            self._selector_mask=torch.empty(64,device='cuda',dtype=torch.int32)
            self._selector_mapping=torch.empty(160,device='cuda',dtype=torch.int32)

        def record_infer_graph(self,external_stream_int=None):
            self._prepare_selector()
            return super().record_infer_graph(external_stream_int=external_stream_int)

        def vision_encoder(self,stream=0):
            self._prepare_selector()
            # Native uniform pooling is overwritten; selector reads full,
            # fresh27-layer SigLIP features. Wrist stays unchanged.
            super().vision_encoder(stream=stream)
            code=self._selector_lib.selective_pool416(self.bufs['vision_x'].ptr.value,
                self._selector_scores.data_ptr(),self._selector_mask.data_ptr(),
                self._selector_mapping.data_ptr(),self.bufs['vision_x_pooled'].ptr.value,stream)
            if code:raise RuntimeError('Selective416 CUDA launch failed:'+str(code))
    return SelectivePipeline
