"""Project-side selective token pipeline with GPU redundancy selection."""


def pipeline_type(base_type, *, visual_tokens=416,preserve_positions=False):
    if visual_tokens not in (320,416,464):raise ValueError('Selective pooling supports320,416 or464 tokens')
    if preserve_positions and visual_tokens==416:raise ValueError('Original-position probe supports dual selected views')
    class SelectivePipeline(base_type):
        def _build_rope_table(self):
            # Build the native table at original512-image coordinates. Compute
            # still uses the reduced sequence; the larger table is only data.
            if not preserve_positions:return super()._build_rope_table()
            reduced=self.encoder_seq_len
            try:
                self.encoder_seq_len=512+self.max_prompt_len
                return super()._build_rope_table()
            finally:self.encoder_seq_len=reduced

        def _set_decoder_rope_for_prompt(self,prompt_len):
            if not preserve_positions:return super()._set_decoder_rope_for_prompt(prompt_len)
            import numpy as np
            self.bufs['decoder_rope_weights'].upload(np.ascontiguousarray(
                self._rope_table_np[512+prompt_len:512+prompt_len+self.chunk_size]))

        def _allocate_buffers(self):
            if self.num_views!=2 or self.vision_pool_factor!=2:
                raise ValueError('Selective pooling requires dual views and pool storage')
            self._pb_original_visual_positions=preserve_positions
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
            if preserve_positions:
                import numpy as np
                from flash_rt.core.cuda_buffer import CudaBuffer
                self._selector_original_rope=CudaBuffer.from_numpy(np.ascontiguousarray(
                    self._rope_table_np[:512+self.max_prompt_len]))

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
            if preserve_positions:
                code=self._selector_lib.selective_rope(self._selector_original_rope.ptr.value,
                    self._selector_mapping.data_ptr(),self.bufs['encoder_rope_weights'].ptr.value,
                    visual_tokens,self.encoder_seq_len,stream)
                if code:raise RuntimeError('Selective original-position launch failed:'+str(code))
    return SelectivePipeline


def align_expert_positions(expert,visual_tokens,fast_prompt_capacity,*,active_fast=None):
    """Use matching action positions with the original-coordinate prefix KV.

    The author denoising/VJP sampler and frozen modules remain intact. Only
    Fast's position_ids input is offset; Full forwards its exact original input.
    Stateless Fast may use disjoint prefix lengths. Stateful Fast supplies an
    explicit geometry callback because its lengths can overlap Full's.
    """
    if visual_tokens not in (320,464) or not 0<fast_prompt_capacity<=200:
        raise ValueError('Need a supported visual and prompt capacity')
    if active_fast is None and fast_prompt_capacity>=512-visual_tokens:
        raise ValueError('Need a disjoint stateless visual prefix family or explicit geometry')
    if active_fast is not None and not callable(active_fast):raise ValueError('Geometry selector must be callable')
    bridge=expert.suffix.paligemma_with_expert
    if hasattr(bridge,'_pb_original_coordinate_forward'):
        raise ValueError('Expert position policy already installed')
    original=bridge.forward;offset=512-visual_tokens
    def forward(*args,**kwargs):
        prefix=kwargs['attention_mask'].shape[-1]-kwargs['inputs_embeds'][1].shape[1]
        compact=(bool(active_fast()) if active_fast is not None else
                 visual_tokens<prefix<=visual_tokens+fast_prompt_capacity)
        if compact:
            if not visual_tokens<prefix<=visual_tokens+fast_prompt_capacity:
                raise ValueError('Selected Fast prefix exceeds its actual geometry')
            kwargs=dict(kwargs,position_ids=kwargs['position_ids']+offset)
        elif prefix<512:
            raise ValueError('Unknown compact geometry in original-position expert')
        return original(*args,**kwargs)
    bridge._pb_original_coordinate_forward=original
    bridge.forward=forward
    return dict(policy='original_camera_and_text_coordinates_v1',visual_tokens=visual_tokens,
        action_position_offset=offset,fast_prompt_capacity=fast_prompt_capacity,
        full_position_offset=0,rtc_sampler_changed=False,explicit_geometry=active_fast is not None)
