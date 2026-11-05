"""Project-owned pipeline subclass: pool one camera, preserve the other.

No upstream patch, new kernel, training or zero/mask substitute. Camera order
stays external then wrist. Only encoder token count changes; SigLIP stays full.
"""
import ctypes


def visual_tokens(config):
    f=config.get('m2_vision_pool_factor',1)
    view=config.get('m2_pool_view','both')
    if view not in ('both','agent','wrist'):
        raise ValueError('Unknown pooled camera')
    return 512//f**2 if view=='both' else 256+256//f**2


def original_position_indices(pooled_view, factor, prompt_capacity):
    """Represent each pooled block by its original top-left token position."""
    small=[r*16+c for r in range(0,16,factor) for c in range(0,16,factor)]
    visual=(small+list(range(256,512)) if pooled_view=='agent' else
            list(range(256))+[256+i for i in small])
    return visual+list(range(512,512+prompt_capacity))


def pipeline_type(base_type, pooled_view, *, preserve_positions=False):
    if pooled_view not in ('agent','wrist'):
        raise ValueError('Choose one pooled camera')
    class MixedViewPipeline(base_type):
        def _build_rope_table(self):
            if not preserve_positions:return super()._build_rope_table()
            import numpy as np
            from flash_rt.core.cuda_buffer import CudaBuffer
            reduced=self.encoder_seq_len
            try:
                self.encoder_seq_len=512+self.max_prompt_len
                super()._build_rope_table()
            finally:self.encoder_seq_len=reduced
            positions=original_position_indices(pooled_view,self.vision_pool_factor,self.max_prompt_len)
            self.bufs['encoder_rope_weights']=CudaBuffer.from_numpy(
                np.ascontiguousarray(self._rope_table_np[positions]))
            self._pb_original_visual_positions=True

        def _set_decoder_rope_for_prompt(self,prompt_len):
            if not preserve_positions:return super()._set_decoder_rope_for_prompt(prompt_len)
            import numpy as np
            self.bufs['decoder_rope_weights'].upload(np.ascontiguousarray(
                self._rope_table_np[512+prompt_len:512+prompt_len+self.chunk_size]))

        def _allocate_buffers(self):
            if self.num_views!=2 or self.vision_pool_factor not in (2,4):
                raise ValueError('Mixed pooling requires dual views and factor2/4')
            self.vision_seq_enc=256+256//self.vision_pool_factor**2
            self.encoder_seq_len=self.vision_seq_enc+self.max_prompt_len
            self.total_kv=self.encoder_seq_len+self.chunk_size
            return super()._allocate_buffers()

        def vision_encoder(self,stream=0):
            # Native encoder initially pools both views into a distinct buffer.
            super().vision_encoder(stream=stream)
            from flash_rt.core.cuda_buffer import _cudart
            row_bytes=1152*2
            small=256//self.vision_pool_factor**2
            src=self.bufs['vision_x'].ptr.value
            dst=self.bufs['vision_x_pooled'].ptr.value
            def copy(destination,source,rows):
                code=_cudart.cudaMemcpyAsync(ctypes.c_void_p(destination),ctypes.c_void_p(source),
                    rows*row_bytes,3,ctypes.c_void_p(stream))
                if code:
                    raise RuntimeError('Mixed view D2D copy failed: '+str(code))
            if pooled_view=='agent':
                # First small block already contains pooled external features.
                copy(dst+small*row_bytes,src+256*row_bytes,256)
            else:
                # Move the small wrist block before restoring full external view.
                copy(dst+256*row_bytes,dst+small*row_bytes,small)
                copy(dst,src,256)
    return MixedViewPipeline
