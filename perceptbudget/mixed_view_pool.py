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


def pipeline_type(base_type, pooled_view):
    if pooled_view not in ('agent','wrist'):
        raise ValueError('Choose one pooled camera')
    class MixedViewPipeline(base_type):
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
