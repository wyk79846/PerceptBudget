"""Experimental one-camera SigLIP refresh; full language encoder and fresh state.

Only the other camera's last Full features are reused. The full KV is always
recomputed. This does not alter FlashRT or the original RTC sampler.
"""
import ctypes


def changed_camera(images, cached_images):
    import numpy as np
    if len(images) != 2 or len(cached_images) != 2:
        raise ValueError('Require two cameras in original order')
    differences = []
    for fresh, old in zip(images, cached_images):
        fresh, old = np.asarray(fresh), np.asarray(old)
        if fresh.shape != old.shape or fresh.shape != (224, 224, 3):
            raise ValueError('Require matched preprocessed 224x224 RGB images')
        differences.append(float(np.mean(np.abs(fresh.astype(np.float32)-old.astype(np.float32)))))
    if not np.isfinite(differences).all():
        raise ValueError('Nonfinite camera change')
    return int(np.argmax(differences)), differences


class CameraFeatureContext:
    """Two prewarmed shared-weight contexts, used on one serialized GPU stream.

    The caller controls Fast admission. Diagnostic cost probes may repeat Fast
    against one fixed Full cache; deployment must prohibit consecutive Fast.
    """
    def __init__(self, full_model, single_model, torch):
        from flash_rt.core.cuda_graph import CUDAGraph
        self.model, self.single, self.torch = full_model, single_model, torch
        self.pipe, self.pl = full_model._pipe, full_model._pipe.pipeline
        self.one = single_model._pipe.pipeline
        if (self.pl.num_views != 2 or self.one.num_views != 1
                or self.pl.vision_seq_enc != 512 or self.one.vision_seq_enc != 256
                or not self.pl._fixed_shape or not self.pipe._force_bf16):
            raise ValueError('Require fixed BF16 unpooled Full and one-view pipelines')
        if self.pl.bufs['vision_x'].nbytes != 2*self.one.bufs['vision_x'].nbytes:
            raise ValueError('Camera feature layouts differ')
        self.stream = self.pipe._graph_torch_stream
        self.handle = ctypes.c_void_p(self.stream.cuda_stream)
        self.cache = torch.empty((2, 256, 1152), dtype=torch.bfloat16, device='cuda')
        self.cached_images = None
        self.cached_step = None
        torch.cuda.synchronize()
        self.vision_graph, self.encoder_graph = CUDAGraph(), CUDAGraph()
        def encode():
            self.pl._copy_lang_embeds_to_encoder_x(stream=self.stream.cuda_stream)
            self.pl.transformer_encoder(self.stream.cuda_stream)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.one.vision_encoder(self.stream.cuda_stream)
                encode()
            self.stream.synchronize()
            self.vision_graph.begin_capture(self.handle)
            self.one.vision_encoder(self.stream.cuda_stream)
            self.vision_graph.end_capture(self.handle)
            self.encoder_graph.begin_capture(self.handle)
            encode()
            self.encoder_graph.end_capture(self.handle)
        self.stream.synchronize()

    def _copy(self, destination, source, size):
        from flash_rt.core.cuda_buffer import _cudart, _check
        _check(_cudart.cudaMemcpyAsync(ctypes.c_void_p(destination), ctypes.c_void_p(source),
                                     size, 3, self.handle), 'camera feature D2D')

    def full(self, snapshot, prompt, *, step=None):
        """Original Full graph plus an immutable, small visual-feature cache."""
        with self.torch.cuda.stream(self.stream):
            self.model.set_prompt(prompt, state=snapshot['state'])
            self.pipe._fill_img_buf({'images': snapshot['images']})
            self.pipe._copy_tensor_to_pipeline_buf_stream(self.pipe._img_buf,
                self.pl.input_images_buf, self.stream.cuda_stream)
            self.pl._context_graph.replay(self.handle)
            self._copy(self.cache.data_ptr(), self.pl.bufs['vision_x'].ptr.value,
                       self.pl.bufs['vision_x'].nbytes)
        self.stream.synchronize()
        self.cached_images = [image.copy() for image in snapshot['images']]
        self.cached_step = step
        return int(self.pl.attn.dec_devpos.item())

    def fast(self, snapshot, prompt, *, step=None, camera=None):
        """Fresh selected view + cached peer -> full VLM, latest state, new KV."""
        if self.cached_images is None:
            raise RuntimeError('Fast requires a preceding Full cache')
        selected, differences = changed_camera(snapshot['images'], self.cached_images)
        if camera is not None:
            if camera not in (0, 1):
                raise ValueError('Camera must be 0 or 1')
            selected = camera
        peer = self.single._pipe
        with self.torch.cuda.stream(self.stream):
            self.model.set_prompt(prompt, state=snapshot['state'])
            peer._fill_img_buf({'images': [snapshot['images'][selected]]})
            peer._copy_tensor_to_pipeline_buf_stream(peer._img_buf, self.one.input_images_buf,
                                                    self.stream.cuda_stream)
            self.vision_graph.replay(self.handle)
            destination = self.pl.bufs['vision_x'].ptr.value
            self._copy(destination, self.cache.data_ptr(), self.pl.bufs['vision_x'].nbytes)
            nbytes = self.one.bufs['vision_x'].nbytes
            self._copy(destination+selected*nbytes, self.one.bufs['vision_x'].ptr.value, nbytes)
            self.encoder_graph.replay(self.handle)
        self.stream.synchronize()
        return int(self.pl.attn.dec_devpos.item()), dict(
            recomputed_camera=selected, cached_camera=1-selected,
            image_change_mean_abs=differences, cached_full_step=self.cached_step,
            cached_age_steps=None if step is None or self.cached_step is None else step-self.cached_step,
            state_enabled=True, visual_tokens=512, kv_recomputed=True)

    def features(self):
        """Diagnostic copy of the full native BF16 visual features."""
        result = self.torch.empty_like(self.cache)
        self._copy(result.data_ptr(), self.pl.bufs['vision_x'].ptr.value,
                   self.pl.bufs['vision_x'].nbytes)
        self.stream.synchronize()
        return result
