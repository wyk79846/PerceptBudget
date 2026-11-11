"""External-view DivPrune; keep wrist tokens and original RoPE coordinates.

The max-min cosine selection follows arxiv:2503.02175, Algorithm 1. Scores
use post-projector features. Selected original SigLIP tokens are then fed to
the unchanged projector/LLM. No training, attention readback, or upstream edit.
"""
import ctypes
from pathlib import Path
import subprocess

_LIB = None


def select_indices(features, keep=64, *, seeds=()):
    """NumPy reference, including the paper's non-random first-token rule."""
    import numpy as np
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not 0 < keep <= len(x):
        raise ValueError('Require a feature matrix and a valid retained count')
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    distances = 1 - x @ x.T
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    selected = []
    seeds=list(seeds)
    if len(seeds)>keep or len(set(seeds))!=len(seeds) or any(not 0<=i<len(x) for i in seeds):
        raise ValueError('Require unique in-range seed indices')
    for k in range(keep):
        index = int(seeds[k]) if k<len(seeds) else int(nearest.argmax())
        selected.append(index)
        nearest = (distances[index].copy() if len(selected) == 1 else
                   np.minimum(nearest, distances[index]))
        nearest[selected] = -np.inf
    return np.sort(np.asarray(selected, dtype=np.int32))


def motion_seeds(current, reference, keep=16):
    """Rank the external image's16x16 patches; zero change uses plain DivPrune."""
    import numpy as np
    current=np.asarray(current);reference=np.asarray(reference)
    if current.shape!=reference.shape or current.ndim!=3 or current.shape[-1]!=3:
        raise ValueError('Require matching RGB images')
    h,w,_=current.shape
    if h%16 or w%16 or keep!=16:
        raise ValueError('Require a16x16 patch grid and the fixed16-token candidate')
    delta=np.abs(current.astype(np.float32)-reference.astype(np.float32))
    scores=delta.reshape(16,h//16,16,w//16,3).mean(axis=(1,3,4)).reshape(256)
    indices=np.lexsort((np.arange(256),-scores))[:keep] if scores.max()>0 else np.empty(0,dtype=np.int32)
    return indices.astype(np.int32),scores


def load():
    global _LIB
    if _LIB is not None:
        return _LIB
    import torch
    major, minor = torch.cuda.get_device_capability()
    source = Path(__file__).with_suffix('.cu')
    directory = Path(__file__).resolve().parents[2] / '任务产出/perceptbudget/build' / f'external-divprune-sm{major}{minor}'
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / 'external_divprune.so'
    if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(['/usr/local/cuda/bin/nvcc', '-O3', '--fmad=false', '-shared',
                        '-Xcompiler', '-fPIC', f'-arch=sm_{major}{minor}',
                        str(source), '-o', str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.external_divprune.argtypes = [ctypes.c_void_p] * 7
    lib.external_divprune.restype = ctypes.c_int
    lib.external_divprune_seeded.argtypes = [ctypes.c_void_p] * 8
    lib.external_divprune_seeded.restype = ctypes.c_int
    lib.external_divprune_rope.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
    lib.external_divprune_rope.restype = ctypes.c_int
    _LIB = lib
    return lib


def pipeline_type(base_type, *, motion_enabled=False):
    from perceptbudget.mixed_view_pool import pipeline_type as mixed_type
    mixed = mixed_type(base_type, 'agent', preserve_positions=True)

    class ExternalDivPrunePipeline(mixed):
        def _prepare_divprune(self):
            if hasattr(self, '_div_lib'):
                return
            import numpy as np
            import torch
            from flash_rt.core.cuda_buffer import CudaBuffer
            self._div_lib = load()
            self._div_projected = torch.empty((256, 2048), dtype=torch.bfloat16, device='cuda')
            self._div_norm = torch.empty(256, dtype=torch.float32, device='cuda')
            self._div_distances = torch.empty((256, 256), dtype=torch.float32, device='cuda')
            self._div_mapping = torch.empty(320, dtype=torch.int32, device='cuda')
            if motion_enabled:
                self._div_seeds=torch.empty(17,dtype=torch.int32,device='cuda')
                self._div_seeds.copy_(torch.zeros(17,dtype=torch.int32))
            self._div_original_rope = CudaBuffer.from_numpy(np.ascontiguousarray(
                self._rope_table_np[:512 + self.max_prompt_len]))

        def set_motion_indices(self, indices):
            if not motion_enabled:
                if len(indices):raise ValueError('Motion selection was not prepared')
                return
            import numpy as np
            import torch
            packed=np.zeros(17,dtype=np.int32)
            if len(indices) not in (0,16) or len(set(indices))!=len(indices) or any(not 0<=i<256 for i in indices):
                raise ValueError('Motion selection needs zero or16 unique external positions')
            packed[0]=len(indices);packed[1:1+len(indices)]=indices
            self._div_seeds.copy_(torch.from_numpy(packed))

        def record_infer_graph(self, external_stream_int=None):
            self._prepare_divprune()
            return super().record_infer_graph(external_stream_int=external_stream_int)

        def vision_encoder(self, stream=0):
            self._prepare_divprune()
            super().vision_encoder(stream=stream)
            # Score in the same projected feature space consumed by the LLM.
            # Gather raw tokens afterwards: projection is token-wise, so the
            # selected embeddings are exactly those of the full input.
            b, w = self.bufs, self.weights
            self.fvk.layer_norm(b['vision_x'].ptr.value,
                w['vision_final_norm_w'], w['vision_final_norm_b'],
                b['vision_x_norm'].ptr.value, 256, 1152, 1e-5, stream=stream)
            self.gemm.bf16_nn(b['vision_x_norm'].ptr.value,
                w['encoder_multi_modal_projector_w'], self._div_projected.data_ptr(),
                256, 2048, 1152, stream=stream)
            self.fvk.add_bias_bf16(self._div_projected.data_ptr(),
                w['encoder_multi_modal_projector_b'], 256, 2048, stream=stream)
            arguments=[
                self._div_projected.data_ptr(), self._div_norm.data_ptr(),
                self._div_distances.data_ptr(), self._div_mapping.data_ptr(),
                b['vision_x'].ptr.value, b['vision_x_pooled'].ptr.value]
            if motion_enabled:
                code=self._div_lib.external_divprune_seeded(*arguments,self._div_seeds.data_ptr(),stream)
            else:code=self._div_lib.external_divprune(*arguments,stream)
            if code:
                raise RuntimeError('External DivPrune launch failed: ' + str(code))
            code = self._div_lib.external_divprune_rope(
                self._div_original_rope.ptr.value, self._div_mapping.data_ptr(),
                b['encoder_rope_weights'].ptr.value, self.encoder_seq_len, stream)
            if code:
                raise RuntimeError('External DivPrune position gather failed: ' + str(code))
    return ExternalDivPrunePipeline


def check_pipeline(pipeline, torch, np):
    """Check live gathered values and every camera/text RoPE row once."""
    torch.cuda.synchronize()
    indices = pipeline._div_mapping.cpu().numpy()
    if (len(set(indices[:64].tolist())) != 64 or np.any(np.diff(indices[:64]) <= 0)
            or indices[0] < 0 or indices[63] >= 256
            or not np.array_equal(indices[64:], np.arange(256, 512))):
        raise RuntimeError('External DivPrune camera indices are invalid')
    def download(name, shape):
        bits = np.empty(shape, dtype=np.uint16)
        pipeline.bufs[name].download(bits)
        return bits
    original = download('vision_x', (512, 1152))
    actual = download('vision_x_pooled', (320, 1152))
    positions = np.concatenate((indices, np.arange(512, 512 + pipeline.max_prompt_len)))
    rope = download('encoder_rope_weights', (pipeline.encoder_seq_len, 256))
    expected_rope = pipeline._rope_table_np[positions]
    mismatch = int(np.count_nonzero(actual != original[indices]))
    rope_mismatch = int(np.count_nonzero(rope != expected_rope.view(np.uint16)))
    if mismatch or rope_mismatch:
        raise RuntimeError(f'External DivPrune value/position mismatch: {mismatch}/{rope_mismatch}')
    return dict(external_token_indices=indices[:64].tolist(),external_tokens=64,wrist_tokens=256,
                mismatched_elements=mismatch,rope_mismatched_elements=rope_mismatch,
                text_position_start=512,action_position_offset=192)
