"""FlashRT-native SEAM-VLS adapter for LIBERO evaluation.

The action denoiser stays on FlashRT's native CUDA/GEMM path. Only the small
post-Euler SEAM correction is added, fused into the residual-add kernel. The
paper method is synchronous; async callers must crop expired actions before
passing a prior. For LIBERO, only the seven physical action dimensions are
guided; the model's remaining 25 output channels are padding.
"""

import ctypes
import math
import time


def queued_raw_tail(queue, step, horizon):
    """Snapshot the still-valid raw tail from the current queue's own origin."""
    import numpy as np

    if step not in queue:
        return None, 0, None
    origin = int(queue[step][2])
    executed = int(step - origin)
    if not 0 <= executed < horizon:
        raise ValueError('queued SEAM actions are outside their original horizon')
    items = []
    for offset in range(executed, horizon):
        item = queue.get(origin + offset)
        if (item is None or len(item) < 5 or item[2] != origin
                or item[3] != offset):
            raise ValueError('SEAM needs a contiguous raw tail from one prediction')
        items.append(item[4])
    tail = np.asarray(items, dtype=np.float32)
    if tail.shape != (horizon - executed, 32) or not np.isfinite(tail).all():
        raise ValueError('invalid raw SEAM queue tail')
    return tail.copy(), executed, origin


class SEAMGuidanceAdapter:
    def __init__(self, model, torch, *, horizon, num_steps,
                 strength=0.1, max_window=20, physical_action_dim=7,
                 latency_prior_seconds=0.4):
        from flash_rt.subgraphs.pi05.seam_vls import enable

        self.model = model
        self.pipe = model._pipe
        self.torch = torch
        self.horizon = int(horizon)
        self.num_steps = int(num_steps)
        self.latency_estimate_seconds = float(latency_prior_seconds)
        self.strength = float(strength)
        self.max_window = int(max_window)
        self.physical_action_dim = int(physical_action_dim)
        started = time.monotonic()
        self.native = enable(model, strength=self.strength,
                             max_window=self.max_window,
                             physical_action_dim=self.physical_action_dim)
        self.pipeline = self.native.pipeline
        if int(self.pipeline.num_steps) != self.num_steps:
            raise RuntimeError(
                f"FlashRT uses N={self.pipeline.num_steps}; SEAM expects "
                f"N={self.num_steps}")
        self.capture_seconds = time.monotonic() - started
        self.solver_precision_diagnostic = self._measure_solver_precision_delta()

    def _run_decode_graph(self, graph, noise):
        """Run a native decode graph with fixed noise and return raw BF16 output."""
        torch = self.torch
        pipe = self.pipe
        pl = self.native.pipeline
        stream = pipe._graph_torch_stream
        context_graph = getattr(pl, '_context_graph', None)
        if context_graph is None:
            raise RuntimeError('FlashRT context graph was not captured')
        with torch.cuda.stream(stream):
            context_graph.replay(pl._graph_stream)
            pipe._copy_tensor_to_pipeline_buf_stream(
                noise, pl.input_noise_buf, stream.cuda_stream)
            graph.replay(pl._graph_stream)
            pipe._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(pipe._noise_out.data_ptr()),
                ctypes.c_void_p(pl.input_noise_buf.ptr.value),
                pipe._noise_out.numel() * 2, 3, stream.cuda_stream)
        stream.synchronize()
        return pipe._noise_out.detach().clone()

    def _measure_solver_precision_delta(self):
        """Measure the solver-precision delta with guidance disabled."""
        import numpy as np

        pl = self.native.pipeline
        baseline_graph = getattr(pl, '_decoder_only_graph', None)
        seam_graph = getattr(pl, '_seam_vls_graph', None)
        if baseline_graph is None or seam_graph is None:
            raise RuntimeError('FlashRT native decoder graphs were not captured')
        noise = self.pipe._noise_buf.detach().clone()
        self.native.set_prior(
            np.zeros((self.horizon, 32), dtype=np.float32), 0, 0)
        baseline = self._run_decode_graph(baseline_graph, noise)
        candidate = self._run_decode_graph(seam_graph, noise)
        difference = candidate.float() - baseline.float()
        max_error = float(difference.abs().max().item())
        baseline_rms = float(baseline.float().square().mean().sqrt().item())
        relative_rmse = float(
            difference.square().mean().sqrt().item() / max(baseline_rms, 1e-12))
        result = {
            'implementation': 'flashrt_native_fused_cuda',
            'window_zero_is_unsteered_fp32_solver': True,
            'bit_exact_to_bf16_accumulation': bool(
                self.torch.equal(candidate, baseline)),
            'max_absolute_difference_from_bf16_accumulation': max_error,
            'relative_rmse_from_bf16_accumulation': relative_rmse,
        }
        if not self.torch.isfinite(candidate).all():
            raise RuntimeError(
                'Unsteered FP32 solver produced non-finite actions')
        return result

    def estimate_delay_steps(self, control_period_seconds):
        if control_period_seconds <= 0:
            raise ValueError('control period must be positive')
        return max(1, math.ceil(
            self.latency_estimate_seconds / control_period_seconds))

    def _aligned_prior(self, previous_tail):
        import numpy as np

        tail = np.asarray(previous_tail, dtype=np.float32)
        if (tail.ndim != 2 or tail.shape[1] != 32 or
                not 1 <= tail.shape[0] <= self.horizon or
                not np.isfinite(tail).all()):
            raise ValueError('previous_tail must be finite with shape (1..H, 32)')
        prior = np.empty((self.horizon, 32), dtype=np.float32)
        prior[:len(tail)] = tail
        prior[len(tail):] = tail[-1]
        return prior

    def predict(self, snapshot, prompt, previous_tail, *, executed,
                delay_estimate, request):
        import numpy as np
        from flash_rt.core.utils.actions import unnormalize_actions
        from perceptbudget.seam_guidance import clip_prior_tail_to_executed_actions

        torch = self.torch
        pipe = self.pipe
        request['inference_started_monotonic'] = time.monotonic()
        self.model.set_prompt(prompt, state=snapshot.get('state'))
        pl = self.native.pipeline
        graph = getattr(pl, '_seam_vls_graph', None)
        context_graph = getattr(pl, '_context_graph', None)
        if graph is None or context_graph is None:
            raise RuntimeError('FlashRT native context/SEAM graphs are unavailable')

        if previous_tail is None:
            if int(executed) != 0 or int(delay_estimate) != 0:
                raise ValueError(
                    'an absent prior is valid only for an unsteered fresh chunk')
            tail = np.empty((0, 32), dtype=np.float32)
            prior_tail = np.zeros((self.horizon, 32), dtype=np.float32)
            prior_clip_stats = {
                'raw_out_of_range_fraction': 0.0,
                'raw_max_abs': 0.0,
                'clip_delta_rms': 0.0,
            }
            guided_start = 0
            guided_window = 0
        else:
            expected = (self.horizon - int(executed), 32)
            tail = np.asarray(previous_tail, dtype=np.float32)
            if tail.shape != expected:
                raise ValueError(
                    f'old SEAM tail shape {tail.shape}, expected {expected}')
            prior_tail, prior_clip_stats = clip_prior_tail_to_executed_actions(
                tail, self.physical_action_dim)
            # Rows before the estimated handoff are expired and cropped on commit.
            # Guide only the surviving overlap, using the matching old-queue rows.
            guided_start = min(max(int(delay_estimate), 0), len(tail))
            guided_window = min(self.max_window, len(tail) - guided_start,
                                self.horizon - guided_start)
        prior = self._aligned_prior(prior_tail)
        self.native.set_prior(prior, guided_start, guided_window)

        stream = pipe._graph_torch_stream
        with torch.cuda.stream(stream):
            pipe._fill_img_buf({'images': snapshot['images']})
            pipe._copy_tensor_to_pipeline_buf_stream(
                pipe._img_buf, pl.input_images_buf, stream.cuda_stream)
            context_graph.replay(pl._graph_stream)
            pipe._noise_buf.normal_()
            pipe._copy_tensor_to_pipeline_buf_stream(
                pipe._noise_buf, pl.input_noise_buf, stream.cuda_stream)
            graph.replay(pl._graph_stream)
            pipe._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(pipe._noise_out.data_ptr()),
                ctypes.c_void_p(pl.input_noise_buf.ptr.value),
                pipe._noise_out.numel() * 2, 3, stream.cuda_stream)
        stream.synchronize()

        raw_actions = pipe._noise_out.float().cpu().numpy().astype(
            np.float32, copy=True)
        actions = unnormalize_actions(raw_actions, pipe.norm_stats)[:, :7]
        request['actions_raw'] = raw_actions.tolist()
        request['seam_guidance'] = {
            'method': ('SEAM VLS on FlashRT native CUDA path'
                       if guided_window else 'unsteered FP32 solver control'),
            'implementation': 'fused Euler residual-add + VLS CUDA kernel',
            'paper_is_synchronous': True,
            'execution_steps_before_request': int(executed),
            'estimated_inference_delay_steps': int(delay_estimate),
            'old_tail_start_offset': int(executed),
            'old_tail_length': int(len(tail)),
            'guided_start': int(guided_start),
            'guided_window': int(guided_window),
            'vls_enabled': bool(guided_window and self.strength > 0),
            'strength': self.strength,
            'physical_action_dim': self.physical_action_dim,
            'solver_accumulation': 'FP32; BF16 expert inputs',
            'prior_matches_executed_action_clip': True,
            'prior_clip_stats': prior_clip_stats,
        }
        request['inference_ended_monotonic'] = time.monotonic()
        self.latency_estimate_seconds = (
            request['inference_ended_monotonic']
            - request['inference_started_monotonic'])
        return actions
