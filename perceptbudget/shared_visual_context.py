"""Experimental Full/wrist contexts sharing frozen weight tensors, not KV."""
import copy
import json
import time
from pathlib import Path


def wrist_frontend(full, torch):
    if full.num_views != 2 or full._vision_pool_factor != 1:
        raise ValueError('Require unpooled dual-view source')
    if getattr(full, '_pb_context_only', False):
        raise ValueError('Prepare shared frontend before releasing native expert')
    clone = copy.copy(full)
    # Separate dictionaries so context-only cleanup cannot mutate the peer.
    # Tensors themselves are deliberately shared, never cloned or reloaded.
    clone._ckpt_bf16 = dict(full._ckpt_bf16)
    clone.num_views = 1
    clone._img_buf = full._img_buf[1:]
    clone.pipeline = clone._fixed_pipeline = None
    clone._prompt_pipeline_cache = {}
    clone.current_prompt_len = 0
    clone.graph_recorded = clone.calibrated = False
    clone.latency_records = []
    clone._frame_count = 0
    clone._flashrt_subgraph_frontend_hooks = list(getattr(full, '_flashrt_subgraph_frontend_hooks', ()))
    from flash_rt.frontends.torch.pi05_rtx import RtxFlashAttnBackend, ENC_L
    clone.attn_backend = RtxFlashAttnBackend(num_views=1,
        encoder_seq_max=256+clone.max_prompt_len, chunk_size=clone.chunk_size, num_encoder_layers=ENC_L)
    return clone


def probe(model, snapshot, prompt, previous, torch, config, directory, metadata):
    import gc
    import numpy as np
    from flash_rt.api import VLAModel
    from flash_rt.subgraphs.pi05.context_action import enable
    from perceptbudget.device_profile import select_images
    from perceptbudget.experiment import _write_json_atomically
    from perceptbudget.rtc_memory import memory_record, release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from scripts import eval_q as base

    directory = Path(directory)
    result = dict(metadata, status='running', measurements=[], memory=[],
        scope='Fixed-input Full/wrist context switching feasibility, one shared checkpoint and one expert; not rollout or quality proof')
    def check_memory(stage):
        r = memory_record(torch, stage)
        result['memory'].append(r)
        if r['available_mib'] < 900:
            raise RuntimeError('Shared context RAM below900MiB; stop, do not reload per request')
    try:
        check_memory('before_second_context')
        wrist = VLAModel(wrist_frontend(model._pipe, torch), 'torch')
        enable(wrist)
        small = select_images(snapshot, 'wrist')
        base._infer_actions(wrist, small, prompt, 50, {})
        if wrist._pipe.pipeline.num_views != 1 or wrist._pipe.pipeline.vision_seq_enc != 256:
            raise RuntimeError('Wrist context did not use actual single-view geometry')
        check_memory('second_context_prepared')
        full_weights, small_weights = model._pipe._ckpt_bf16, wrist._pipe._ckpt_bf16
        if not all(full_weights[k] is small_weights[k] for k in full_weights):
            raise RuntimeError('Checkpoint weight sharing failed')
        if model._pipe.pipeline is wrist._pipe.pipeline or model._pipe.attn_backend is wrist._pipe.attn_backend:
            raise RuntimeError('Contexts must own separate KV and graphs')
        result['shared_checkpoint_tensors'] = len(full_weights)
        result['native_release'] = [release_unused_native_expert(m, torch) for m in (model, wrist)]
        full = OriginalOpenPiRTCAdapter(model, torch, config)
        gate = json.loads((Path(config['verified_reference_run'])/'summary.json').read_text())
        if not gate.get('passed') or full.expert.provenance != gate['original_source_hashes']:
            raise RuntimeError('Frozen source gate mismatch')
        r = {}
        full.bootstrap(snapshot, prompt, r)
        lengths = [r['rtc_guidance']['valid_prefix_length']]
        fast = copy.copy(full)
        fast.model, fast.pipe, fast.pipeline = wrist, wrist._pipe, wrist._pipe.pipeline
        r = {}
        fast.bootstrap(small, prompt, r)
        lengths.append(r['rtc_guidance']['valid_prefix_length'])
        result['graph_audit'] = full.prepare_step_bank(lengths)
        fast.step_bank = full.step_bank
        fast._step_bank_stream = full._step_bank_stream
        result['valid_prefixes'] = lengths
        gc.collect()
        check_memory('shared_expert_graphs_ready')
        references = {}
        for index, mode in enumerate(('full','wrist')*3):
            adapter, inputs = (full, snapshot) if mode == 'full' else (fast, small)
            torch.manual_seed(20261023)
            r = {}
            begin = time.monotonic()
            actions = adapter.predict(inputs, prompt, previous, delay=5, executed=5, request=r)
            elapsed = (time.monotonic()-begin)*1000
            raw = np.asarray(r['actions_raw'], dtype=np.float32)
            if mode not in references:
                references[mode] = raw.copy()
            difference = float(np.max(np.abs(raw-references[mode])))
            if difference != 0 or r['rtc_guidance'].get('variable_length_fallback') or not np.isfinite(actions).all():
                raise RuntimeError('Context switch replay parity/fallback failed')
            result['measurements'].append(dict(mode=mode, milliseconds=elapsed,
                raw_max_abs_vs_same_mode_first=difference, guidance=r['rtc_guidance']))
            np.savez_compressed(directory/f'shared_view_{index}.npz', actions=actions, raw=raw)
            check_memory('after_'+str(index))
            print('SHARED_VIEW='+mode+' ms='+str(elapsed), flush=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(directory/'shared_visual_probe.json', result)
    return result
