"""Experimental Full/wrist contexts sharing frozen weight tensors, not KV."""
import copy
import json
import time
from pathlib import Path

VISUAL_PREFIX_REFERENCES = {
    'libero_spatial': ('20261021T094320.446597Z-fast-quality-full-1e41ec3b',
                      '20261021T101812.515387Z-fast-quality-wrist-f5d7b76c'),
    'libero_object': ('20261021T095551.276524Z-fast-quality-full-56fd7548',
                     '20261021T101944.442981Z-fast-quality-wrist-ab0a2ce9'),
    'libero_goal': ('20261021T101328.104719Z-fast-quality-full-02768e71',
                   '20261021T102104.966625Z-fast-quality-wrist-324c98dc'),
}
POOL_PREFIX_REFERENCES = {'libero_spatial':'20261021T110243.748047Z-fast-quality-pool2-46646e05'}


def historical_visual_prefixes(config, suite, task_id, full_valid, wrist_valid, *, allowed_text_range=None):
    """Prewarm observed shape envelope only; never reuse actions or select policy.

    Unseen shapes must fail closed before the expert, not silently run slowly.
    This is NOT exhaustive coverage of all tokenizer state bins.
    """
    if task_id != 0 or suite not in VISUAL_PREFIX_REFERENCES:
        raise ValueError('Historical visual prewarm supports the three task0 probes only')
    root = Path(config['output_root'])/'M2'
    mode = config.get('m2_visual_fast_mode', 'wrist')
    if mode not in ('wrist','pool2','pool2_agent'):
        raise ValueError('Unsupported fresh visual mode')
    fast_tokens = 320 if mode=='pool2_agent' else 128 if mode=='pool2' else 256
    text_lengths = [full_valid-512, wrist_valid-fast_tokens]
    sources = []
    for views, name in zip((2, 1), VISUAL_PREFIX_REFERENCES[suite]):
        run = root/name
        previous = json.loads((run/'run.json').read_text())['config_snapshot']
        for key in ('checkpoint_provenance_revision','H','N','state_enabled','state_prompt_capacity'):
            if previous[key] != config[key]:
                raise ValueError('Historical visual prewarm mismatch: '+key)
        if previous['m2_view'] != ('full' if views==2 else 'wrist'):
            raise ValueError('Historical prewarm camera geometry mismatch')
        requests = json.loads((run/'rtc/q5_requests.json').read_text())
        text_lengths.extend(r['rtc_guidance']['valid_prefix_length']-views*256 for r in requests)
        sources.append(str(run))
    if mode=='pool2' and suite in POOL_PREFIX_REFERENCES:
        run=root/POOL_PREFIX_REFERENCES[suite]
        previous=json.loads((run/'run.json').read_text())['config_snapshot']
        for key in ('checkpoint_provenance_revision','H','N','state_enabled','state_prompt_capacity'):
            if previous[key]!=config[key]:
                raise ValueError('Historical pool prewarm mismatch: '+key)
        if previous.get('m2_vision_pool_factor')!=2 or previous.get('m2_pool_view','both')!='both':
            raise ValueError('Historical pool geometry mismatch')
        requests=json.loads((run/'rtc/q5_requests.json').read_text())
        text_lengths.extend(r['rtc_guidance']['valid_prefix_length']-128 for r in requests)
        sources.append(str(run))
    low, high = min(text_lengths), max(text_lengths)
    if allowed_text_range is not None:
        floor, ceiling=allowed_text_range
        if not floor<=low<=high<=ceiling:
            raise ValueError('Observed text lengths outside verified tokenizer bounds')
        margin=2 if mode.startswith('pool2') else 0
        low,high=max(floor,low-margin),min(ceiling,high+margin)
    if not 0<=low<=high<=200 or high-low+1>17:
        raise ValueError('Historical visual prefix envelope invalid')
    full = list(range(512+low,512+high+1))
    wrist = list(range(fast_tokens+low,fast_tokens+high+1))
    return full+wrist, dict(kind='historical_shape_envelope_not_exhaustive', sources=sources,
        text_range=[low,high], theoretical_text_range=allowed_text_range,
        margin_lengths=2 if mode.startswith('pool2') and allowed_text_range is not None else 0,
        full=full, fast_mode=mode, fast=wrist,
        unseen_prefix='abort before expert; no slow fallback, padding or runtime capture')


def wrist_frontend(full, torch, *, fast_kind='wrist'):
    if fast_kind not in ('wrist','pool2','pool2_agent'):
        raise ValueError('Unknown shared visual geometry')
    if full.num_views != 2 or full._vision_pool_factor != 1:
        raise ValueError('Require unpooled dual-view source')
    if getattr(full, '_pb_context_only', False):
        raise ValueError('Prepare shared frontend before releasing native expert')
    clone = copy.copy(full)
    if fast_kind=='pool2_agent':
        from perceptbudget.compact_state import frontend_type
        from perceptbudget.mixed_view_pool import pipeline_type
        from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
        kind=frontend_type(type(full),full._fixed_pipeline.max_prompt_len,
                           pipeline_type=pipeline_type(Pi05Pipeline,'agent'))
        clone=kind.__new__(kind)
        clone.__dict__.update(full.__dict__)
    # Separate dictionaries so context-only cleanup cannot mutate the peer.
    # Tensors themselves are deliberately shared, never cloned or reloaded.
    clone._ckpt_bf16 = dict(full._ckpt_bf16)
    clone.num_views = 1 if fast_kind=='wrist' else 2
    clone._vision_pool_factor = 1 if fast_kind=='wrist' else 2
    clone._img_buf = full._img_buf[1:] if fast_kind=='wrist' else full._img_buf
    clone.pipeline = clone._fixed_pipeline = None
    clone._prompt_pipeline_cache = {}
    clone.current_prompt_len = 0
    clone.graph_recorded = clone.calibrated = False
    clone.latency_records = []
    clone._frame_count = 0
    clone._flashrt_subgraph_frontend_hooks = list(getattr(full, '_flashrt_subgraph_frontend_hooks', ()))
    from flash_rt.frontends.torch.pi05_rtx import RtxFlashAttnBackend, ENC_L
    clone.attn_backend = RtxFlashAttnBackend(num_views=clone.num_views,
        encoder_seq_max=256*clone.num_views+clone.max_prompt_len, chunk_size=clone.chunk_size, num_encoder_layers=ENC_L)
    return clone


def probe(model, snapshot, prompt, previous, torch, config, directory, metadata, *, fast_kind='wrist'):
    import gc
    import numpy as np
    from flash_rt.api import VLAModel
    from flash_rt.subgraphs.pi05.context_action import enable
    from perceptbudget.device_profile import select_images
    from perceptbudget.experiment import _write_json_atomically
    from perceptbudget.rtc_memory import memory_record, release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.rtc_visual_alternation import visual_geometry
    from scripts import eval_q as base

    directory = Path(directory)
    result = dict(metadata, status='running', measurements=[], memory=[],
        fast_kind=fast_kind,
        scope='Fixed-input Full/'+fast_kind+' switching feasibility, one shared checkpoint and expert; not rollout or quality proof')
    def check_memory(stage):
        r = memory_record(torch, stage)
        result['memory'].append(r)
        if r['available_mib'] < 900:
            raise RuntimeError('Shared context RAM below900MiB; stop, do not reload per request')
    try:
        check_memory('before_second_context')
        wrist = VLAModel(wrist_frontend(model._pipe, torch, fast_kind=fast_kind), 'torch')
        enable(wrist)
        small = select_images(snapshot, 'wrist' if fast_kind=='wrist' else 'full')
        base._infer_actions(wrist, small, prompt, 50, {})
        geometry=visual_geometry(fast_kind)
        if wrist._pipe.pipeline.num_views!=geometry['num_views'] or wrist._pipe.pipeline.vision_seq_enc!=geometry['visual_tokens']:
            raise RuntimeError('Candidate context geometry mismatch')
        if fast_kind.startswith('pool2'):
            from scripts.diagnose_pool_parity import pool_check
            result['pool_layout']=pool_check(wrist._pipe.pipeline,torch,np,'agent' if fast_kind=='pool2_agent' else 'both')
            if result['pool_layout']['mismatched_elements']:
                raise RuntimeError('Candidate pooling layout mismatch')
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
        for index, mode in enumerate(('full',fast_kind)*3):
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
