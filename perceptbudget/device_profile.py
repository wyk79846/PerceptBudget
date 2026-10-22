"""M2 discovery and synthetic sensor diagnostics, not a model energy benchmark."""
import ast
import hashlib
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from perceptbudget.device_control import inventory
from perceptbudget.power import PowerMonitor


def inspect_fast_interface(flashrt_root):
    root = Path(flashrt_root)
    paths = ['flash_rt/frontends/torch/pi05_rtx.py', 'flash_rt/models/pi05/pipeline_rtx.py']
    evidence = []
    view_argument_found = False
    for name in paths:
        path = root / name
        source = path.read_text()
        tree = ast.parse(source)
        if 'frontends/' in name:
            view_argument_found = any(isinstance(n, ast.FunctionDef) and n.name == '__init__'
                and 'num_views' in [a.arg for a in n.args.args + n.args.kwonlyargs]
                for n in ast.walk(tree))
        evidence.append(dict(path=str(path), sha256=hashlib.sha256(source.encode()).hexdigest(),
            relevant_lines=[dict(line=i, source=line.strip()) for i, line in enumerate(source.splitlines(), 1)
                            if any(key in line for key in ('num_views:', 'self.num_views =',
                                'self.vision_seq =', 'self.vision_seq_enc =',
                                'self.encoder_seq_len =', 'img_list[:self.num_views]'))]))
    return dict(scope='source inspection only; no live Fast speed/quality validation',
                project_loader_current_views=2, upstream_num_views_interface=view_argument_found,
                expected_visual_tokens=dict(full=512, single_view=256),
                expected_encoder_capacity_with_200_text_tokens=dict(full=712, single_view=456),
                flashrt_modified=False, source=evidence)


def sensor_steps(*, block_seconds=2., pairs=3, poll_interval=.02):
    """Matched synthetic GPU busy/idle steps to inspect sensor response only.

    Does not change frequencies, load a checkpoint, run LIBERO, or assert railgate.
    GPU lock must be held by the calling process.
    """
    if block_seconds < 1 or pairs < 1:
        raise ValueError('sensor windows require >=1 second and >=1 pair')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(memory['MemAvailable'].split()[0]) < 768 * 1024:
        raise RuntimeError('Available RAM below 768 MiB; sensor load not started')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; sensor load steps not started')
    torch.manual_seed(7)
    a = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    monitor = PowerMonitor(poll_interval=poll_interval)
    blocks = []
    monitor.start()
    try:
        for mode in ['idle', 'busy'] * pairs + ['idle']:
            start = time.monotonic()
            iterations = 0
            while time.monotonic() - start < block_seconds:
                if mode == 'busy':
                    torch.mm(a, b, out=out)
                    torch.cuda.synchronize()
                    iterations += 1
                else:
                    time.sleep(min(.01, max(0, block_seconds - (time.monotonic()-start))))
            blocks.append(dict(mode=mode, start_monotonic=start,
                               stop_monotonic=time.monotonic(), iterations=iterations))
    finally:
        torch.cuda.synchronize()
        monitor.stop()
    readings = monitor.readings
    valid = [r for r in readings if r.power_mw is not None]
    changes = [r.timestamp for a, r in zip(valid, valid[1:]) if a.power_mw != r.power_mw]
    intervals = [b-a for a, b in zip(changes, changes[1:])]
    windows = []
    for block in blocks:
        middle = [r.power_mw/1000 for r in valid
                  if block['start_monotonic']+.5 <= r.timestamp < block['stop_monotonic']-.2]
        windows.append(dict(block, middle_mean_power_w=statistics.mean(middle) if middle else None))
    return dict(scope='synthetic matmul sensor-response diagnostic, not Full/Fast/Continue service power',
                power_boundary='INA3221 VDD_IN module input', poll_interval_seconds=poll_interval,
                blocks=windows, samples=[asdict(r) for r in readings],
                missing_samples=len(readings)-len(valid), observed_value_changes=len(changes),
                median_observed_change_interval_seconds=statistics.median(intervals) if intervals else None,
                warning='Observed value changes do not establish hardware update period or response latency',
                hardware_writes=False)


def discovery(config):
    return dict(status='discovery_completed', device=inventory(),
                fast_interface=inspect_fast_interface(config['flashrt_root']),
                baseline_id=config['baseline_id'], full_service_measurements_started=False,
                clock_settings_changed=False)


def select_images(snapshot, view):
    """Do not zero or mask the discarded image; remove it from the input list."""
    if view not in ('full', 'agent', 'wrist'):
        raise ValueError('Unknown view')
    if len(snapshot['images']) != 2:
        raise ValueError('Expected the original two camera inputs')
    result = dict(snapshot)
    result['images'] = (list(snapshot['images']) if view == 'full'
                        else [snapshot['images'][0 if view == 'agent' else 1]])
    return result


def fixed_view_probe(config, directory, source_run, kind, repeats=5, *, clock_screen=False):
    """Prepared-input Full/Fast RTC compute probe. No environment or clock writes.

    One model load for Full geometry, one for single-view geometry; agent/wrist
    reuse the same single-view model and graph. Not an end-to-end service L or
    a quality test: old target and delay are fixed to isolate compute cost.
    """
    import json
    import os
    import numpy as np
    from scripts import eval_q as base
    from perceptbudget.h50_checkpoint import image_input, normalize_state
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.power import _integrate_report

    if kind not in ('full', 'single') or not 1 <= repeats <= 20:
        raise ValueError('Invalid fixed-view probe configuration')
    config = dict(config, m2_profile=True, m2_view='full' if kind == 'full' else 'agent',
                  rtc_audit_inputs=False, rtc_audit_attention_same_input=False,
                  rtc_audit_fused_norm=False)
    directory, source_run = Path(directory), Path(source_run)
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch, _ = base._check_cuda_and_native(config)
    torch.use_deterministic_algorithms(True)
    task = base._load_tasks([0], ['libero_spatial'])[0]
    source_config = json.loads((source_run/'run.json').read_text())['config_snapshot']
    for key in ('H', 'N', 'checkpoint_provenance_revision', 'state_enabled'):
        if config[key] != source_config[key]:
            raise ValueError('Fixed input source mismatch: '+key)
    requests = json.loads((source_run/'rtc/q5_requests.json').read_text())
    first = requests[0]
    with np.load(source_run/'rtc/q5_frames.npz') as saved:
        frame = saved['frames'][0]
    if frame.shape != (256, 512, 3) or first['observation_step'] != 0:
        raise ValueError('Unexpected source frame or observation step')
    stats = load_libero_norm_stats(config['norm_stats_path'])
    state = np.asarray(first['observation_raw_state'], dtype=np.float32)
    snapshot = dict(images=[image_input(frame[:, :256]), image_input(frame[:, 256:])],
                    raw_state=state, state=normalize_state(state, stats))
    previous = np.asarray(first['actions_raw'], dtype=np.float32)[5:]
    if previous.shape != (45, 32):
        raise ValueError('Expected H50 fixed target after five executed steps')
    start_load = time.monotonic()
    model = base._load_and_check_model(config, torch)
    from flash_rt.subgraphs.pi05.context_action import enable
    enable(model)
    views = ('full',) if kind == 'full' else ('agent', 'wrist')
    initial = select_images(snapshot, views[0])
    base._infer_actions(model, initial, task['instruction'], 50, {})
    cleanup = release_unused_native_expert(model, torch)
    adapter = OriginalOpenPiRTCAdapter(model, torch, config)
    gate = json.loads((Path(config['verified_reference_run'])/'summary.json').read_text())
    if not gate.get('passed') or adapter.expert.provenance != gate['original_source_hashes']:
        raise RuntimeError('Frozen original-source gate mismatch')
    adapter.bootstrap(initial, task['instruction'], {})
    valid = int(adapter.expert.prefix_mask.shape[1])
    audit = adapter.prepare_step_bank([valid])
    pl = model._pipe.pipeline
    expected_views = 2 if kind == 'full' else 1
    if pl.num_views != expected_views or pl.vision_seq != expected_views * 256:
        raise RuntimeError('View count did not reduce the actual pipeline geometry')
    expected_pool=config.get('m2_vision_pool_factor',1)
    from perceptbudget.mixed_view_pool import visual_tokens
    tokens=visual_tokens(config) if kind=='full' else 256
    if pl.vision_pool_factor != expected_pool or pl.vision_seq_enc != tokens:
        raise RuntimeError('Requested token pooling did not change encoder geometry')
    pool_layout=None
    if expected_pool>1:
        from scripts.diagnose_pool_parity import pool_check
        pool_layout=pool_check(pl,torch,np,config.get('m2_pool_view','both'))
        if pool_layout['mismatched_elements']:
            raise RuntimeError('Pooled token CPU layout audit failed')
    result = dict(status='running', kind=kind, source_run=str(source_run),
        scope='prepared-input context plus complete RTC VJP; not service L, success rate or dynamic energy saving',
        model_loads=1, load_and_prepare_seconds=time.monotonic()-start_load,
        input_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
        old_target_sha256=hashlib.sha256(previous.tobytes()).hexdigest(),
        fixed_delay_steps=5, fixed_executed_steps=5,
        num_views=pl.num_views, visual_tokens=pl.vision_seq_enc,
        visual_tokens_before_pool=pl.vision_seq, vision_pool_factor=pl.vision_pool_factor,
        encoder_capacity=pl.encoder_seq_len, valid_prefix=valid,
        graph_audit=audit, native_expert_release=cleanup, measurements=[],
        pooled_view=config.get('m2_pool_view','both'),pool_layout_audit=pool_layout,
        clock_settings_changed=False, flashrt_modified=False)
    if clock_screen:
        if kind != 'full':
            raise ValueError('Clock screening requires unmodified Full views')
        from perceptbudget.device_clock_screen import screen
        return screen(adapter, snapshot, task['instruction'], previous, torch,
                      directory, result)
    for view in views:
        image_input_snapshot = select_images(snapshot, view)
        for _ in range(2):
            adapter.predict(image_input_snapshot, task['instruction'], previous,
                            delay=5, executed=5, request={})
        monitor = PowerMonitor(poll_interval=.02)
        rows = []
        monitor.start()
        started = time.monotonic()
        try:
            for i in range(repeats):
                torch.manual_seed(20261021+i)
                request = {}
                begin = time.monotonic()
                actions = adapter.predict(image_input_snapshot, task['instruction'], previous,
                                          delay=5, executed=5, request=request)
                rows.append(dict(milliseconds=(time.monotonic()-begin)*1000,
                                 guidance=request.get('rtc_guidance'),
                                 finite_actions=bool(np.isfinite(actions).all()),
                                 action_sha256=hashlib.sha256(actions.tobytes()).hexdigest()))
                if request.get('rtc_guidance', {}).get('variable_length_fallback'):
                    raise RuntimeError('Unexpected slow fallback in fixed-view probe')
            stop = time.monotonic()
        finally:
            monitor.stop()
        power = _integrate_report(monitor.readings, started, stop, max_gap_seconds=.1)
        result['measurements'].append(dict(view=view, requests=rows,
            median_ms=statistics.median(r['milliseconds'] for r in rows),
            mean_ms=statistics.mean(r['milliseconds'] for r in rows),
            aggregate_energy_j=power.total_energy_mj/1000 if power.total_energy_mj is not None else None,
            average_energy_per_request_j=power.total_energy_mj/1000/repeats if power.total_energy_mj is not None else None,
            power_coverage=asdict(power), raw_power=[asdict(r) for r in monitor.readings]))
        (directory/f'{kind}_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    result['status'] = 'completed'
    (directory/f'{kind}_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result
