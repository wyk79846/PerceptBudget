"""Small paired Full/Fast screen, resident model per pipeline geometry."""
import copy
import gc
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

SUITES = ('libero_spatial', 'libero_object', 'libero_goal')


def calibration_plan(kind):
    views = ('full',) if kind == 'full' else ('agent', 'wrist')
    return [(suite, 0, view) for suite in SUITES for view in views]


def select_candidate(full, single):
    """Only task0 calibration results; holdout results cannot affect selection."""
    def passed(rows, view):
        eligible = [r for r in rows if r['view'] == view and r['task_id'] == 0]
        return (len(eligible) == 3 and {r['suite'] for r in eligible} == set(SUITES)
                and all(r['outcome'] == 'success' and r['audit_passed']
                        and r['underflow_steps'] == 0 for r in eligible))
    if not passed(full, 'full'):
        return None
    # Predeclared tie-break, not tuned to holdout outcomes or five timing samples.
    return next((v for v in ('agent', 'wrist') if passed(single, v)), None)


def resident_quality(config, directory, kind):
    import numpy as np
    from scripts import eval_q as base, probe_delay_buffer as probe
    from scripts.complete_m1_h50 import cleanup, calibrate, initial_info
    from scripts.run_m1_h50 import compare_initial
    from scripts.audit_rtc_system import audit
    from perceptbudget.device_profile import select_images
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import memory_record, release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.rtc_source_step_graph import h50_state_prefix_lengths
    from perceptbudget.measurement import timed
    from perceptbudget.experiment import create_run, _write_json_atomically

    if kind not in ('full', 'single'):
        raise ValueError('Unknown quality geometry')
    directory = Path(directory)
    cfg = copy.deepcopy(config)
    cfg.update(m2_profile=True, m2_view='full' if kind == 'full' else 'agent',
        candidate_q=[5], disk_backed_frames=True, rtc_audit_inputs=True,
        release_rtc_graph_before_artifacts=False,
        execution_mode='m2_fast_quality_actual_async', diagnostic_condition='actual_history')
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    path = directory/f'{kind}_quality.json'
    if path.exists():
        raise ValueError('Quality worker output already exists; do not overwrite episodes')
    record = dict(status='loading', kind=kind, episodes=[], model_loads=0, suffix_loads=0,
        graph_audits=[], memory=[], q=5, selection=None,
        scope='task0 calibration; task3 holdout; seed7/init0; no clock changes',
        holdout_policy='Full task3 collected early to avoid reloading, never used to select a camera')
    def save():
        _write_json_atomically(path, record)
    save()
    torch, _ = base._check_cuda_and_native(cfg)
    torch.use_deterministic_algorithms(True)
    base._patch_egl_cleanup()
    def mem(stage):
        cleanup(torch)
        value = memory_record(torch, stage)
        record['memory'].append(value)
        save()
        if value['available_mib'] < 900:
            raise RuntimeError('Resident quality RAM headroom below900MiB; stop without per-episode reload')
    stats = load_libero_norm_stats(cfg['norm_stats_path'])
    model = adapter = root_expert = env = active = None
    original_snapshot = base._make_snapshot
    current_view = cfg['m2_view']
    def snapshot(observation, stats, state_enabled):
        return select_images(original_snapshot(observation, stats, state_enabled), current_view)
    try:
        with patch.object(base, '_make_snapshot', snapshot):
            mem('before_model_load')
            model = base._load_and_check_model(cfg, torch)
            record['model_loads'] += 1
            from flash_rt.subgraphs.pi05.context_action import enable
            enable(model)
            _, env_class = base._libero_api()
            cohorts = (0, 3) if kind == 'full' else (0,)
            # single-view holdout is appended only after a calibration-only decision.
            pending = [(suite, tid) for tid in cohorts for suite in SUITES]
            while pending:
                suite, tid = pending.pop(0)
                if adapter is not None:
                    torch.cuda.synchronize()
                    adapter.step_bank = None
                    adapter.expert = root_expert
                    mem('old_bank_released')
                base._seed_process(7, np, torch)
                task = base._load_tasks([tid], [suite])[0]
                env = base._make_env(env_class, task)
                env._pb_lock_fixtures = True
                observation, calibration = calibrate(env, task, np)
                views = (('full',) if kind == 'full' else
                         ('agent', 'wrist') if tid == 0 else (record['selection'],))
                current_view = views[0]
                prepared = snapshot(observation, stats, True)
                cfg.update(rtc_suite=suite)
                cfg['screen'].update(suites=[suite], task_ids=[tid])
                if adapter is None:
                    base._infer_actions(model, prepared, task['instruction'], 50, {})
                    expected_views = 2 if kind == 'full' else 1
                    if model._pipe.pipeline.num_views != expected_views or \
                       model._pipe.pipeline.vision_seq != 256*expected_views:
                        raise RuntimeError('Selected perception geometry not active')
                    record['native_release'] = release_unused_native_expert(model, torch)
                    adapter = OriginalOpenPiRTCAdapter(model, torch, cfg)
                    root_expert = adapter.expert
                    record['suffix_loads'] += 1
                    gate = json.loads((Path(cfg['verified_reference_run'])/'summary.json').read_text())
                    if not gate.get('passed') or root_expert.provenance != gate['original_source_hashes']:
                        raise RuntimeError('Frozen source gate mismatch')
                adapter.config = cfg
                first = {}
                adapter.bootstrap(prepared, task['instruction'], first)
                num_views = 2 if kind == 'full' else 1
                lengths = h50_state_prefix_lengths(first['rtc_guidance']['valid_prefix_length'],
                                                   task['instruction'], num_views=num_views)
                cfg['source_step_prefixes'] = lengths
                graph = adapter.prepare_step_bank(lengths, progress=lambda r: mem('prefix_'+str(r['valid'])))
                record['graph_audits'].append(dict(suite=suite, task_id=tid, **graph))
                previous = np.asarray(first['actions_raw'], dtype=np.float32)[5:]
                timing = timed(torch, lambda: adapter.predict(prepared, task['instruction'], previous,
                                       delay=5, executed=5, request={}), count=3)
                cfg['rtc_initial_delay_steps'] = math.ceil(timing['median_ms']/1000/calibration['median_seconds'])
                cfg['rtc_initial_control_period_seconds'] = calibration['median_seconds']
                cfg['rtc_initial_control_period_source'] = 'same-environment preflight12 ticks, then reset'
                for view in views:
                    current_view = view
                    child_cfg = copy.deepcopy(cfg)
                    child_cfg['m2_view'] = view
                    adapter.config = child_cfg
                    active = create_run(child_cfg, 'M2', 'fast-quality-'+view)
                    trace_dir = active/'rtc'
                    trace_dir.mkdir()
                    _write_json_atomically(active/'control_period_calibration.json', calibration)
                    mem('before_'+suite+str(tid)+view)
                    with (trace_dir/'trace.jsonl').open('w') as stream:
                        episode = probe.run_episode(env, task, 5, model, stats, child_cfg, np, torch,
                            trace_dir, stream, rtc_adapter=adapter, condition='m2_'+view,
                            refresh_policy='latest_due', pending_arm_gain=1.)
                    _write_json_atomically(active/'summary.json', dict(status='validating', episode=episode))
                    checked = audit(active)
                    _write_json_atomically(active/'system_audit.json', checked)
                    if not checked['passed']:
                        raise RuntimeError('Quality episode independent record audit failed')
                    requests = json.loads((trace_dir/'q5_requests.json').read_text())
                    if any(r['rtc_guidance'].get('variable_length_fallback') for r in requests):
                        raise RuntimeError('Unexpected slow fallback in quality screen')
                    result = dict(status='completed', episode=episode, audit_passed=True,
                        view=view, suite=suite, task_id=tid,
                        steady_inference_median_ms=float(np.median([
                            (r['inference_ended_monotonic']-r['inference_started_monotonic'])*1000
                            for r in requests[1:]])) if len(requests)>1 else None)
                    info = initial_info(active, episode)
                    review = None
                    if kind == 'single':
                        full = json.loads((directory/'full_quality.json').read_text())
                        reference = next(r['initial'] for r in full['episodes']
                                         if r['suite']==suite and r['task_id']==tid)
                        review = compare_initial(reference, info)
                    result['initial_pair_review'] = review
                    _write_json_atomically(active/'summary.json', result)
                    metadata = json.loads((active/'run.json').read_text())
                    metadata['status'] = 'completed'
                    _write_json_atomically(active/'run.json', metadata)
                    record['episodes'].append(dict(suite=suite, task_id=tid, view=view,
                        run=str(active), outcome=episode['status'], steps=episode['steps'],
                        wall_seconds=episode['real_seconds'], audit_passed=True,
                        underflow_steps=len(episode['underflow_steps']), initial=info,
                        initial_pair_review=review, inference_median_ms=result['steady_inference_median_ms']))
                    record['status'] = 'running'
                    save()
                    print('quality_episode='+json.dumps(record['episodes'][-1]), flush=True)
                    del episode, requests, result
                    active = None
                    mem('episode_released')
                env.close()
                env = None
                mem('environment_closed')
                if kind == 'single' and tid == 0 and not pending:
                    full = json.loads((directory/'full_quality.json').read_text())
                    record['selection'] = select_candidate(full['episodes'], record['episodes'])
                    save()
                    if record['selection']:
                        pending.extend((s, 3) for s in SUITES)
            record['status'] = 'completed'
            save()
    except BaseException as exc:
        record.update(status='invalid', error=f'{type(exc).__name__}: {exc}', active_run=str(active) if active else None)
        save()
        if active:
            _write_json_atomically(active/'invalid.json', dict(error=record['error']))
        raise
    finally:
        if env is not None:
            env.close()
        if adapter is not None:
            adapter.release_graph()
    return record
