"""Small paired Full/Fast screen, resident model per pipeline geometry."""
import copy
import gc
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

SUITES = ('libero_spatial', 'libero_object', 'libero_goal')


def audit_paired_bootstraps(episodes):
    """Require identical starting inputs and outputs, not identical future trajectories."""
    import numpy as np
    cases=[]
    for suite,tid in sorted({(e['suite'],e['task_id']) for e in episodes}):
        pair=[e for e in episodes if e['suite']==suite and e['task_id']==tid]
        if len(pair) not in (2,3,4):
            raise ValueError('Require two, three or four resident episodes per task')
        requests=[json.loads((Path(e['run'])/'rtc/q5_requests.json').read_text())[0] for e in pair]
        for i in range(1,len(pair)):
            checks={'initial_hashes_equal':pair[0]['initial']['initial_hashes']==pair[i]['initial']['initial_hashes']}
            for field in ('observation_raw_state','model_state','actions_raw','actions'):
                checks[field+'_equal']=np.array_equal(requests[0][field],requests[i][field])
            checks['noise_equal']=np.array_equal(requests[0]['rtc_solver_inputs']['noise'],requests[i]['rtc_solver_inputs']['noise'])
            cases.append(dict(suite=suite,task_id=tid,reference_run=pair[0]['run'],candidate_run=pair[i]['run'],
                passed=all(checks.values()),checks=checks,
                raw_max_abs=float(np.max(np.abs(np.asarray(requests[0]['actions_raw'])-np.asarray(requests[i]['actions_raw']))))))
    return dict(passed=bool(cases) and all(c['passed'] for c in cases),cases=cases,
        scope='Exact starting state/images/noise/actions only; real async timing and policy-dependent future inputs may diverge')


def quality_run_purpose(label):
    # Keep the semantic episode/view label intact; only shorten the directory tag.
    short = {'pool2_agent_alternate': 'pool2_agent_alt'}.get(label, label)
    return 'fast-quality-' + short


def install_render_buffer_cleanup(*, shared_egl=False):
    """Release MuJoCo render buffers, retain the upstream EGL-destroy workaround.

    Process-local M2 diagnostic only. Call after base._patch_egl_cleanup().
    Does not change resets, simulation, observation processing or model math.
    """
    import robosuite.utils.binding_utils as binding
    import threading
    audit = dict(freed=0, errors=[], egl_destroy_enabled=False,
                 shared_egl=shared_egl, egl_contexts_created=0, explicit_sim_frees=0)
    if shared_egl:
        original_factory = binding.GLContext
        cache = {}
        owner = threading.get_ident()
        def context_factory(*, max_width, max_height, device_id):
            if threading.get_ident() != owner:
                raise RuntimeError('M2 shared EGL is sequential and main-thread only')
            if cache and device_id not in cache:
                raise RuntimeError('Cannot share EGL across different GPU devices')
            if device_id not in cache:
                cache[device_id] = original_factory(max_width=max_width, max_height=max_height,
                                                    device_id=device_id)
                audit['egl_contexts_created'] += 1
            return cache[device_id]
        binding.GLContext = context_factory
    def free_buffers(context):
        if getattr(context, '_pb_buffers_freed', False):
            return
        try:
            gl = getattr(context, 'gl_ctx', None)
            con = getattr(context, 'con', None)
            if gl is not None and con is not None:
                gl.make_current()
                con.free()
                context._pb_buffers_freed = True
                audit['freed'] += 1
        except Exception as exc:
            audit['errors'].append(f'{type(exc).__name__}: {exc}')
    binding.MjRenderContext.__del__ = free_buffers
    # A render context retains its sim and may remain referenced by old sensor
    # closures during hard reset. Waiting for __del__/GC allows old and new
    # GPU render buffers to overlap. Release at the explicit sim-free boundary.
    previous_free = binding.MjSim.free
    def free_sim(sim):
        context = getattr(sim, '_render_context_offscreen', None)
        if context is not None:
            free_buffers(context)
            audit['explicit_sim_frees'] += 1
        return previous_free(sim)
    binding.MjSim.free = free_sim
    return audit


def reset_reuse_audit(config, directory, *, render_cleanup=False, shared_egl=False):
    """No model: verify constructor-preserving resets on all six screen tasks."""
    import hashlib
    import time
    import numpy as np
    from scripts import eval_q as base, probe_delay_buffer as probe
    from perceptbudget.libero_input import raw_libero_state
    from perceptbudget.experiment import _write_json_atomically
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch, _ = base._check_cuda_and_native(config)
    base._patch_egl_cleanup()
    cleanup_record = None
    _, cls = base._libero_api()
    batch = Path(config['output_root'])/'M1/20261020T160525.143512Z-m1-h50-67126d0b'
    index = json.loads((batch/'index.json').read_text())
    result = dict(status='running', passed=False, cases=[], scope='reset-only replay; no model/clock changes')
    def frame(obs):
        return np.concatenate([obs['agentview_image'], obs['robot0_eye_in_hand_image']], axis=1)
    for tid in (0, 3):
        for suite in SUITES:
            task = base._load_tasks([tid], [suite])[0]
            source = Path(index['cells'][f'{suite}_task{tid:02d}_q5_actual_history']['result']['run'])
            actions = [json.loads(line)['action'] for line in (source/'rtc/trace.jsonl').open()
                       if '"control_step"' in line][:20]
            captures = []
            for candidate in (False, True):
                soft = candidate and not render_cleanup
                if candidate and render_cleanup and cleanup_record is None:
                    cleanup_record = install_render_buffer_cleanup(shared_egl=shared_egl)
                base._seed_process(7, np, torch)
                env = base._make_env(cls, task)
                env._pb_lock_fixtures = True
                if soft:
                    env.env.hard_reset = False
                try:
                    # Same calibration resets and 12 zero controls as quality runner.
                    from scripts.complete_m1_h50 import calibrate
                    observation, _ = calibrate(env, task, np)
                    observation, _ = base._prepare_env_episode(env, task, np)
                    rows = []
                    for action in [None, *actions]:
                        if action is not None:
                            observation, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
                        rows.append(dict(state=env.env.sim.get_state().flatten().copy(),
                            raw=raw_libero_state(observation).copy(), image=frame(observation).copy(),
                            contacts=probe.snapshot_scene(env, observation)['contacts'],
                            success=bool(env.check_success())))
                    captures.append(rows)
                finally:
                    env.close()
                    gc.collect()
            states = max(float(np.max(np.abs(a['state']-b['state']))) for a,b in zip(*captures))
            raw = max(float(np.max(np.abs(a['raw']-b['raw']))) for a,b in zip(*captures))
            pixels = [np.abs(a['image'].astype(np.int16)-b['image'].astype(np.int16))
                      for a,b in zip(*captures)]
            changed = max(int(np.any(d!=0, axis=-1).sum()) for d in pixels)
            maximum = max(int(d.max()) for d in pixels)
            contact = all(a['contacts']==b['contacts'] and a['success']==b['success']
                          for a,b in zip(*captures))
            row = dict(suite=suite, task_id=tid, replay_steps=len(actions), state_max_abs=states,
                       raw_state_max_abs=raw, max_changed_pixels=changed, max_channel_delta=maximum,
                       contact_and_success_equal=contact,
                       passed=states==0 and raw==0 and changed<=4 and maximum<=1 and contact)
            result['cases'].append(row)
            result['candidate'] = ('hard_reset_shared_egl' if shared_egl else
                                   'hard_reset_render_buffer_cleanup' if render_cleanup else 'soft_reset')
            result['render_cleanup'] = cleanup_record
            _write_json_atomically(Path(directory)/'reset_reuse_audit.json', result)
            print('reset_audit='+json.dumps(row), flush=True)
            if not row['passed']:
                result['status'] = 'failed_gate'
                _write_json_atomically(Path(directory)/'reset_reuse_audit.json', result)
                return result
    result.update(status='completed', passed=True)
    _write_json_atomically(Path(directory)/'reset_reuse_audit.json', result)
    return result


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


def resident_quality(config, directory, kind, *, resume=False, cleanup_gate=None, calibration_only=False, task_plan=None, recompute_comparison=False, split_delay_comparison=False, single_insertion=False, paired_split=False, visual_comparison=False, visual_fast_mode='wrist', record_only=False):
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

    explicit_visual=bool(config.get('m4_enabled') or config.get('m3_fast_enabled'))
    dual_visual=visual_comparison or explicit_visual
    if explicit_visual:visual_fast_mode='selective416' if config.get('m3_fast_selective416') else 'pool2_agent'
    if kind not in ('full', 'single'):
        raise ValueError('Unknown quality geometry')
    if record_only and (kind != 'full' or not task_plan or resume or recompute_comparison or visual_comparison
                        or config.get('m2_vision_pool_factor', 1) != 1):
        raise ValueError('M3 recording requires new explicit Full-view tasks without perception changes')
    dvac_paired=bool(config.get('m3_dvac_comparison'))
    if dvac_paired and (not record_only or config.get('m3_scheduler_mode') not in ('deadline_high','deadline_dvfs')):
        raise ValueError('DVAC comparison requires scheduled M3 recording')
    if visual_comparison and (kind!='full' or not task_plan or resume or recompute_comparison
                             or split_delay_comparison or single_insertion or paired_split):
        raise ValueError('Fresh visual comparison requires an isolated Full-geometry task plan')
    if visual_fast_mode not in ('wrist','pool2','pool2_agent','selective416'):
        raise ValueError('Unsupported visual Fast mode')
    if split_delay_comparison and not recompute_comparison:
        raise ValueError('Split-delay comparison requires expert recompute comparison')
    if single_insertion and not (recompute_comparison and split_delay_comparison):
        raise ValueError('Single insertion requires explicit per-mode-delay recompute comparison')
    if paired_split and (single_insertion or not (recompute_comparison and split_delay_comparison)):
        raise ValueError('Paired split requires alternating per-mode-delay comparison')
    if recompute_comparison and (kind!='full' or task_plan is None or resume or
        config.get('m2_vision_pool_factor',1)!=1):
        raise ValueError('Expert recompute comparison requires Full geometry and explicit new tasks')
    if task_plan is not None:
        allowed_suites=set(SUITES)|({'libero_10'} if record_only else set())
        if kind != 'full' or resume or not task_plan or len(set(task_plan)) != len(task_plan) or any(
                s not in allowed_suites or type(t) is not int or not 0<=t<10 for s,t in task_plan):
            raise ValueError('Custom quality plan requires unique Full-view tasks and a new run')
    directory = Path(directory)
    cfg = copy.deepcopy(config)
    cfg.update(m2_profile=True, m2_view='full' if kind == 'full' else 'agent',
        candidate_q=[5], disk_backed_frames=True, rtc_audit_inputs=True,
        release_rtc_graph_before_artifacts=False,
        execution_mode='m2_fast_quality_actual_async', diagnostic_condition='actual_history')
    if recompute_comparison:
        from perceptbudget.rtc_kv_reuse import RecomputeRTCAdapter
        OriginalOpenPiRTCAdapter=RecomputeRTCAdapter
        cfg.update(m2_expert_recompute=True,m2_reuse_enabled=False)
    if dual_visual:
        from perceptbudget.rtc_visual_alternation import FreshViewRTCAdapter as OriginalOpenPiRTCAdapter
        cfg.update(m2_expert_recompute=False, m2_visual_alternation=True,
                   m2_visual_fast_enabled=False, m2_visual_fast_mode=visual_fast_mode,
                   m2_delay_modes=['full',visual_fast_mode])
    if explicit_visual:
        from perceptbudget.m4_visual import M4ViewRTCAdapter as OriginalOpenPiRTCAdapter
        cfg['m2_fast_no_state']=True
    from perceptbudget.model import vision_pool_factor
    pool_factor = vision_pool_factor(cfg)
    from perceptbudget.mixed_view_pool import visual_tokens
    expected_tokens=visual_tokens(cfg) if kind=='full' else 256
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    path = directory/('recording.json' if record_only else f'{kind}_quality.json')
    if path.exists() and not resume:
        raise ValueError('Quality worker output already exists; do not overwrite episodes')
    record = dict(status='loading', kind=kind, episodes=[], model_loads=0, suffix_loads=0,
        graph_audits=[], memory=[], q=5, selection=None,
        scope='task0 calibration; task3 holdout; seed7/init0; no clock changes',
        holdout_policy='Full task3 collected early to avoid reloading, never used to select a camera')
    if cfg.get('m3_margin_scan'):
        record.update(margin_long_run=cfg.get('m3_margin_long_run',False),margin_round=cfg.get('m3_margin_round',1))
    if cfg.get('m3_resume_plan'):
        continuation=cfg.pop('m3_resume_plan')
        record.update(episodes=copy.deepcopy(continuation['episodes']),resume_source=continuation['source'],
            previous_model_loads=continuation['previous_model_loads'],
            retested_original_episodes=continuation.get('retested_original_episodes',[]))
    if cfg.get('m3_formal_comparison'):
        record['formal_trigger_strategy']=cfg['m3_formal_signal']
    if resume and path.exists():
        record = json.loads(path.read_text())
        for row in record['episodes']:
            if not audit(Path(row['run']))['passed']:
                raise ValueError('Saved episode failed independent resume audit')
        _write_json_atomically(directory/f'{kind}_before_resume_{__import__("time").time_ns()}.json', record)
        record.setdefault('restarts', []).append(dict(reason='exception continuation; completed episodes retained',
                                                     previous_error=record.get('error')))
        record.update(status='loading', error=None, active_run=None)
    if cleanup_gate:
        gate_record = json.loads((Path(cleanup_gate)/'reset_reuse_audit.json').read_text())
        if not gate_record.get('passed') or gate_record.get('candidate') not in ('hard_reset_render_buffer_cleanup', 'hard_reset_shared_egl') \
           or len(gate_record['cases']) != 6 or gate_record['render_cleanup']['errors']:
            raise ValueError('Require six-task passed hard-reset cleanup gate')
        record['cleanup_gate'] = str(cleanup_gate)
    record['calibration_only'] = calibration_only
    if task_plan is not None:
        record.update(task_plan=list(task_plan), vision_pool_factor=pool_factor,
            pooled_view=cfg.get('m2_pool_view','both'),visual_tokens=expected_tokens,
            scope='fixed dual-view pooled perception across requested tasks; not Full/Fast switching',
            holdout_policy='No calibration selection or extra holdout; execute the explicit task plan only')
    if record_only:
        scheduled = cfg.get('m3_scheduler_mode') in ('deadline_high','deadline_dvfs')
        cfg.update(m3_record_only=not scheduled,
                   execution_mode='m3_'+cfg.get('m3_scheduler_mode','record_only')+'_actual_async')
        record.update(scope='M3 Full baseline recording; q5 actual async RTC; action hints reviewed after each episode; no scheduling or clock changes',
                      holdout_policy='Explicit calibration tasks only', mode='deadline_high' if scheduled else 'record_only')
        if scheduled:
            record['scope'] = 'M3 event/buffer/age deadlines; high-frequency Full/Continue; actual async RTC; no clock changes'
            if cfg['m3_scheduler_mode']=='deadline_dvfs':
                record.update(scope='M3 deadlines with low-frequency Continue and early high-frequency Full recovery',mode='deadline_dvfs')
    if recompute_comparison:
        record.update(recompute_comparison=True,
            scope='paired Full vs alternating stale-KV expert-only RTC; no spatial compression',
            reuse_contract='bootstrap Full; at most one expert-only request after each Full; execution before original H-step nominal expiry')
    if dual_visual:
        from perceptbudget.rtc_visual_alternation import episode_geometry
        record.update(visual_comparison=True,fast_visual_mode=visual_fast_mode,
            scope='Paired Full vs alternating fresh '+visual_fast_mode+' contexts; complete RTC, no stale KV',
            **episode_geometry(visual_fast_mode,True))
    if cfg.get('m4_enabled'):
        record.update(scope='M4 explicit per-request Full/Fast and joint state selection; actual async RTC',
            m4_enabled=True,visual_comparison=False)
    expected = {(s, 0, v) for s, _, v in calibration_plan(kind)}
    completed = {(r['suite'], r['task_id'], r['view']) for r in record['episodes']}
    if calibration_only and expected <= completed:
        record['status'] = 'completed'
        _write_json_atomically(path, record)
        return record
    def save():
        _write_json_atomically(path, record)
    save()
    torch, _ = base._check_cuda_and_native(cfg)
    torch.use_deterministic_algorithms(True)
    base._patch_egl_cleanup()
    if cleanup_gate:
        record['render_buffer_cleanup'] = install_render_buffer_cleanup(
            shared_egl=gate_record.get('candidate') == 'hard_reset_shared_egl')
    setup_started=time.monotonic()
    def stage_progress(stage):
        row=dict(stage=stage,elapsed_seconds=round(time.monotonic()-setup_started,3))
        record.setdefault('stage_timings',[]).append(row)
        print('resident_stage='+json.dumps(row),flush=True)
        save()
    def mem(stage):
        cleanup(torch)
        value = memory_record(torch, stage)
        value['elapsed_seconds']=time.monotonic()-setup_started
        print('resident_stage='+json.dumps(dict(stage=stage,elapsed_seconds=round(value['elapsed_seconds'],3))),flush=True)
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
            cohorts = (0, 3) if kind == 'full' and not calibration_only else (0,)
            # single-view holdout is appended only after a calibration-only decision.
            pending = [(suite, tid) for tid in cohorts for suite in SUITES]
            if task_plan is not None:
                pending = list(task_plan)
            if kind == 'single' and record.get('selection') and not calibration_only:
                pending.extend((suite, 3) for suite in SUITES)
            while pending:
                suite, tid = pending.pop(0)
                views = (('full',) if kind == 'full' else
                         ('agent', 'wrist') if tid == 0 else (record['selection'],))
                if recompute_comparison:
                    views=('full','kv_once') if single_insertion else ('full','kv_reuse_split') if paired_split else ('full','kv_reuse','kv_reuse_split') if split_delay_comparison else ('full','kv_reuse')
                if dual_visual:
                    views=('full',visual_fast_mode+'_alternate')
                if explicit_visual:views=('full',)
                if dvac_paired:
                    from perceptbudget.closed_loop import comparison_variants
                    if cfg.get('m3_formal_comparison'):
                        cfg['m3_formal_task_index']=list(task_plan).index((suite,tid))
                    if cfg.get('m3_trigger_comparison') and task_plan is not None:
                        cfg['m3_trigger_task_index']=list(task_plan).index((suite,tid))
                    views=comparison_variants(cfg)
                done = {(r['suite'], r['task_id'], r['view']) for r in record['episodes']}
                views = tuple(v for v in views if (suite, tid, v) not in done)
                if not views:
                    continue
                if adapter is not None:
                    torch.cuda.synchronize()
                    adapter.step_bank = None
                    adapter.expert = root_expert
                    mem('resident_banks_retained' if dual_visual else 'old_bank_released')
                base._seed_process(7, np, torch)
                task = base._load_tasks([tid], [suite])[0]
                env = base._make_env(env_class, task)
                env._pb_lock_fixtures = True
                stage_progress('environment_created')
                observation, calibration = calibrate(env, task, np)
                stage_progress('environment_calibrated')
                prepared_start=(dict(observation=observation,physics=env.env.sim.get_state().flatten().copy(),
                                     suite=task['suite'],task_id=task['task_id'])
                    if (cfg.get('m4_acceptance_group')=='joint' or
                        (cfg.get('m4_acceptance_group')=='m3' and cfg.get('experimental_setup_reuse',False))) else None)
                current_view = 'full' if dvac_paired else views[0]
                prepared = snapshot(observation, stats, True)
                cfg.update(rtc_suite=suite)
                cfg['screen'].update(suites=[suite], task_ids=[tid])
                if adapter is None:
                    base._infer_actions(model, prepared, task['instruction'], 50, {})
                    expected_views = 2 if kind == 'full' else 1
                    if model._pipe.pipeline.num_views != expected_views or \
                       model._pipe.pipeline.vision_seq != 256*expected_views or \
                       model._pipe.pipeline.vision_seq_enc != expected_tokens:
                        raise RuntimeError('Selected perception geometry not active')
                    if dual_visual:
                        from perceptbudget.shared_visual_context import wrist_frontend
                        from flash_rt.api import VLAModel
                        wrist_model = VLAModel(wrist_frontend(model._pipe, torch, fast_kind=visual_fast_mode,
                            no_state=cfg.get('m2_fast_no_state',False),
                            no_state_prompt_capacity=cfg.get('m4_fast_prompt_capacity')), 'torch')
                        enable(wrist_model)
                        fast_prepared = select_images(prepared, 'wrist' if visual_fast_mode=='wrist' else 'full')
                        if cfg.get('m2_fast_no_state',False):
                            fast_prepared['state'] = None
                        base._infer_actions(wrist_model, fast_prepared, task['instruction'], 50, {})
                        if visual_fast_mode.startswith('pool2'):
                            from scripts.diagnose_pool_parity import pool_check
                            layout=pool_check(wrist_model._pipe.pipeline,torch,np,'agent' if visual_fast_mode=='pool2_agent' else 'both')
                            record['fast_pool_layout']=layout
                            if layout['mismatched_elements']:
                                raise RuntimeError('Shared pooled context layout mismatch')
                        if not all(model._pipe._ckpt_bf16[k] is wrist_model._pipe._ckpt_bf16[k]
                                   for k in model._pipe._ckpt_bf16):
                            raise RuntimeError('Visual alternation weight sharing failed')
                        mem('wrist_context_prepared')
                    record['native_release'] = release_unused_native_expert(model, torch)
                    if dual_visual:
                        record['wrist_native_release'] = release_unused_native_expert(wrist_model, torch)
                    adapter = OriginalOpenPiRTCAdapter(model, torch, cfg)
                    if dual_visual:
                        adapter.attach_wrist(wrist_model)
                    root_expert = adapter.expert
                    record['suffix_loads'] += 1
                    gate = json.loads((Path(cfg['verified_reference_run'])/'summary.json').read_text())
                    if not gate.get('passed') or root_expert.provenance != gate['original_source_hashes']:
                        raise RuntimeError('Frozen source gate mismatch')
                adapter.config = cfg
                if explicit_visual:
                    adapter.select_next_mode('Full')
                stage_progress('model_ready')
                first = {}
                adapter.bootstrap(prepared, task['instruction'], first)
                num_views = 2 if kind == 'full' else 1
                lengths = h50_state_prefix_lengths(first['rtc_guidance']['valid_prefix_length'],
                                                   task['instruction'], num_views=num_views,pool_factor=pool_factor,
                                                   pooled_view=cfg.get('m2_pool_view','both'))
                if dual_visual:
                    wrist_valid = adapter.prepare_wrist_context(prepared, task['instruction'])
                    if visual_fast_mode=='selective416':
                        full_lengths=list(lengths)
                        fast_lengths=[n-96 for n in full_lengths]
                        if wrist_valid!=first['rtc_guidance']['valid_prefix_length']-96:
                            raise RuntimeError('Selective416 changed prompt/state tokenization')
                        coverage=dict(kind='same_state_tokenizer_bounds_visual_offset',full=full_lengths,fast=fast_lengths,fast_mode='selective416',fast_state_enabled=True)
                        lengths=full_lengths+fast_lengths
                    elif cfg.get('m4_acceptance_group') in ('joint','m3_fast'):
                        from perceptbudget.m4_acceptance import visual_prefixes
                        lengths,coverage=visual_prefixes(lengths,wrist_valid,cfg['m4_fast_prompt_capacity'])
                    else:
                        from perceptbudget.shared_visual_context import historical_visual_prefixes
                        lengths, coverage = historical_visual_prefixes(cfg, suite, tid,
                            first['rtc_guidance']['valid_prefix_length'], wrist_valid,
                            allowed_text_range=[min(lengths)-512,max(lengths)-512])
                    cfg['m2_visual_prefix_coverage'] = coverage
                    record.setdefault('visual_prefix_coverage', []).append(dict(suite=suite,task_id=tid,**coverage))
                cfg['source_step_prefixes'] = lengths
                graph = adapter.prepare_step_bank(lengths, progress=lambda r: mem('prefix_'+str(r['valid'])))
                stage_progress('graph_bank_ready')
                record['graph_audits'].append(dict(suite=suite, task_id=tid, **graph))
                previous = np.asarray(first['actions_raw'], dtype=np.float32)[5:]
                timing = timed(torch, lambda: adapter.predict(prepared, task['instruction'], previous,
                                       delay=5, executed=5, request={}), count=3)
                stage_progress('inference_calibrated')
                cfg['rtc_initial_delay_steps'] = math.ceil(timing['median_ms']/1000/calibration['median_seconds'])
                if split_delay_comparison:
                    adapter.bootstrap(prepared,task['instruction'],{})
                    cfg['m2_reuse_enabled']=True
                    def expert_probe():
                        # Fixed-input calibration only: re-enable one reuse of
                        # the same Full snapshot outside the measured episodes.
                        adapter._recompute_context['reused']=False
                        request={'observation_step':5}
                        result=adapter.predict(prepared,task['instruction'],previous,
                            delay=5,executed=5,request=request)
                        if request['kv_recompute']['mode']!='expert_only':
                            raise RuntimeError('Expert-only delay calibration ran Full')
                        return result
                    expert_timing=timed(torch,expert_probe,count=3)
                    cfg['m2_reuse_enabled']=False
                    cfg['rtc_initial_delays_by_mode']=dict(full=cfg['rtc_initial_delay_steps'],
                        expert_only=math.ceil(expert_timing['median_ms']/1000/calibration['median_seconds']))
                    record['mode_delay_calibration']=dict(full=timing,expert_only=expert_timing,
                        initial_delays=cfg['rtc_initial_delays_by_mode'],
                        control_period_seconds=calibration['median_seconds'])
                    record.setdefault('mode_delay_calibrations', []).append(dict(suite=suite, task_id=tid,
                        **record['mode_delay_calibration']))
                if dual_visual:
                    cfg['m2_visual_fast_enabled']=True
                    def wrist_probe():
                        if explicit_visual:adapter.select_next_mode('Fast')
                        adapter._guided_requests=cfg.get('m2_full_per_fast',1)-1  # Calibration only.
                        request={'observation_step':5}
                        actions=adapter.predict(prepared,task['instruction'],previous,delay=5,executed=5,request=request)
                        if request['context_refresh']['mode']!=visual_fast_mode:
                            raise RuntimeError('Wrist delay calibration did not encode a new wrist context')
                        return actions
                    try:
                        wrist_timing=timed(torch,wrist_probe,count=3)
                    finally:
                        cfg['m2_visual_fast_enabled']=False
                        if explicit_visual:adapter.select_next_mode('Full')
                    if visual_fast_mode=='selective416':
                        # New geometry must not inherit stateless320's latency.
                        cfg['m3_fast_service_calibration_seconds']=[t/1000 for t in wrist_timing['trials_ms']]
                        cfg['m3_fast_energy_prior_scope']='old320 M2 energy is only a ranking proxy; report measured task energy'
                    cfg['rtc_initial_delays_by_mode']={'full':cfg['rtc_initial_delay_steps'],
                        visual_fast_mode:math.ceil(wrist_timing['median_ms']/1000/calibration['median_seconds'])}
                    record.setdefault('mode_delay_calibrations', []).append(dict(suite=suite,task_id=tid,
                        full=timing,**{visual_fast_mode:wrist_timing},initial_delays=cfg['rtc_initial_delays_by_mode'],
                        control_period_seconds=calibration['median_seconds']))
                if cfg.get('m4_enabled'):
                    from perceptbudget.m4_costs import load_service_costs
                    measured=load_service_costs(cfg['m4_settings'])
                    cfg['rtc_initial_delays_by_compute']={f'{mode}:{profile}':math.ceil(max(c['p95_service_seconds'] for c in measured
                        if c['mode']==service and c['profile']==profile)/calibration['median_seconds'])
                        for mode,service in [('full','Full'),('pool2_agent','Fast')] for profile in ('baseline','gpu_emc')}
                cfg['rtc_initial_control_period_seconds'] = calibration['median_seconds']
                cfg['rtc_initial_control_period_source'] = 'same-environment preflight12 ticks, then reset'
                for view in views:
                    current_view = 'full' if recompute_comparison or visual_comparison or dvac_paired else view
                    child_cfg = copy.deepcopy(cfg)
                    if prepared_start is not None:child_cfg['m4_reuse_calibrated_start']=True
                    child_cfg['m2_view'] = current_view
                    episode_q=5
                    if cfg.get('m4_acceptance_group')=='q30':
                        from perceptbudget.m4_acceptance import configure_baseline
                        episode_q=configure_baseline(child_cfg)
                    if cfg.get('m3_watchdog_checks'):
                        child_cfg.update(m3_watchdog_case=view.removeprefix('watchdog_'),
                            m3_dvac_enabled=True,m3_action_events_enabled=False)
                    if dvac_paired:
                        child_cfg['m3_dvac_enabled']=view!='deadline_high'
                        if cfg.get('m3_trigger_comparison'):
                            trigger,_,device=view.partition('__device_')
                            trigger,_,coverage=trigger.partition('__coverage_')
                            if coverage:
                                child_cfg['m3_event_coverage_mode']=coverage
                            strategy,_,repeat=trigger.partition('__repeat')
                            child_cfg['m3_trigger_strategy']=strategy
                            child_cfg['m3_trigger_repeat']=int(repeat) if repeat else 1
                            if cfg.get('m3_margin_scan'):
                                from perceptbudget.m3_margin import configure_margin
                                configure_margin(child_cfg,child_cfg['m3_trigger_repeat'])
                            if 'events_and_variance_no_turn' in (cfg.get('m3_trigger_variants') or []):
                                child_cfg['m3_ignore_turn']=strategy=='events_and_variance_no_turn'
                            if 'events_and_variance_latest' in (cfg.get('m3_trigger_variants') or []):
                                child_cfg['m3_signal_policy']='latest' if strategy=='events_and_variance_latest' else 'earliest'
                            if 'events_and_variance_valley' in (cfg.get('m3_trigger_variants') or []):
                                child_cfg['m3_events']['slowdown_detector']=(
                                    'valley' if strategy=='events_and_variance_valley' else 'drop')
                            if 'events_and_variance_hold3' in (cfg.get('m3_trigger_variants') or []):
                                child_cfg['m3_events']['slowdown_confirm_steps']=(
                                    3 if strategy=='events_and_variance_hold3' else 2)
                            if strategy in ('events_and_variance_open_probe','events_and_variance_slowdown_probe'):
                                from perceptbudget.action_events import select_open_refresh_probe, select_slowdown_refresh_probe
                                reference=next(e for e in record['episodes'] if e['suite']==suite and e['task_id']==tid
                                               and e['view']=='events_and_variance')
                                reference_run=Path(reference['run'])
                                reference_requests=json.loads((reference_run/'rtc/q5_requests.json').read_text())
                                reference_episode=json.loads((reference_run/'summary.json').read_text())['episode']
                                selector=(select_slowdown_refresh_probe if strategy=='events_and_variance_slowdown_probe' else select_open_refresh_probe)
                                child_cfg['m3_open_probe_plan']=selector(reference_requests,reference_episode['commits'])
                                if child_cfg['m3_open_probe_plan'] is None:
                                    skipped=dict(suite=suite,task_id=tid,view=view,
                                        reason_code=('no_slowdown_refresh' if strategy=='events_and_variance_slowdown_probe' else 'no_comparable_open_event'),reference_run=str(reference_run),
                                        reason='This baseline has no eligible target refresh; no intervention executed')
                                    record.setdefault('skipped_episodes',[]).append(skipped)
                                    record['diagnostic_status']='not_applicable'
                                    _write_json_atomically(directory/'open_probe_plan.json',dict(status='not_applicable',**skipped))
                                    save()
                                    print('quality_episode_skipped='+json.dumps(skipped),flush=True)
                                    continue
                                _write_json_atomically(directory/'open_probe_plan.json',child_cfg['m3_open_probe_plan'])
                            from perceptbudget.closed_loop import configure_trigger_signals
                            configure_trigger_signals(child_cfg,strategy)
                            if cfg.get('m3_device_comparison'):
                                child_cfg['m3_scheduler_mode']='deadline_high' if device=='high' else 'deadline_dvfs'
                                child_cfg['execution_mode']='m3_'+child_cfg['m3_scheduler_mode']+'_actual_async'
                                child_cfg['m3_device_profile']=device
                        if cfg.get('m3_dvac_horizon_comparison'):
                            child_cfg['m3_timing']['maximum_action_offset']=20 if view=='dvac_h20' else cfg['H']
                        if view.startswith('fixed_q'):
                            from perceptbudget.closed_loop import configure_fixed_baseline
                            episode_q=configure_fixed_baseline(child_cfg,int(view.removeprefix('fixed_q')))
                    if cfg.get('m3_formal_comparison'):
                        from perceptbudget.m3_comparison import configure_group
                        episode_q=configure_group(child_cfg,view)
                    if recompute_comparison:
                        child_cfg['m2_reuse_enabled']=view!='full'
                        child_cfg['m2_split_delay']=view in ('kv_reuse_split','kv_once')
                        child_cfg['m2_single_reuse']=view=='kv_once'
                    if dual_visual:
                        child_cfg['m2_visual_fast_enabled']=view!='full'
                        child_cfg['m2_split_delay']=view!='full' or explicit_visual
                    adapter.config = child_cfg
                    label = view if pool_factor == 1 else 'pool'+str(pool_factor)+'-'+cfg.get('m2_pool_view','both')
                    active = create_run(child_cfg, 'M4' if cfg.get('m4_enabled') or cfg.get('m4_acceptance_group') else 'M3' if record_only else 'M2',
                                        'record-full' if record_only else quality_run_purpose(label))
                    record.update(active_run=str(active),active_group=view)
                    save()
                    trace_dir = active/'rtc'
                    trace_dir.mkdir()
                    _write_json_atomically(active/'control_period_calibration.json', calibration)
                    mem('before_'+suite+str(tid)+view)
                    with (trace_dir/'trace.jsonl').open('w') as stream:
                        start_options={'prepared_start':prepared_start} if prepared_start is not None else {}
                        episode = probe.run_episode(env, task, episode_q, model, stats, child_cfg, np, torch,
                            trace_dir, stream, rtc_adapter=adapter, condition='m2_'+label,
                            refresh_policy='latest_due', pending_arm_gain=1.,**start_options)
                    _write_json_atomically(active/'summary.json', dict(status='validating', episode=episode))
                    checked = audit(active)
                    _write_json_atomically(active/'system_audit.json', checked)
                    if not checked['passed']:
                        raise RuntimeError('Quality episode independent record audit failed')
                    if cfg.get('m3_formal_comparison'):
                        from perceptbudget.m3_comparison import audit_group_clocks
                        clocks_checked=audit_group_clocks(active,view)
                        _write_json_atomically(active/'formal_clock_audit.json',clocks_checked)
                        if not clocks_checked['passed']:
                            raise RuntimeError('Formal group frequency or restoration check failed')
                    requests = json.loads((trace_dir/f'q{episode_q}_requests.json').read_text())
                    if any(r['rtc_guidance'].get('variable_length_fallback') for r in requests):
                        raise RuntimeError('Unexpected slow fallback in quality screen')
                    result = dict(status='completed', episode=episode, audit_passed=True,
                        view=view, suite=suite, task_id=tid,
                        vision_pool_factor=pool_factor, pooled_view=cfg.get('m2_pool_view','both'),visual_tokens=expected_tokens,
                        steady_inference_median_ms=float(np.median([
                            (r['inference_ended_monotonic']-r['inference_started_monotonic'])*1000
                            for r in requests[1:]])) if len(requests)>1 else None)
                    if dual_visual:
                        result.update(episode_geometry(visual_fast_mode,view!='full' or explicit_visual))
                    info = initial_info(active, episode)
                    review = None
                    if dvac_paired and view!=views[0]:
                        reference=next(r['initial'] for r in record['episodes']
                            if r['suite']==suite and r['task_id']==tid and r['view']==views[0])
                        review=compare_initial(reference,info)
                    if kind == 'single':
                        full = json.loads((directory/'full_quality.json').read_text())
                        reference = next(r['initial'] for r in full['episodes']
                                         if r['suite']==suite and r['task_id']==tid)
                        review = compare_initial(reference, info)
                    result['initial_pair_review'] = review
                    if record_only:
                        from perceptbudget.action_events import EventConfig, analyze_episode
                        event_record = analyze_episode(active, EventConfig(**child_cfg.get('m3_events', {})))
                        _write_json_atomically(active/'action_events.json', event_record)
                        result['action_event_summary'] = event_record['summary']
                    _write_json_atomically(active/'summary.json', result)
                    metadata = json.loads((active/'run.json').read_text())
                    metadata['status'] = 'completed'
                    _write_json_atomically(active/'run.json', metadata)
                    record['episodes'].append(dict(suite=suite, task_id=tid, view=view,
                        vision_pool_factor=pool_factor, pooled_view=cfg.get('m2_pool_view','both'),visual_tokens=expected_tokens,
                        run=str(active), outcome=episode['status'], steps=episode['steps'],
                        wall_seconds=episode['real_seconds'], audit_passed=True,
                        underflow_steps=len(episode['underflow_steps']), initial=info,
                        initial_pair_review=review, inference_median_ms=result['steady_inference_median_ms']))
                    if dual_visual:
                        record['episodes'][-1].update(episode_geometry(visual_fast_mode,view!='full' or explicit_visual))
                    if record_only:
                        record['episodes'][-1]['action_events'] = str(active/'action_events.json')
                    record['status'] = 'running'
                    save()
                    print('quality_episode='+json.dumps(record['episodes'][-1]), flush=True)
                    del episode, requests, result
                    checked = None
                    active = None
                    record.update(active_run=None,active_group=None)
                    mem('episode_released')
                env.close()
                env = None
                mem('environment_closed')
                if cleanup_gate and record['render_buffer_cleanup']['errors']:
                    raise RuntimeError('Render buffer cleanup reported errors')
                if kind == 'single' and tid == 0 and not pending:
                    full = json.loads((directory/'full_quality.json').read_text())
                    record['selection'] = select_candidate(full['episodes'], record['episodes'])
                    save()
                    if record['selection'] and not calibration_only:
                        pending.extend((s, 3) for s in SUITES)
            if cfg.get('m3_formal_comparison') or cfg.get('m3_margin_scan'):
                paired=audit_paired_bootstraps(record['episodes'])
                _write_json_atomically(directory/'paired_bootstrap_audit.json',paired)
                record['paired_bootstrap_audit']=paired
                record['comparison_initials_exact']=paired['passed']
            if not record.get('skipped_episodes') and (cfg.get('m3_event_dedup_comparison') or
                    tuple(cfg.get('m3_trigger_variants') or []) in (('events_only','events_and_variance'),('events_only','variance_only')) or
                    any(s in (cfg.get('m3_trigger_variants') or []) for s in ('events_and_variance_hold3','events_and_variance_open_probe','events_and_variance_valley','events_and_variance_latest','events_and_variance_no_turn','events_and_variance_slowdown_probe','events_and_variance_merge','events_only_merge'))):
                paired=audit_paired_bootstraps(record['episodes'])
                _write_json_atomically(directory/'paired_bootstrap_audit.json',paired)
                record['paired_bootstrap_audit']=paired
                if not paired['passed']:
                    raise RuntimeError('Resident comparison bootstrap differs; do not attribute outcomes to the policy change')
            if not record.get('skipped_episodes') and any(s in (cfg.get('m3_trigger_variants') or []) for s in ('events_and_variance_open_probe','events_and_variance_slowdown_probe')):
                from perceptbudget.action_events import audit_open_probe_prefix
                prefix=audit_open_probe_prefix(record['episodes'][0]['run'],record['episodes'][1]['run'],
                    json.loads((directory/'open_probe_plan.json').read_text()))
                _write_json_atomically(directory/'open_probe_prefix_audit.json',prefix)
                record['open_probe_prefix_audit']=prefix
                if not prefix['passed']:
                    raise RuntimeError('Open-probe pre-intervention prefix differs; comparison is invalid')
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
