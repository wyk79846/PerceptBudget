"""Full live-sensor service timing at one fixed LIBERO state, not a rollout."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from perceptbudget.device_clock_recovery import RecoveryLease
from perceptbudget.device_clock_screen import energy_window, frequency_check
from perceptbudget.device_control import inventory
from perceptbudget.device_clocks import MATRIX_PROFILES
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor


def full_preparation_peer(adapter,model):
    """Preparation-only eager Full context, sharing frozen suffix weights, no reload."""
    import copy
    peer=copy.copy(adapter)
    peer.model,peer.pipe,peer.pipeline=model,model._pipe,model._pipe.pipeline
    peer.expert=copy.copy(adapter.step_bank.fallback if adapter.step_bank is not None else adapter.expert)
    peer.expert.keys=peer.expert.values=peer.expert.prefix_mask=None
    peer.step_bank=peer.graph=peer.graph_expert=peer.fallback_expert=None
    return peer


def fresh_observation(env):
    """Force live camera/state sensors, without a zero-action physics step."""
    import numpy as np
    start = time.monotonic()
    before = env.env.sim.get_state().flatten().copy()
    observation = env.env._get_observations(force_update=True)
    after = env.env.sim.get_state().flatten()
    if not np.array_equal(before, after):
        raise RuntimeError('Observation acquisition advanced or changed physics')
    return observation, dict(start_monotonic=start, stop_monotonic=time.monotonic(),
        physics_sha256=hashlib.sha256(np.asarray(after, dtype=np.float64).tobytes()).hexdigest(),
        force_update=True, physics_advanced=False)


def prepare_submission(actions, raw_actions, old_queue, observation_step, consumer_step):
    """Use the existing runtime crop/commit implementation, no new alignment."""
    import numpy as np
    from scripts.probe_delay_buffer import commit
    actions = np.asarray(actions, dtype=np.float32)
    raw = np.asarray(raw_actions, dtype=np.float32)
    if actions.shape != (50, 7) or raw.shape != (50, 32):
        raise ValueError('Full H50 output shape mismatch')
    if not np.isfinite(actions).all() or not np.isfinite(raw).all() or consumer_step < observation_step:
        raise ValueError('Invalid actions or consumer timeline')
    queue = dict(old_queue)
    expired, overwritten = commit(queue, actions, observation_step, consumer_step,
                                  prediction_id=1, raw_actions=raw)
    targets = [target for target in sorted(queue) if queue[target][1] == 1]
    interval = [max(observation_step, consumer_step), observation_step+50]
    if not targets:
        interval = [observation_step+50, observation_step+50]
    payload = np.asarray([queue[t][0] for t in targets], dtype=np.float32).copy()
    return dict(observation_step=observation_step, consumer_step=consumer_step,
                nominal_prediction_end_step=observation_step+50,
                new_interval_steps=interval, expired_offsets=expired, overwritten=overwritten,
                target_steps=targets, action_payload=payload,
                limitation='Nominal H50 extent is not a validated safe buffer')


def stable_reference_observation(env, directory):
    """Use two consecutive identical fresh renders for the static reference."""
    import numpy as np
    started = time.monotonic()
    frames = []
    for attempt in range(4):
        obs, acquisition = fresh_observation(env)
        frame = np.concatenate((obs['agentview_image'],obs['robot0_eye_in_hand_image']),axis=1).copy()
        if frames and np.array_equal(frame, frames[-1]):
            acquisition.update(reference_camera_reads=attempt+1,
                               start_monotonic=started,stop_monotonic=time.monotonic())
            return obs, acquisition
        frames.append(frame)
    np.savez_compressed(Path(directory)/'unstable_reference_frames.npz',frames=np.asarray(frames))
    raise RuntimeError('Cannot establish two identical fresh static reference frames')


def checked_fresh_observation(env, expected_frame, directory, label, *, max_reads=3):
    """Static calibration only: reread real sensors, never substitute saved pixels."""
    import numpy as np
    if max_reads not in (1, 2, 3):
        raise ValueError('Static camera validation supports at most3 reads')
    started = time.monotonic()
    rejected = []
    for attempt in range(max_reads):
        obs, acquisition = fresh_observation(env)
        current = np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1).copy()
        if np.array_equal(current, expected_frame):
            acquisition.update(start_monotonic=started, stop_monotonic=time.monotonic(),
                real_camera_reads=attempt+1, rejected_frame_diagnostics=rejected,
                calibration_only=True)
            return obs, acquisition
        prefix = Path(directory)/(label+'_frame_attempt'+str(attempt))
        np.savez_compressed(prefix.with_suffix('.npz'), actual=current, reference=expected_frame)
        same_shape = current.shape == expected_frame.shape
        delta = np.abs(current.astype(np.int16)-expected_frame.astype(np.int16)) if same_shape else None
        diagnostic = dict(attempt=attempt, actual_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
            reference_sha256=hashlib.sha256(expected_frame.tobytes()).hexdigest(),
            max_abs_pixel_difference=None if delta is None else int(delta.max()),
            changed_elements=None if delta is None else int(np.count_nonzero(delta)),
            acquisition=acquisition, rejected_read=True)
        _write_json_atomically(prefix.with_suffix('.json'), diagnostic)
        rejected.append(str(prefix.with_suffix('.json')))
    raise RuntimeError('Fresh static calibration images still differ after3 real camera reads; '
                       'input rejected before inference; inspect '+str(prefix))


def measure_one(adapter, env, prompt, stats, previous_raw, old_queue, *, torch, np,state_enabled=True,
                expected_frame=None, diagnostic_directory=None, diagnostic_label='request'):
    """Start device timing is supplied by root SERVE; this covers new input to packet."""
    from scripts import eval_q as base
    if expected_frame is None:
        obs, acquisition = fresh_observation(env)
        frame_before = None
    else:
        obs, acquisition = checked_fresh_observation(env, expected_frame, diagnostic_directory, diagnostic_label)
        frame_before = np.concatenate((obs['agentview_image'],obs['robot0_eye_in_hand_image']),axis=1).copy()
    preprocess_start = time.monotonic()
    snapshot = base._make_snapshot(obs, stats, state_enabled)
    preprocess_end = time.monotonic()
    request = {'observation_step': 5, 'observation_monotonic': acquisition['start_monotonic']}
    infer_start = time.monotonic()
    actions = adapter.predict(snapshot, prompt, previous_raw, delay=5, executed=5, request=request)
    infer_end = time.monotonic()
    if not state_enabled and (request.get('rtc_guidance',{}).get('state_in_prompt') is not False
            or adapter.pipeline.vision_seq_enc!=320 or adapter.pipeline.num_views!=2
            or adapter.pipeline.vision_pool_factor!=2):
        raise RuntimeError('Selected Fast live-service state or geometry mismatch')
    if request.get('rtc_guidance', {}).get('variable_length_fallback'):
        raise RuntimeError('Live service used unprepared slow fallback')
    submit_start = time.monotonic()
    packet = prepare_submission(actions, request['actions_raw'], old_queue, 5, 5)
    ready = time.monotonic()
    frame_hash = hashlib.sha256(np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1).tobytes()).hexdigest()
    if frame_before is not None and hashlib.sha256(frame_before.tobytes()).hexdigest()!=frame_hash:
        np.savez_compressed(Path(diagnostic_directory)/(diagnostic_label+'_mutated_during_predict.npz'),
            before=frame_before, after=np.concatenate((obs['agentview_image'],obs['robot0_eye_in_hand_image']),axis=1))
        raise RuntimeError('Live observation pixels mutated during preprocessing/inference')
    return dict(acquisition=acquisition, preprocess_start_monotonic=preprocess_start,
                preprocess_stop_monotonic=preprocess_end,
                inference_start_monotonic=infer_start, inference_stop_monotonic=infer_end,
                submit_start_monotonic=submit_start, ready_monotonic=ready,
                packet=packet, request=request, frame_sha256=frame_hash,
                raw_state=snapshot['raw_state'].tolist())


def validate_profiles(profiles):
    profiles = tuple(profiles)
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in MATRIX_PROFILES for p in profiles):
        raise ValueError('Use distinct supported Full service profiles')
    return profiles


def validation_old_chunk(metadata, stats):
    """Common recorded Full chunk keeps Full/Fast calibration physics paired."""
    import numpy as np
    from perceptbudget.h50_checkpoint import decode_actions
    source = Path(metadata['source_run'])/'rtc/q5_requests.json'
    record = json.loads(source.read_text())[0]
    raw = np.asarray(record['actions_raw'], dtype=np.float32)
    if record['observation_step'] != 0 or raw.shape != (50,32) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid common Full calibration chunk')
    return raw, decode_actions(raw, stats)


def summarize_services(rows):
    return dict(services=len(rows),
        service_median_ms=statistics.median(r['service_ms'] for r in rows),
        service_max_ms=max(r['service_ms'] for r in rows),
        mean_service_energy_j=statistics.mean(r['energy']['energy_j'] for r in rows),
        acquisition_median_ms=statistics.median((r['measured']['acquisition']['stop_monotonic']-
            r['measured']['acquisition']['start_monotonic'])*1000 for r in rows),
        preprocess_median_ms=statistics.median((r['measured']['preprocess_stop_monotonic']-
            r['measured']['preprocess_start_monotonic'])*1000 for r in rows),
        predict_median_ms=statistics.median((r['measured']['inference_stop_monotonic']-
            r['measured']['inference_start_monotonic'])*1000 for r in rows),
        submit_median_ms=statistics.median((r['measured']['ready_monotonic']-
            r['measured']['submit_start_monotonic'])*1000 for r in rows))


def matched_hold_baseline(directory, label, profile, idle_profile, idle, rows, *, prepare):
    """Replay batch clock/hold timeline without acquisition/inference, plus3s tail.

    Mean batch increment only; includes replay/logging differences, no event tail claim.
    """
    origin = rows[0]['timing']['down']['command_monotonic']
    duration = rows[-1]['measured']['ready_monotonic']-origin+3.
    monitor = PowerMonitor(.02, max_gap_seconds=.1)
    monitor.start()
    baseline_origin = None
    records = []
    events = []
    try:
        prepare()
        for chunk in range(len(rows)//5):
            with RecoveryLease(idle_profile, idle, directory, 'hold_'+label+'_'+str(chunk),
                               serve_profile=profile) as lease:
                for local_index in range(5):
                    source = rows[chunk*5+local_index]
                    offset = source['timing']['down']['command_monotonic']-origin
                    if baseline_origin is not None:
                        time.sleep(max(0., baseline_origin+offset-time.monotonic()))
                    lease.send('RUN')
                    event = lease.receive('SERVE', local_index)
                    if baseline_origin is None:
                        baseline_origin = event['down']['command_monotonic']
                    stop = baseline_origin+source['measured']['ready_monotonic']-origin
                    time.sleep(max(0., stop-time.monotonic()))
                    records.append(dict(index=chunk*5+local_index, timing=event,
                        requested_stop_monotonic=stop, actual_stop_monotonic=time.monotonic(),
                        down_schedule_error_ms=(event['down']['command_monotonic']-baseline_origin-offset)*1000))
                    lease.send('DONE')
                    lease.receive('REQUEST_ACK', local_index)
            events.extend(lease.events)
        for row in records:
            row['frequency_check'] = frequency_check(events, profile,
                row['timing']['up']['command_monotonic'], row['actual_stop_monotonic'])
            if not row['frequency_check']['matches']:
                raise RuntimeError('Matched hold service frequency mismatch')
            if idle:
                row['idle_frequency_check'] = frequency_check(events, idle_profile,
                    row['timing']['down']['observed_match_monotonic'],
                    row['timing']['up']['command_monotonic'])
                if not row['idle_frequency_check']['matches']:
                    raise RuntimeError('Matched hold idle frequency mismatch')
        stop = baseline_origin+duration
        if time.monotonic() >= stop:
            raise RuntimeError('Matched hold exceeded common sampling window')
        time.sleep(max(0., stop-time.monotonic()))
        monitor.sample()
        time.sleep(.025)
        monitor.sample()
        energy = energy_window(monitor, baseline_origin, stop)
        if energy['energy_j'] is None:
            raise RuntimeError('Matched hold baseline power incomplete')
        return dict(energy=energy, schedule=records,
            max_abs_down_schedule_error_ms=max(abs(r['down_schedule_error_ms']) for r in records))
    finally:
        monitor.stop()
        _write_json_atomically(directory/('hold_'+label+'_power.json'), [asdict(r) for r in monitor.readings])


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata, *, profiles=('baseline',)):
    import gc
    import ctypes
    import numpy as np
    from scripts import eval_q as base
    from scripts.eval_pooled_views import CLEANUP_GATE
    from scripts.probe_delay_buffer import commit
    from perceptbudget.device_quality import install_render_buffer_cleanup
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import memory_record

    directory = Path(directory)
    profiles = validate_profiles(profiles)
    cfg = adapter.config
    validation = bool(cfg.get('m2_service_validation'))
    requests_per_condition = 20 if validation else 5
    histories = (('hot', 0.), ('short', .2), ('long', 1.)) if validation else (('hot', 0.),)
    gate = json.loads((CLEANUP_GATE/'reset_reuse_audit.json').read_text())
    if not gate['passed'] or len(gate['cases']) != 6 or gate['render_cleanup']['errors']:
        raise RuntimeError('Require passed hard-reset render cleanup gate')
    base._patch_egl_cleanup()
    cleanup = install_render_buffer_cleanup(shared_egl=True)
    _, env_class = base._libero_api()
    task = base._load_tasks([0], ['libero_spatial'])[0]
    stats = load_libero_norm_stats(cfg['norm_stats_path'])
    result = dict(metadata, status='running', requests=[], conditions=[], render_cleanup=cleanup,
        scope='Full simulation-side service: device readiness, forced live dual-view/state acquisition, '
              'preprocess, original RTC, existing queue commit preparation at static physics step5; '
              'not asynchronous rollout, physical camera transport, or success-rate test',
        suite='libero_spatial', task_id=0, profile=profiles[0] if len(profiles)==1 else None,
        profiles=list(profiles), reference_profile=profiles[0], requested_extra_idle_seconds=0.,
        planned_requests=requests_per_condition*len(profiles)*len(histories),
        validation_batch=cfg.get('m2_validation_batch'), histories=list(histories),
        requests_per_condition=requests_per_condition,
        conditioning_delay_steps=5, conditioning_executed_steps=5,
        actual_control_steps_during_each_service=0, first_gpu_operator_timestamp=None,
        power_boundary='INA3221 VDD_IN module input', clock_settings_changed=False,
        warmup_input_frame_sha256=metadata.get('input_frame_sha256'), input_frame_sha256=None,
        input_mode='fresh forced live camera/state at fixed physics step5')
    selected_fast=metadata.get('selected_fast',False)
    if selected_fast:
        result.update(scope='Selected Fast simulation-side service: device readiness, forced live dual images, '
            'external64+wrist256 with no state, original RTC, queue commit preparation at static physics step5; '
            'old queue prepared by Full with state; not asynchronous rollout or physical camera transport',
            input_mode='fresh forced live dual camera at fixed physics step5; raw state audited but not encoded',
            full_preparation_state=True)
    if validation:
        result.update(single_request_energy_status='Diagnostic only: subsecond INA response is unresolved',
            static_input_policy='At most3 real reads, exact pixels only; rejected reads saved, all acquisition/retry cost counted',
            calibrated_energy_field='conditions[].mean_batch_increment_per_request_j',
            energy_scope='Mean long batch minus same-duration hold; includes logging and replay differences, not event-tail energy')
    path = directory/('selected_fast_service.json' if selected_fast else 'full_service.json')
    env = None
    monitor = None
    started = time.monotonic()
    try:
        base._seed_process(7, np, torch)
        env = base._make_env(env_class, task)
        env._pb_lock_fixtures = True
        obs, _ = base._prepare_env_episode(env, task, np)
        first = {}
        prep_adapter=getattr(adapter,'full_preparation_adapter',None) if selected_fast else adapter
        if prep_adapter is None:
            raise RuntimeError('Selected Fast service requires Full preparation peer')
        first_actions = prep_adapter.bootstrap(base._make_snapshot(obs, stats, True), task['instruction'], first)
        if selected_fast:
            if first['rtc_guidance']['state_in_prompt'] is not True or prep_adapter.pipeline.vision_seq_enc!=512:
                raise RuntimeError('Old action preparation did not use Full with state')
            result['full_preparation_guidance']=first['rtc_guidance']
            prep_adapter.expert.keys=prep_adapter.expert.values=prep_adapter.expert.prefix_mask=None
        if validation:
            common_raw, first_actions = validation_old_chunk(metadata, stats)
            first = dict(first, actions_raw=common_raw.tolist())
            result['old_queue_source'] = 'common recorded Full chunk from verified source_run, not per-mode regenerated chunk'
        queue = {}
        commit(queue, first_actions, 0, 0, 0, first['actions_raw'])
        for step in range(5):
            obs, _, done, _ = env.step(queue.pop(step)[0])
            if done or env.check_success():
                raise RuntimeError('Preparation ended task before live service')
        previous_raw = np.asarray(first['actions_raw'], dtype=np.float32)[5:].copy()
        result['old_target_sha256'] = hashlib.sha256(previous_raw.tobytes()).hexdigest()
        obs, initial_acquisition = stable_reference_observation(env, directory) if validation else fresh_observation(env)
        validation_frame = np.concatenate((obs['agentview_image'],obs['robot0_eye_in_hand_image']),axis=1).copy()
        validation_options = dict(expected_frame=validation_frame, diagnostic_directory=directory) if validation else {}
        # Reuse the resident suffix and graph stream, release the warmup-only
        # exact-length bank before preparing this real state's prefix.
        root_expert = adapter.step_bank.fallback if adapter.step_bank is not None else adapter.expert
        adapter.step_bank = None
        adapter.expert = root_expert
        gc.collect()
        torch.cuda.empty_cache()
        live_bootstrap = {}
        adapter.bootstrap(base._make_snapshot(obs, stats, not selected_fast), task['instruction'], live_bootstrap)
        valid = live_bootstrap['rtc_guidance']['valid_prefix_length']
        result['warmup_valid_prefix'] = result['valid_prefix']
        result['valid_prefix'] = valid
        result['live_graph_audit'] = adapter.prepare_step_bank([valid])
        for _ in range(2):
            adapter.predict(base._make_snapshot(obs, stats, not selected_fast), task['instruction'], previous_raw,
                            delay=5, executed=5, request={})
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
        mem = memory_record(torch, 'full_service_live_prepared')
        result.update(memory=mem, live_prepare_seconds=time.monotonic()-started,
                      initial_physics_sha256=initial_acquisition['physics_sha256'])
        if mem['available_mib'] < 900:
            raise RuntimeError('Full service RAM below900MiB')
        device = inventory()
        result['device_before_service'] = dict(power_mode=device['power_mode'], thermal=device['thermal'])
        references = {}
        scenarios = [(profile, state, idle) for state, idle in histories for profile in profiles]
        for profile, state, idle in scenarios:
            rows = []
            label = profile+'_'+state if validation else profile
            idle_profile = 'gpu_emc' if idle else profile
            thermal_before = inventory()['thermal']
            monitor = PowerMonitor(.02, max_gap_seconds=.1)
            monitor.start()
            events = []
            try:
                if validation:
                    torch.manual_seed(20261023)
                    measure_one(adapter, env, task['instruction'], stats,
                                previous_raw, queue, torch=torch, np=np, state_enabled=not selected_fast,
                                **dict(validation_options, diagnostic_label='warm_'+label))
                for chunk in range(requests_per_condition//5):
                    lease_label = 'full_service_'+label+'_'+str(chunk) if validation else 'full_service_'+profile
                    with RecoveryLease(idle_profile, idle, directory, lease_label,
                                       serve_profile=profile) as lease:
                        result['clock_settings_changed'] = True
                        for local_index in range(5):
                            index = chunk*5+local_index
                            torch.manual_seed(20261023+index)
                            lease.send('RUN')
                            event = lease.receive('SERVE', local_index)
                            if event['serve_profile'] != profile:
                                raise RuntimeError('Wrong Full service frequency profile')
                            measured = measure_one(adapter, env, task['instruction'], stats,
                                                   previous_raw, queue, torch=torch, np=np,state_enabled=not selected_fast,
                                                   **dict(validation_options, diagnostic_label=label+'_'+str(index)))
                            lease.send('DONE')
                            lease.receive('REQUEST_ACK', local_index)
                            monitor.sample()
                            begin = event['up']['command_monotonic']
                            ready = measured['ready_monotonic']
                            energy = energy_window(monitor, begin, ready)
                            if energy['energy_j'] is None:
                                raise RuntimeError('Full service power coverage incomplete')
                            if measured['acquisition']['physics_sha256'] != result['initial_physics_sha256']:
                                raise RuntimeError('Physics changed during isolated service measurement')
                            packet = measured['packet']
                            payload = packet.pop('action_payload')
                            raw = np.asarray(measured['request']['actions_raw'], dtype=np.float32)
                            np.savez_compressed(directory/f'packet_{label}_{index}.npz', actions=payload, raw_actions=raw)
                            if index not in references:
                                references[index] = (measured['frame_sha256'], measured['raw_state'], raw.copy())
                            ref_frame, ref_state, ref_actions = references[index]
                            difference = float(np.max(np.abs(raw-ref_actions)))
                            row = dict(profile=profile, state_history=state, requested_idle_seconds=idle,
                                idle_profile=idle_profile, index=index, seed=20261023+index,
                                timing=event, measured=measured, energy=energy,
                                service_ms=(ready-begin)*1000,
                                frame_matches_reference=measured['frame_sha256']==ref_frame,
                                state_matches_reference=measured['raw_state']==ref_state,
                                raw_action_max_abs_difference_vs_reference=difference,
                                observation_age_at_ready_seconds=ready-measured['acquisition']['start_monotonic'])
                            if not row['frame_matches_reference'] or not row['state_matches_reference'] or difference != 0:
                                _write_json_atomically(directory/('rejected_'+label+'_'+str(index)+'.json'),row)
                                raise RuntimeError('Paired observation/state/actions changed across service profiles/histories')
                            rows.append(row)
                            result['requests'].append(row)
                            _write_json_atomically(path, result)
                            print('FULL_SERVICE_REQUEST='+label+'/'+str(index)+' ms='+str(row['service_ms']), flush=True)
                    events.extend(lease.events)
                if validation:
                    batch_start = rows[0]['timing']['down']['command_monotonic']
                    batch_stop = rows[-1]['measured']['ready_monotonic']+3.
                    time.sleep(max(0., batch_stop-time.monotonic()))
                    monitor.sample()
                    time.sleep(.025)
                    monitor.sample()
                    batch_energy = energy_window(monitor, batch_start, batch_stop)
                    if batch_energy['energy_j'] is None:
                        raise RuntimeError('Long service batch energy incomplete')
            finally:
                monitor.stop()
                _write_json_atomically(directory/f'full_service_{label}_power.json',
                                       [asdict(r) for r in monitor.readings])
            for row in rows:
                check = frequency_check(events, profile, row['timing']['up']['command_monotonic'],
                                        row['measured']['ready_monotonic'])
                check['note'] = '20ms software polling, not GPU launch timestamp'
                if not check['matches']:
                    raise RuntimeError('Full service actual frequency mismatch')
                row['frequency_check'] = check
                if idle:
                    hold = frequency_check(events, idle_profile,
                        row['timing']['down']['observed_match_monotonic'], row['timing']['up']['command_monotonic'])
                    if not hold['matches']:
                        raise RuntimeError('Live recovery idle frequency mismatch')
                    row['idle_frequency_check'] = hold
            result['conditions'].append(dict(profile=profile, state_history=state,
                requested_idle_seconds=idle, restored=True,
                thermal_before=thermal_before, thermal_after=inventory()['thermal'],
                max_raw_action_difference_vs_reference=max(r['raw_action_max_abs_difference_vs_reference'] for r in rows),
                **summarize_services(rows)))
            if validation:
                def prepare():
                    torch.manual_seed(20261023)
                    measure_one(adapter, env, task['instruction'], stats, previous_raw, queue,
                                torch=torch, np=np, state_enabled=not selected_fast,
                                **dict(validation_options, diagnostic_label='hold_warm_'+label))
                hold = matched_hold_baseline(directory, label, profile, idle_profile, idle, rows, prepare=prepare)
                result['conditions'][-1].update(batch_energy=batch_energy, matched_hold=hold,
                    mean_batch_increment_per_request_j=(batch_energy['energy_j']-hold['energy']['energy_j'])/len(rows),
                    energy_interpretation='Batch-integrated minus matched hold, mean only; includes logs and preparation differences; short request energy fields are diagnostic, not resolved single-event measurements')
            _write_json_atomically(path, result)
        result.update(status='completed', restored=True)
        if validation:
            result['camera_extra_reads'] = sum(r['measured']['acquisition'].get('real_camera_reads',1)-1 for r in result['requests'])
        if len(profiles)==1:
            result.update(summarize_services(result['requests']))
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if monitor is not None:
            monitor.stop()
        if env is not None:
            env.close()
            if validation:
                env = None
                gc.collect()
                ctypes.CDLL(None).malloc_trim(0)
                torch.cuda.empty_cache()
        _write_json_atomically(path, result)
    if validation and not selected_fast and not cfg.get('m2_skip_continue_validation', False):
        from perceptbudget.device_continue import screen as continue_screen
        adapter.config['m2_continue_validation'] = True
        try:
            continuation = continue_screen(adapter, snapshot, prompt, previous, torch,
                directory, dict(metadata, shared_model_with_full_service=True), history_windows=True)
            if continuation['status'] != 'completed' or len(continuation['windows']) != 60:
                raise RuntimeError('Incomplete selected Continue validation')
        finally:
            adapter.config.pop('m2_continue_validation', None)
    return result
