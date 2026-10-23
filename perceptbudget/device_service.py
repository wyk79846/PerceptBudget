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


def measure_one(adapter, env, prompt, stats, previous_raw, old_queue, *, torch, np):
    """Start device timing is supplied by root SERVE; this covers new input to packet."""
    from scripts import eval_q as base
    obs, acquisition = fresh_observation(env)
    preprocess_start = time.monotonic()
    snapshot = base._make_snapshot(obs, stats, True)
    preprocess_end = time.monotonic()
    request = {'observation_step': 5, 'observation_monotonic': acquisition['start_monotonic']}
    infer_start = time.monotonic()
    actions = adapter.predict(snapshot, prompt, previous_raw, delay=5, executed=5, request=request)
    infer_end = time.monotonic()
    if request.get('rtc_guidance', {}).get('variable_length_fallback'):
        raise RuntimeError('Live service used unprepared slow fallback')
    submit_start = time.monotonic()
    packet = prepare_submission(actions, request['actions_raw'], old_queue, 5, 5)
    ready = time.monotonic()
    frame_hash = hashlib.sha256(np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1).tobytes()).hexdigest()
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
        planned_requests=5*len(profiles), conditioning_delay_steps=5, conditioning_executed_steps=5,
        actual_control_steps_during_each_service=0, first_gpu_operator_timestamp=None,
        power_boundary='INA3221 VDD_IN module input', clock_settings_changed=False,
        warmup_input_frame_sha256=metadata.get('input_frame_sha256'), input_frame_sha256=None,
        input_mode='fresh forced live camera/state at fixed physics step5')
    path = directory/'full_service.json'
    env = None
    monitor = None
    started = time.monotonic()
    try:
        base._seed_process(7, np, torch)
        env = base._make_env(env_class, task)
        env._pb_lock_fixtures = True
        obs, _ = base._prepare_env_episode(env, task, np)
        first = {}
        first_actions = adapter.bootstrap(base._make_snapshot(obs, stats, True), task['instruction'], first)
        queue = {}
        commit(queue, first_actions, 0, 0, 0, first['actions_raw'])
        for step in range(5):
            obs, _, done, _ = env.step(queue.pop(step)[0])
            if done or env.check_success():
                raise RuntimeError('Preparation ended task before live service')
        previous_raw = np.asarray(first['actions_raw'], dtype=np.float32)[5:].copy()
        result['old_target_sha256'] = hashlib.sha256(previous_raw.tobytes()).hexdigest()
        obs, initial_acquisition = fresh_observation(env)
        # Reuse the resident suffix and graph stream, release the warmup-only
        # exact-length bank before preparing this real state's prefix.
        root_expert = adapter.step_bank.fallback if adapter.step_bank is not None else adapter.expert
        adapter.step_bank = None
        adapter.expert = root_expert
        gc.collect()
        torch.cuda.empty_cache()
        live_bootstrap = {}
        adapter.bootstrap(base._make_snapshot(obs, stats, True), task['instruction'], live_bootstrap)
        valid = live_bootstrap['rtc_guidance']['valid_prefix_length']
        result['warmup_valid_prefix'] = result['valid_prefix']
        result['valid_prefix'] = valid
        result['live_graph_audit'] = adapter.prepare_step_bank([valid])
        for _ in range(2):
            adapter.predict(base._make_snapshot(obs, stats, True), task['instruction'], previous_raw,
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
        for profile in profiles:
            rows = []
            thermal_before = inventory()['thermal']
            monitor = PowerMonitor(.02, max_gap_seconds=.1)
            monitor.start()
            try:
                with RecoveryLease(profile, 0., directory, 'full_service_'+profile,
                                   serve_profile=profile) as lease:
                    result['clock_settings_changed'] = True
                    for index in range(5):
                        torch.manual_seed(20261023+index)
                        lease.send('RUN')
                        event = lease.receive('SERVE', index)
                        if event['serve_profile'] != profile:
                            raise RuntimeError('Wrong Full service frequency profile')
                        measured = measure_one(adapter, env, task['instruction'], stats,
                                               previous_raw, queue, torch=torch, np=np)
                        lease.send('DONE')
                        lease.receive('REQUEST_ACK', index)
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
                        np.savez_compressed(directory/f'packet_{profile}_{index}.npz', actions=payload, raw_actions=raw)
                        if index not in references:
                            references[index] = (measured['frame_sha256'], measured['raw_state'], raw.copy())
                        ref_frame, ref_state, ref_actions = references[index]
                        difference = float(np.max(np.abs(raw-ref_actions)))
                        row = dict(profile=profile, index=index, seed=20261023+index,
                            timing=event, measured=measured, energy=energy,
                            service_ms=(ready-begin)*1000,
                            frame_matches_reference=measured['frame_sha256']==ref_frame,
                            state_matches_reference=measured['raw_state']==ref_state,
                            raw_action_max_abs_difference_vs_reference=difference,
                            observation_age_at_ready_seconds=ready-measured['acquisition']['start_monotonic'])
                        rows.append(row)
                        result['requests'].append(row)
                        _write_json_atomically(path, result)
                        if not row['frame_matches_reference'] or not row['state_matches_reference'] or difference != 0:
                            raise RuntimeError('Paired observation/state/actions changed across service profiles')
                        print('FULL_SERVICE_REQUEST='+profile+'/'+str(index)+' ms='+str(row['service_ms']), flush=True)
            finally:
                monitor.stop()
                _write_json_atomically(directory/f'full_service_{profile}_power.json',
                                       [asdict(r) for r in monitor.readings])
            for row in rows:
                check = frequency_check(lease.events, profile, row['timing']['up']['command_monotonic'],
                                        row['measured']['ready_monotonic'])
                check['note'] = '20ms software polling, not GPU launch timestamp'
                if not check['matches']:
                    raise RuntimeError('Full service actual frequency mismatch')
                row['frequency_check'] = check
            result['conditions'].append(dict(profile=profile, restored=True,
                thermal_before=thermal_before, thermal_after=inventory()['thermal'],
                max_raw_action_difference_vs_reference=max(r['raw_action_max_abs_difference_vs_reference'] for r in rows),
                **summarize_services(rows)))
            _write_json_atomically(path, result)
        result.update(status='completed', restored=True)
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
        _write_json_atomically(path, result)
    return result
