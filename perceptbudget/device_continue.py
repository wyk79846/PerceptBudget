"""Bounded LIBERO Continue windows with a resident model, never replayed tails."""
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
from pathlib import Path
import statistics
import time

from perceptbudget.device_clocks import MATRIX_PROFILES
from perceptbudget.device_clock_screen import Lease, ROOT, energy_window, frequency_check
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor

STEPS = 5
REPEATS = 5


def continue_observation(env, history, references, directory, label, *, validated=False):
    """M2 static phase images use the same exact fresh-read guard as Full/Fast."""
    from perceptbudget.device_service import fresh_observation, checked_fresh_observation, stable_reference_observation
    if not validated:
        return fresh_observation(env)
    if history in references:
        return checked_fresh_observation(env, references[history], directory, label)
    return stable_reference_observation(env, directory)


def load_history_resume(directory,config):
    """Resume only checked windows; never edit or relabel the failed source run."""
    import json
    directory=Path(directory).resolve()
    root=(Path(config['output_root'])/'M2').resolve()
    if directory.parent!=root:
        raise ValueError('Continue resume source must be an M2 run')
    prior=json.loads((directory/'run.json').read_text())['config_snapshot']
    for key in ('checkpoint_provenance_revision','H','N','state_enabled','state_prompt_capacity',
                'rtc_beta','rtc_author_order','rtc_exact_flash_attention','rtc_fused_norm','source_packed_conditioning'):
        if prior.get(key)!=config.get(key):
            raise ValueError('Continue resume configuration mismatch: '+key)
    record=json.loads((directory/'continue_control.json').read_text())
    if 'no-inference continue history' not in record.get('scope','').lower() or record.get('preflight_only'):
        raise ValueError('Require real Continue history evidence, not preflight or camera comparison')
    seen=set()
    for w in record['windows']:
        key=(w['profile'],w['requested_history_seconds'],w['repeat'])
        n={0.:0,.2:3,1.:15}.get(key[1])
        if key in seen or key[0] not in MATRIX_PROFILES or key[2] not in range(REPEATS) or n is None:
            raise ValueError('Invalid or duplicate Continue resume window')
        seen.add(key)
        if (not w.get('restored') or not w['frequency_check']['matches']
                or not w['refresh_frequency_check']['matches'] or w['committed_prefix_max_abs_error']!=0
                or w['executed_steps']!=5 or w['deadline_overruns']
                or [r['source_action_index'] for r in w['rows']]!=list(range(n,n+5))
                or any(r['inference_active'] for r in w['rows'])):
            raise ValueError('Unchecked Continue resume window')
        for field in ('transient_continue','including_fresh_images'):
            p=w[field]['power_coverage']
            if p['coverage_fraction']!=1 or p['missing_count'] or w[field]['energy_j'] is None:
                raise ValueError('Incomplete Continue resume power')
    return record


class ContinueLease(Lease):
    def __init__(self, profile, directory, label, *, duration=2):
        if profile not in MATRIX_PROFILES:
            raise ValueError('Continue profile outside whitelist')
        super().__init__(profile, directory)
        self.path = directory/(label+'_clocks.jsonl')
        if duration not in (2,4):
            raise ValueError('Continue lease duration outside whitelist')
        self.duration = duration

    def command(self):
        return super().command()+['--trace-interval', '.02']


def control_window(env, observation, actions, lease, scene, frame, trace, *, np,
                   period=.05, capture_frames=True,steps=STEPS,source_offset=0):
    """Five sequential committed indices, stop on terminal, no inference API."""
    actions = np.asarray(actions, dtype=np.float32)
    if type(steps) is not int or steps<1 or type(source_offset) is not int or source_offset<0 or source_offset+steps>50:
        raise ValueError('Continue indices outside original H50')
    if actions.ndim != 2 or actions.shape[0] < steps or actions.shape[1] != 7 or not np.isfinite(actions[:steps]).all():
        raise ValueError('Need at least five finite seven-dimensional actions')
    committed = actions[:steps].copy()
    rows, frames = [], []
    start = deadline = time.monotonic()
    for index, action in enumerate(committed):
        lease.ensure(.1)
        tick = time.monotonic()
        before = scene(env, observation)
        if capture_frames:
            frames.append(frame(observation))
        action_start = time.monotonic()
        observation, reward, done, _ = env.step(action)
        action_end = time.monotonic()
        after = scene(env, observation)
        success = bool(env.check_success())
        deadline += period
        row = dict(step=source_offset+index, source_action_index=source_offset+index, source_prediction_id='fresh_bootstrap',
                   tick_monotonic=tick, action_start_monotonic=action_start, action_end_monotonic=action_end,
                   before=before, after=after, action=action.tolist(), reward=float(reward),
                   success=success, env_done=bool(done), inference_active=False,
                   queue_after_step_actions=steps-index-1,
                   deadline_overrun_seconds=max(0., time.monotonic()-deadline))
        rows.append(row)
        trace(row)
        if done or success:
            break
        if time.monotonic() > deadline:
            deadline = time.monotonic()
        else:
            time.sleep(max(0., deadline-time.monotonic()))
    stop = time.monotonic()
    lease.ensure()
    if capture_frames:
        frames.append(frame(observation))
    return dict(start_monotonic=start, stop_monotonic=stop, duration_seconds=stop-start,
                executed_steps=len(rows), actual_hz=len(rows)/(stop-start),
                env_step_median_ms=statistics.median((r['action_end_monotonic']-r['action_start_monotonic'])*1000 for r in rows),
                deadline_overruns=sum(r['deadline_overrun_seconds']>0 for r in rows),
                rows=rows), frames


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata, *, camera_compare=False,history_windows=False):
    import gc
    import ctypes
    import numpy as np
    from scripts import eval_q as base, probe_delay_buffer as probe
    from scripts.eval_pooled_views import CLEANUP_GATE
    from perceptbudget.device_quality import install_render_buffer_cleanup
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import memory_record
    import json

    directory = Path(directory)
    config = adapter.config
    validation = bool(config.get('m2_continue_validation'))
    repeats = 20 if validation else REPEATS
    profiles = ('gpu_emc',) if validation else MATRIX_PROFILES
    gate = json.loads((CLEANUP_GATE/'reset_reuse_audit.json').read_text())
    if not gate['passed'] or len(gate['cases']) != 6 or gate['render_cleanup']['errors']:
        raise RuntimeError('Render cleanup gate missing or failed')
    base._patch_egl_cleanup()
    cleanup = install_render_buffer_cleanup(shared_egl=True)
    _, env_class = base._libero_api()
    task = base._load_tasks([0], ['libero_spatial'])[0]
    stats = load_libero_norm_stats(config['norm_stats_path'])
    result = dict(metadata, status='running', windows=[], render_cleanup=cleanup,
        scope='Spatial0 bounded first-five-action Continue with live LIBERO physics, dual-view rendering, '
              'scene checks and trace; Full immediately before each window; transient power, NOT steady hold or success rate',
        planned_windows=15, max_control_steps=75, committed_steps=STEPS,
        cleanup_gate=str(CLEANUP_GATE), memory=[], clock_settings_changed=False)
    if camera_compare:
        result.update(planned_windows=30, max_control_steps=150, comparison_period_seconds=.07,
            scope='Resident-model paired camera-on/off Continue, five committed actions at shared 70ms period; '
                  'fresh dual images after both; not production cadence, safe-buffer validation, task success or steady idle power')
    if history_windows:
        if camera_compare:
            raise ValueError('History windows cannot compare camera modes')
        result.update(planned_windows=45,max_control_steps=495,comparison_period_seconds=.07,
            scope='No-inference Continue history at fixed target clocks: execute0/3/15 original actions, then measure next5 at70ms ticks; no repeated prefix, no hardware sleep or safe-buffer claim')
        if validation:
            result.update(planned_windows=60, max_control_steps=660,
                validation_batch=config.get('m2_validation_batch'),
                shared_model_with_full_service=not config.get('m2_continue_validation_separate',False),
                requests_per_condition=20, profiles=list(profiles))
        if config.get('m2_continue_resume_run'):
            source=Path(config['m2_continue_resume_run'])
            previous_record=load_history_resume(source,config)
            result['windows']=previous_record['windows']
            for w in result['windows']:
                w.setdefault('evidence_run',str(source))
            result.update(resume_source=str(source),resumed_windows=len(result['windows']),
                prior_model_loads=previous_record.get('model_loads',1),current_process_model_loads=1,
                model_loads=previous_record.get('model_loads',1)+1)
    path = directory/'continue_control.json'
    env = None
    reference = None
    refreshed_reference = None
    history_frame_references={}
    initial_frame_references={}
    history_frame_hashes={w['requested_history_seconds']:w['refreshed_frame_sha256'] for w in result['windows']}
    done={(w['profile'],w.get('requested_history_seconds'),w['repeat']) for w in result['windows']}
    if result['windows']:
        first_window=result['windows'][0]
        from perceptbudget.h50_checkpoint import decode_actions
        reference=(first_window['initial_sim_sha256'],first_window['initial_frame_sha256'],
                   decode_actions(np.asarray(first_window['prediction']['actions_raw'],dtype=np.float32),stats)[:STEPS])
    def frame(obs):
        return np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
    try:
        base._seed_process(7, np, torch)
        env = base._make_env(env_class, task)
        env._pb_lock_fixtures = True
        for profile in profiles:
            plan = [(r, off) for r in range(repeats)
                            for off in ((False, True) if r%2==0 else (True, False))] if camera_compare else [(r, False) for r in range(repeats)]
            if history_windows:
                plan=[(r,True,h) for h in (0.,.2,1.) for r in range(repeats)]
            else:
                plan=[(r,off,0.) for r,off in plan]
            for repeat, cameras_disabled,history in plan:
                if (profile,history,repeat) in done:
                    continue
                label = profile+'_'+str(repeat)+('_off' if cameras_disabled else '_on') if camera_compare else profile+'_'+str(repeat)
                if history_windows:
                    label+='_'+str(history)
                print('CONTINUE_START='+label, flush=True)
                base._seed_process(7, np, torch)
                observation, hz = base._prepare_env_episode(env, task, np)
                initial_acquisition = None
                if validation:
                    observation, initial_acquisition = continue_observation(env, 'initial', initial_frame_references,
                        directory, label+'_initial', validated=True)
                    initial_frame_references.setdefault('initial',frame(observation).copy())
                observation_time = time.monotonic() if initial_acquisition is None else initial_acquisition['start_monotonic']
                if hz != 20:
                    raise RuntimeError('Unexpected nominal control frequency')
                prepared = base._make_snapshot(observation, stats, True)
                initial = probe.snapshot_scene(env, observation)
                initial_hash = hashlib.sha256(np.asarray(initial['sim_state'], dtype=np.float64).tobytes()).hexdigest()
                initial_frame_hash = hashlib.sha256(frame(observation).tobytes()).hexdigest()
                gc.collect()
                ctypes.CDLL(None).malloc_trim(0)
                mem = memory_record(torch, label+'_prepared')
                result['memory'].append(mem)
                if mem['available_mib'] < 900:
                    raise RuntimeError('Continue RAM headroom below900MiB')
                monitor = PowerMonitor(.02, max_gap_seconds=.1)
                monitor.start()
                try:
                    request = {}
                    actions = adapter.bootstrap(prepared, task['instruction'], request)
                    prefix = np.asarray(actions[:STEPS], dtype=np.float32)
                    if reference is None:
                        reference = (initial_hash, initial_frame_hash, prefix.copy())
                    if (initial_hash, initial_frame_hash) != reference[:2]:
                        raise RuntimeError('Initial physics/frame hashes not paired')
                    action_error = float(np.max(np.abs(prefix-reference[2])))
                    if action_error != 0:
                        raise RuntimeError('Fresh committed prefixes differ; not a controlled clock comparison')
                    with ContinueLease(profile, directory, label,duration=4 if history_windows else 2) as lease:
                        result['clock_settings_changed'] = True
                        if config.get('reuse_control_forward'):
                            from perceptbudget.env_forward import reuse_control_forward
                            context = reuse_control_forward(env)
                        else:
                            context = nullcontext()
                        from perceptbudget.continue_camera import suspend_cameras
                        with (directory/(label+'_trace.jsonl')).open('w') as stream, context, \
                                (suspend_cameras(env) if cameras_disabled else nullcontext()):
                            consumed=0; history_record=None
                            if history_windows and history:
                                import math
                                consumed=math.ceil(history/.07)
                                history_record,_=control_window(env,observation,actions,lease,
                                    probe.snapshot_scene,frame,lambda row:base._trace(stream,'history_control_step',**row),
                                    np=np,period=.07,capture_frames=False,steps=consumed)
                                if history_record['executed_steps']!=consumed or history_record['rows'][-1]['success'] or history_record['rows'][-1]['env_done']:
                                    raise RuntimeError('History ended before valid measurement window')
                                observation=env.env._get_observations(force_update=True)
                            window, frames = control_window(env, observation, actions[consumed:], lease,
                                probe.snapshot_scene, frame,
                                lambda row: base._trace(stream, 'control_step', **row), np=np,
                                period=.07 if camera_compare or history_windows else .05,
                                capture_frames=not (camera_compare or history_windows),source_offset=consumed)
                        if camera_compare or history_windows:
                            if window['deadline_overruns']:
                                raise RuntimeError('Shared 70ms cadence missed; camera energy comparison invalid')
                            fresh, acquisition = continue_observation(env, history, history_frame_references,
                                directory, label, validated=validation)
                            refreshed = frame(fresh)
                            if refreshed_reference is None:
                                refreshed_reference = refreshed.copy()
                            if camera_compare and not np.array_equal(refreshed, refreshed_reference):
                                raise RuntimeError('Restored fresh camera images differ across paired windows')
                            if history_windows:
                                current_hash=hashlib.sha256(refreshed.tobytes()).hexdigest()
                                expected_hash=history_frame_hashes.setdefault(history,current_hash)
                                if current_hash!=expected_hash:
                                    reference_frame=history_frame_references.get(history)
                                    arrays={'actual':refreshed}
                                    diagnostic=dict(label=label,expected_sha256=expected_hash,actual_sha256=current_hash,
                                        actual_final_sim_sha256=hashlib.sha256(np.asarray(window['rows'][-1]['after']['sim_state'],dtype=np.float64).tobytes()).hexdigest(),
                                        reference_pixels_available=reference_frame is not None)
                                    if reference_frame is not None:
                                        arrays['reference']=reference_frame
                                        delta=np.abs(refreshed.astype(np.int16)-reference_frame.astype(np.int16))
                                        diagnostic.update(max_abs_pixel_difference=int(delta.max()),changed_elements=int(np.count_nonzero(delta)))
                                    np.savez_compressed(directory/(label+'_image_mismatch.npz'),**arrays)
                                    _write_json_atomically(directory/(label+'_image_mismatch.json'),diagnostic)
                                    raise RuntimeError('Restored history camera images differ across paired clocks')
                                history_frame_references.setdefault(history,refreshed.copy())
                            window['refreshed_frame_sha256'] = hashlib.sha256(refreshed.tobytes()).hexdigest()
                            window['fresh_acquisition'] = acquisition
                            refreshed_stop = time.monotonic()
                        monitor.sample()
                        power = energy_window(monitor, window['start_monotonic'], window['stop_monotonic'])
                        if power['energy_j'] is None:
                            raise RuntimeError('Continue power coverage incomplete')
                        if camera_compare or history_windows:
                            window['including_fresh_images'] = energy_window(monitor, window['start_monotonic'], refreshed_stop)
                            if window['including_fresh_images']['energy_j'] is None:
                                raise RuntimeError('Fresh camera energy coverage incomplete')
                            window['refresh_frequency_check'] = frequency_check(lease.events, profile,
                                window['start_monotonic'], refreshed_stop)
                            if not window['refresh_frequency_check']['matches']:
                                raise RuntimeError('Frequency mismatch during restored camera acquisition')
                finally:
                    monitor.stop()
                    _write_json_atomically(directory/(label+'_power.json'), [asdict(r) for r in monitor.readings])
                check = frequency_check(lease.events, profile, window['start_monotonic'], window['stop_monotonic'])
                check['note'] = '20ms software samples; not hard real-time guarantee'
                if not check['matches']:
                    raise RuntimeError('Continue frequencies did not match target')
                np.savez_compressed(directory/(label+'_frames.npz'), frames=np.asarray(frames))
                window.update(profile=profile, repeat=repeat, restored=True,
                    cameras_disabled=cameras_disabled,
                    requested_history_seconds=history,history_execution=history_record,
                    consumed_before_window=consumed,
                    frequency_check=check, transient_continue=power,
                    initial_sim_sha256=initial_hash, initial_frame_sha256=initial_frame_hash,
                    committed_prefix_max_abs_error=action_error,
                    prediction=request, nominal_committed_seconds=.25,
                    source_observation_step=0, committed_indices=list(range(consumed,consumed+STEPS)),
                    source_observation_monotonic=observation_time,
                    prediction_ready_monotonic=request['inference_ended_monotonic'],
                    committed_interval_steps=[consumed, consumed+STEPS],
                    nominal_expected_execution_interval_monotonic=[window['start_monotonic'],
                                                                  window['start_monotonic']+.25],
                    full_end_to_control_start_seconds=window['start_monotonic']-request['inference_ended_monotonic'],
                    final_sim_sha256=hashlib.sha256(np.asarray(window['rows'][-1]['after']['sim_state'], dtype=np.float64).tobytes()).hexdigest())
                if validation:
                    window['initial_camera_acquisition'] = initial_acquisition
                if camera_compare or history_windows:
                    window['nominal_expected_execution_interval_monotonic'][1] = window['start_monotonic']+STEPS*.07
                result['windows'].append(window)
                _write_json_atomically(path, result)
                print('CONTINUE_DONE='+label+' hz='+str(window['actual_hz']), flush=True)
        result['summary'] = []
        for profile in profiles:
          for disabled,history in ([(True,h) for h in (0.,.2,1.)] if history_windows else [(off,0.) for off in ((False, True) if camera_compare else (False,))]):
            rows = [w for w in result['windows'] if w['profile']==profile and w['cameras_disabled']==disabled and w['requested_history_seconds']==history]
            summary = dict(profile=profile, cameras_disabled=disabled, windows=len(rows),
                executed_steps=sum(w['executed_steps'] for w in rows),
                mean_actual_hz=statistics.mean(w['actual_hz'] for w in rows),
                env_step_median_ms=statistics.median(w['env_step_median_ms'] for w in rows),
                mean_transient_power_w=statistics.mean(w['transient_continue']['mean_power_w'] for w in rows),
                mean_window_energy_j=statistics.mean(w['transient_continue']['energy_j'] for w in rows),
                deadline_overruns=sum(w['deadline_overruns'] for w in rows),
                paired_final_state_equal=len({w['final_sim_sha256'] for w in rows})==1,
                restored=all(w['restored'] for w in rows))
            summary['requested_history_seconds']=history
            if camera_compare or history_windows:
                summary['mean_energy_including_fresh_images_j'] = statistics.mean(w['including_fresh_images']['energy_j'] for w in rows)
            result['summary'].append(summary)
        paired=all(len({w['final_sim_sha256'] for w in result['windows'] if w['requested_history_seconds']==h})==1 for h in (0.,.2,1.)) if history_windows else len({w['final_sim_sha256'] for w in result['windows']})==1
        result.update(status='completed', all_final_states_equal=paired)
        if not result['all_final_states_equal']:
            raise RuntimeError('Paired actions produced different final physics; do not qualify clock comparison')
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if env is not None:
            env.close()
        _write_json_atomically(path, result)
    return result
