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


class ContinueLease(Lease):
    def __init__(self, profile, directory, label):
        if profile not in MATRIX_PROFILES:
            raise ValueError('Continue profile outside whitelist')
        super().__init__(profile, directory)
        self.path = directory/(label+'_clocks.jsonl')
        self.duration = 2

    def command(self):
        return super().command()+['--trace-interval', '.02']


def control_window(env, observation, actions, lease, scene, frame, trace, *, np,
                   period=.05, capture_frames=True):
    """Five sequential committed indices, stop on terminal, no inference API."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < STEPS or actions.shape[1] != 7 or not np.isfinite(actions[:STEPS]).all():
        raise ValueError('Need at least five finite seven-dimensional actions')
    committed = actions[:STEPS].copy()
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
        row = dict(step=index, source_action_index=index, source_prediction_id='fresh_bootstrap',
                   tick_monotonic=tick, action_start_monotonic=action_start, action_end_monotonic=action_end,
                   before=before, after=after, action=action.tolist(), reward=float(reward),
                   success=success, env_done=bool(done), inference_active=False,
                   queue_after_step_actions=STEPS-index-1,
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


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata, *, camera_compare=False):
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
    path = directory/'continue_control.json'
    env = None
    reference = None
    refreshed_reference = None
    def frame(obs):
        return np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1)
    try:
        base._seed_process(7, np, torch)
        env = base._make_env(env_class, task)
        env._pb_lock_fixtures = True
        for profile in MATRIX_PROFILES:
            plan = [(r, off) for r in range(REPEATS)
                    for off in ((False, True) if r%2==0 else (True, False))] if camera_compare else [(r, False) for r in range(REPEATS)]
            for repeat, cameras_disabled in plan:
                label = profile+'_'+str(repeat)+('_off' if cameras_disabled else '_on') if camera_compare else profile+'_'+str(repeat)
                print('CONTINUE_START='+label, flush=True)
                base._seed_process(7, np, torch)
                observation, hz = base._prepare_env_episode(env, task, np)
                observation_time = time.monotonic()
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
                    with ContinueLease(profile, directory, label) as lease:
                        result['clock_settings_changed'] = True
                        if config.get('reuse_control_forward'):
                            from perceptbudget.env_forward import reuse_control_forward
                            context = reuse_control_forward(env)
                        else:
                            context = nullcontext()
                        from perceptbudget.continue_camera import suspend_cameras
                        with (directory/(label+'_trace.jsonl')).open('w') as stream, context, \
                                (suspend_cameras(env) if cameras_disabled else nullcontext()):
                            window, frames = control_window(env, observation, actions, lease,
                                probe.snapshot_scene, frame,
                                lambda row: base._trace(stream, 'control_step', **row), np=np,
                                period=.07 if camera_compare else .05, capture_frames=not camera_compare)
                        if camera_compare:
                            if window['deadline_overruns']:
                                raise RuntimeError('Shared 70ms cadence missed; camera energy comparison invalid')
                            from perceptbudget.device_service import fresh_observation
                            fresh, acquisition = fresh_observation(env)
                            refreshed = frame(fresh)
                            if refreshed_reference is None:
                                refreshed_reference = refreshed.copy()
                            if not np.array_equal(refreshed, refreshed_reference):
                                raise RuntimeError('Restored fresh camera images differ across paired windows')
                            window['refreshed_frame_sha256'] = hashlib.sha256(refreshed.tobytes()).hexdigest()
                            window['fresh_acquisition'] = acquisition
                            refreshed_stop = time.monotonic()
                        monitor.sample()
                        power = energy_window(monitor, window['start_monotonic'], window['stop_monotonic'])
                        if power['energy_j'] is None:
                            raise RuntimeError('Continue power coverage incomplete')
                        if camera_compare:
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
                    frequency_check=check, transient_continue=power,
                    initial_sim_sha256=initial_hash, initial_frame_sha256=initial_frame_hash,
                    committed_prefix_max_abs_error=action_error,
                    prediction=request, nominal_committed_seconds=.25,
                    source_observation_step=0, committed_indices=list(range(STEPS)),
                    source_observation_monotonic=observation_time,
                    prediction_ready_monotonic=request['inference_ended_monotonic'],
                    committed_interval_steps=[0, STEPS],
                    nominal_expected_execution_interval_monotonic=[window['start_monotonic'],
                                                                  window['start_monotonic']+.25],
                    full_end_to_control_start_seconds=window['start_monotonic']-request['inference_ended_monotonic'],
                    final_sim_sha256=hashlib.sha256(np.asarray(window['rows'][-1]['after']['sim_state'], dtype=np.float64).tobytes()).hexdigest())
                if camera_compare:
                    window['nominal_expected_execution_interval_monotonic'][1] = window['start_monotonic']+STEPS*.07
                result['windows'].append(window)
                _write_json_atomically(path, result)
                print('CONTINUE_DONE='+label+' hz='+str(window['actual_hz']), flush=True)
        result['summary'] = []
        for profile in MATRIX_PROFILES:
          for disabled in ((False, True) if camera_compare else (False,)):
            rows = [w for w in result['windows'] if w['profile']==profile and w['cameras_disabled']==disabled]
            summary = dict(profile=profile, cameras_disabled=disabled, windows=len(rows),
                executed_steps=sum(w['executed_steps'] for w in rows),
                mean_actual_hz=statistics.mean(w['actual_hz'] for w in rows),
                env_step_median_ms=statistics.median(w['env_step_median_ms'] for w in rows),
                mean_transient_power_w=statistics.mean(w['transient_continue']['mean_power_w'] for w in rows),
                mean_window_energy_j=statistics.mean(w['transient_continue']['energy_j'] for w in rows),
                deadline_overruns=sum(w['deadline_overruns'] for w in rows),
                paired_final_state_equal=len({w['final_sim_sha256'] for w in rows})==1,
                restored=all(w['restored'] for w in rows))
            if camera_compare:
                summary['mean_energy_including_fresh_images_j'] = statistics.mean(w['including_fresh_images']['energy_j'] for w in rows)
            result['summary'].append(summary)
        result.update(status='completed', all_final_states_equal=len({w['final_sim_sha256'] for w in result['windows']})==1)
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
