"""Bounded real-control down/up test with one existing RTC queue handoff."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from perceptbudget.continue_camera import suspend_cameras
from perceptbudget.device_clock_screen import energy_window, frequency_check
from perceptbudget.device_idle_payback import PairedIdleLease, AFTER_IDLE_SECONDS, summarize
from perceptbudget.device_service import fresh_observation
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor
from perceptbudget.rtc_memory import memory_record

DELAYS = (.2, 1.)
OBSERVATION_STEPS = {.2: 4, 1.: 20}
PERIOD = .05
NEW_STEPS = 5


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


class NoWriteLease:
    """Preflight only: exercise timing/physics without any sudo or clock writes."""
    events = ()
    def __init__(self, profile, delay, directory, label):
        self.delay = delay
    def __enter__(self):
        self.deadline = time.monotonic()+40
        return self
    def ensure(self, remaining=0):
        if time.monotonic()+remaining >= self.deadline:
            raise RuntimeError('Preflight window exceeded40s')
    def send(self, token):
        if token == 'RUN':
            self.origin = time.monotonic()
        elif token != 'DONE':
            raise ValueError('Unexpected token')
    def receive(self, status, index):
        if status == 'REQUEST_ACK':
            return dict(status=status, request_index=index)
        if status != 'SERVE':
            raise ValueError('Unexpected status')
        time.sleep(max(0., self.origin+self.delay-time.monotonic()))
        now = time.monotonic()
        return dict(request_index=index, preflight_only=True,
            down=dict(command_monotonic=self.origin, observed_match_monotonic=self.origin),
            up=dict(command_monotonic=now, observed_match_monotonic=now))
    def __exit__(self, *exc):
        pass


def execute_handoff(env, initial_observation, old_actions, old_request, *, observation_step,
                    adapter, prompt, stats, torch, np, executor, lease, index, delay_estimate):
    """Main thread owns MuJoCo; worker waits for clocks and predicts concurrently."""
    from scripts import eval_q as base
    from scripts.probe_delay_buffer import commit, buffer_count
    queue = {}
    commit(queue, old_actions, 0, 0, 0, old_request['actions_raw'])
    rows, committed, measured = [], None, None
    future = None
    observation = initial_observation
    start = deadline = time.monotonic()
    new_steps = 0
    try:
        for step in range(55):
            lease.ensure(.1)
            tick = time.monotonic()
            if step == observation_step:
                observation, acquisition = fresh_observation(env)
                snapshot = base._make_snapshot(observation, stats, True)
                raw_previous = np.asarray(old_request['actions_raw'], dtype=np.float32)[step:].copy()
                if not 0 <= delay_estimate <= len(raw_previous):
                    raise RuntimeError('RTC delay exceeds remaining old chunk')
                request = dict(observation_step=step, observation_monotonic=acquisition['start_monotonic'])
                def infer(snapshot=snapshot, raw_previous=raw_previous, request=request, step=step):
                    event = lease.receive('SERVE', index)
                    actions = adapter.predict(snapshot, prompt, raw_previous, delay=delay_estimate,
                                              executed=step, request=request)
                    if request['rtc_guidance'].get('variable_length_fallback'):
                        raise RuntimeError('Live handoff used unprepared slow fallback')
                    return actions, request, event
                future = executor.submit(infer)
                measured = dict(acquisition=acquisition, snapshot_frame_sha256=digest(np.concatenate(
                    (observation['agentview_image'], observation['robot0_eye_in_hand_image']), axis=1)),
                    snapshot_state=snapshot['raw_state'].tolist(), observation_step=step,
                    snapshot_ready_monotonic=time.monotonic())
            if future is not None and future.done() and committed is None:
                actions, request, event = future.result()
                available = buffer_count(queue, step)
                expired, overwritten = commit(queue, actions, observation_step, step, 1, request['actions_raw'])
                committed = dict(step=step, monotonic=time.monotonic(), expired_offsets=expired,
                    overwritten_old_actions=overwritten, old_buffer_actions_at_commit=available,
                    actual_consumed_during_service=step-observation_step,
                    nominal_old_buffer_remaining_seconds=available*PERIOD,
                    raw_action_sha256=digest(np.asarray(request['actions_raw'], dtype=np.float32)))
                measured.update(request=request, timing=event, committed=committed)
            item = queue.pop(step, None)
            if item is None:
                raise RuntimeError('Old actions exhausted before new RTC chunk was ready; no zero-action substitution')
            action = np.asarray(item[0], dtype=np.float32)
            if not np.isfinite(action).all():
                raise RuntimeError('Nonfinite control action')
            action_start = time.monotonic()
            with suspend_cameras(env):
                observation, reward, done, _ = env.step(action)
            action_end = time.monotonic()
            terminal = bool(done or env.check_success())
            if item[1] == 1:
                new_steps += 1
            deadline += PERIOD
            overrun = max(0., time.monotonic()-deadline)
            rows.append(dict(step=step, source_prediction_id=item[1], source_offset=item[3],
                source_origin_step=item[2], action=action.tolist(), tick_monotonic=tick,
                action_start_monotonic=action_start, action_end_monotonic=action_end,
                deadline_overrun_seconds=overrun, terminal=terminal,
                inference_pending=future is not None and not future.done()))
            if terminal or new_steps == NEW_STEPS:
                break
            if time.monotonic() > deadline:
                deadline = time.monotonic()
            else:
                time.sleep(max(0., deadline-time.monotonic()))
        if committed is None or new_steps != NEW_STEPS:
            raise RuntimeError('Task terminated or control budget ended before five new actions')
        measured.update(control_rows=rows, new_actions_executed=new_steps,
            actual_control_hz=(len(rows)-1)/(rows[-1]['action_start_monotonic']-rows[0]['action_start_monotonic']),
            control_overruns=sum(r['deadline_overrun_seconds'] > 0 for r in rows),
            old_actions_executed=committed['step'], underflows=0,
            control_start_monotonic=start, control_stop_monotonic=time.monotonic())
        return measured
    finally:
        if future is not None:
            # No simulated steps are inserted during shutdown; propagate worker errors.
            future.result()


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata):
    import numpy as np
    from scripts import eval_q as base
    from scripts.eval_pooled_views import CLEANUP_GATE
    from perceptbudget.device_quality import install_render_buffer_cleanup
    from perceptbudget.libero_input import load_libero_norm_stats
    directory = Path(directory)
    path = directory/'continue_payback.json'
    preflight = bool(adapter.config.get('m2_continue_payback_preflight'))
    repeats = 1 if preflight else 5
    lease_class = NoWriteLease if preflight else PairedIdleLease
    result = dict(metadata, status='running', rows=[], model_loads=1,
        scope='Spatial0 real control during scheduled idle/recovery and Full RTC; one existing queue handoff then five new actions; not full-task success or B_safe validation',
        preflight_only=preflight, clock_settings_changed=False, planned_rows=4*repeats,
        fixed_delay_steps=None, fixed_executed_steps=None,
        conditioning_delay_mode='initial measured latency plus50ms acquisition allowance, rounded up to nominal control ticks; one refresh only',
        nominal_period_seconds=PERIOD, observation_steps=OBSERVATION_STEPS,
        requested_idle_seconds=list(DELAYS), power_boundary='INA3221 VDD_IN',
        limitations=['Nominal control20Hz is measured, not guaranteed; fresh images may cause overruns.',
            'Fixed observation indices pair inputs; actual observation age and dispatch time are recorded.',
            'Different commit steps can produce different post-handoff actions; do not claim paired final physics.',
            'Common energy tail is a diagnostic boundary, not execution buffer or proven sensor settling.'])
    env = None
    monitor = None
    def trim_and_record(stage):
        gc.collect()
        torch.cuda.empty_cache()
        ctypes.CDLL(None).malloc_trim(0)
        record = memory_record(torch, stage)
        with (directory/'control_memory.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('CONTINUE_PAYBACK_MEMORY='+json.dumps(record), flush=True)
        return record
    try:
        gate = json.loads((CLEANUP_GATE/'reset_reuse_audit.json').read_text())
        if not gate['passed'] or len(gate['cases']) != 6 or gate['render_cleanup']['errors']:
            raise RuntimeError('Require validated render cleanup gate')
        base._patch_egl_cleanup()
        result['render_cleanup'] = install_render_buffer_cleanup(shared_egl=True)
        task = base._load_tasks([0], ['libero_spatial'])[0]
        _, cls = base._libero_api()
        stats = load_libero_norm_stats(adapter.config['norm_stats_path'])
        base._seed_process(7, np, torch)
        env = base._make_env(cls, task)
        env._pb_lock_fixtures = True
        trim_and_record('environment_created')
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Find only the three live shapes outside all timed windows; one bank rebuild.
            observation, _ = base._prepare_env_episode(env, task, np)
            initial_snapshot = base._make_snapshot(observation, stats, True)
            first = {}
            torch.manual_seed(7)
            first_actions = executor.submit(adapter.bootstrap, initial_snapshot, prompt, first).result()
            trim_and_record('live_bootstrap')
            snapshots = {0: initial_snapshot}
            for step in range(20):
                with suspend_cameras(env):
                    observation, _, done, _ = env.step(first_actions[step])
                if done or env.check_success():
                    raise RuntimeError('Warmup task terminated before live shape discovery')
                if step+1 in (4, 20):
                    observation, _ = fresh_observation(env)
                    snapshots[step+1] = base._make_snapshot(observation, stats, True)
            root = adapter.step_bank.fallback if adapter.step_bank is not None else adapter.expert
            adapter.step_bank = None
            adapter.expert = root
            gc.collect()
            torch.cuda.empty_cache()
            lengths = set()
            for live in snapshots.values():
                req = {}
                executor.submit(adapter.bootstrap, live, prompt, req).result()
                lengths.add(req['rtc_guidance']['valid_prefix_length'])
                trim_and_record('discover_'+str(req['rtc_guidance']['valid_prefix_length']))
            result['live_graph_audit'] = adapter.prepare_step_bank(sorted(lengths),
                progress=lambda row: trim_and_record('graph_'+str(row['valid'])))
            result['live_valid_prefixes'] = sorted(lengths)
            delay_estimates = {}
            for delay in DELAYS:
                step = OBSERVATION_STEPS[delay]
                req = {}
                executor.submit(adapter.predict, snapshots[step], prompt,
                    np.asarray(first['actions_raw'], dtype=np.float32)[step:],
                    delay=10, executed=step, request=req).result()
                delay_estimates[delay] = math.ceil((adapter.latency_estimate_seconds+.05)/PERIOD)
                trim_and_record('timing_'+str(delay))
            result['initial_delay_estimates'] = delay_estimates
            result['memory'] = trim_and_record('continue_payback_prepared')
            if result['memory']['available_mib'] < 900:
                raise RuntimeError('Continue payback RAM below900MiB')
            references = {}
            boot_reference = None
            for delay in DELAYS:
                for profile in (('gpu_emc', 'baseline') if delay == .2 else ('baseline', 'gpu_emc')):
                    label = str(delay).replace('.', 'p')+'_'+profile
                    monitor = PowerMonitor(.02, max_gap_seconds=.1)
                    monitor.start()
                    time.sleep(.05)
                    try:
                        with lease_class(profile, delay, directory, label) as lease:
                            result['clock_settings_changed'] = not preflight
                            for index in range(repeats):
                                base._seed_process(7, np, torch)
                                observation, _ = base._prepare_env_episode(env, task, np)
                                if trim_and_record('continue_payback_reset')['available_mib'] < 900:
                                    raise RuntimeError('Continue payback reset RAM below900MiB')
                                old_request = {}
                                old_actions = executor.submit(adapter.bootstrap,
                                    base._make_snapshot(observation, stats, True), prompt, old_request).result()
                                boot_hash = digest(np.asarray(old_request['actions_raw'], dtype=np.float32))
                                if boot_reference is None:
                                    boot_reference = boot_hash
                                if boot_hash != boot_reference:
                                    raise RuntimeError('Reset/bootstrap actions differ across paired windows')
                                torch.manual_seed(20261024+index)
                                lease.send('RUN')
                                measured = execute_handoff(env, observation, old_actions, old_request,
                                    observation_step=OBSERVATION_STEPS[delay], adapter=adapter, prompt=prompt,
                                    stats=stats, torch=torch, np=np, executor=executor, lease=lease, index=index,
                                    delay_estimate=delay_estimates[delay])
                                event = measured['timing']
                                origin = event['down']['command_monotonic']
                                request = measured['request']
                                end = measured['committed']['monotonic']
                                identity = (measured['snapshot_frame_sha256'], measured['snapshot_state'],
                                            measured['committed']['raw_action_sha256'])
                                key = (delay, index)
                                if key not in references:
                                    references[key] = identity
                                if identity != references[key]:
                                    raise RuntimeError('Paired live observation/state/RTC output changed across clock strategies')
                                stop = origin+delay+AFTER_IDLE_SECONDS
                                if measured['control_stop_monotonic'] >= stop-1.:
                                    raise RuntimeError('Control exceeded shorter common measurement window')
                                time.sleep(max(0., stop-time.monotonic()))
                                monitor.sample()
                                time.sleep(.025)
                                monitor.sample()
                                cycle = energy_window(monitor, origin, stop)
                                early = energy_window(monitor, origin, stop-1.)
                                ready = energy_window(monitor, origin, end)
                                if any(w['energy_j'] is None for w in (cycle, early, ready)):
                                    raise RuntimeError('Continue payback power coverage incomplete')
                                row = dict(idle_seconds=delay, profile=profile, index=index,
                                    measured=measured, matched_cycle=cycle, early_cycle=early,
                                    through_handoff=ready, actions_ready_from_origin_ms=(end-origin)*1000,
                                    actual_ready_from_origin_ms=(request['inference_ended_monotonic']-origin)*1000,
                                    action_max_abs_difference=0., inputs_and_raw_output_exactly_paired=True,
                                    preflight_only=preflight)
                                if not preflight:
                                    row['frequency_check'] = frequency_check(lease.events, 'baseline',
                                        request['inference_started_monotonic'], request['inference_ended_monotonic'])
                                    hold = frequency_check(lease.events, profile,
                                        event['down']['observed_match_monotonic'], event['up']['command_monotonic'])
                                    row['hold_frequency_check'] = hold
                                    if not row['frequency_check']['matches'] or not hold['matches']:
                                        raise RuntimeError('Continue payback inference/hold frequency mismatch')
                                result['rows'].append(row)
                                _write_json_atomically(path, result)
                                lease.send('DONE')
                                lease.receive('REQUEST_ACK', index)
                                print('CONTINUE_PAYBACK '+label+' index='+str(index)
                                      +' commit='+str(measured['committed']['step']), flush=True)
                            if not preflight:
                                result.setdefault('restored_conditions', []).append(label)
                    finally:
                        monitor.stop()
                        _write_json_atomically(directory/(label+'_power.json'), [asdict(r) for r in monitor.readings])
            result.update(status='completed', restored=not preflight,
                          summary=[] if preflight else summarize(result['rows'], delays=DELAYS, repeats=repeats))
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
