#!/usr/bin/env python3
"""Shared LIBERO control loop, timing and traces for RTC/M1 evaluation.

Not a standalone CLI. Retired SEAM and hard-prefix experiments are in Git history.
"""

import hashlib
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from perceptbudget.libero_input import raw_libero_state
from perceptbudget.power import PowerMonitor


def buffer_count(queue, step):
    end = step
    while end in queue:
        end += 1
    return end - step


def reference_prefix_count(queue, step, q):
    end = step
    while end in queue and queue[end][3] < q:
        end += 1
    return end - step


def refresh_due(step, previous_request_step, q, policy):
    if policy in ('latest_due', 'post_commit'):
        return step > 0 and step-previous_request_step >= q
    if policy == 'boundary':
        return step > 0 and step % q == 0
    raise ValueError('unknown refresh policy')




def commit(queue, actions, origin, step, prediction_id, raw_actions=None):
    expired = overwritten = 0
    for offset, action in enumerate(actions):
        target = origin + offset
        if target < step:
            expired += 1
            continue
        overwritten += target in queue
        item = (action, prediction_id, origin, offset)
        queue[target] = item if raw_actions is None else item + (raw_actions[offset],)
    return expired, overwritten






def commit_aligned(queue, actions, origin, step, prediction_id,
                   origin_xyz, current_xyz, history, policy):
    """Experimental phase alignment, with the original horizon end unchanged.

    A fresh chunk is never recycled, and delayed arrivals never gain H new
    slots. Retain H-elapsed actions, selecting a prefix offset by measured
    motion rather than assuming elapsed steps equal completed motion.
    """
    import numpy as np
    elapsed = max(0, step-origin)
    remaining = max(0, len(actions)-elapsed)
    limit = min(elapsed, len(actions))
    gain = 0.01  # m per normalized command; fallback before enough motion.
    useful = [r for r in history[-32:]
              if np.linalg.norm(np.subtract(r['after']['raw_state'][:3],
                                             r['before']['raw_state'][:3])) > 0.001]
    if useful:
        commands = np.clip(np.asarray([r['action'][:3] for r in useful]), -1, 1)
        displacements = np.asarray([np.subtract(r['after']['raw_state'][:3],
                                                r['before']['raw_state'][:3]) for r in useful])
        gain = float(np.clip(np.sum(commands*displacements) /
                             max(float(np.sum(commands*commands)), 1e-9), 0.001, 0.05))
    offset = 0
    distance = None
    if policy == 'progress' and limit:
        xyz = np.asarray(origin_xyz) + np.vstack([
            np.zeros(3), np.cumsum(np.clip(np.asarray(actions)[:limit, :3], -1, 1)*gain, axis=0)])
        distances = np.linalg.norm(xyz-np.asarray(current_xyz), axis=1)
        offset = int(np.argmin(distances))
        distance = float(distances[offset])
        # No plausible spatial match: keep the fresh near-term feedback command.
        if distance > 0.02:
            offset = 0
        old = [r['action'][6] for r in history if origin <= r['step'] < step]
        for i in range(1, min(offset+1, len(actions))):
            if actions[i-1][6]*actions[i][6] < 0:
                target_sign = np.sign(actions[i][6])
                if not any(np.sign(value) == target_sign for value in old[i:]):
                    offset = min(offset, i)
                    break
    elif policy not in ('progress', 'head'):
        raise ValueError('unknown alignment policy')
    overwritten = 0
    for i in range(remaining):
        target = step+i
        overwritten += target in queue
        queue[target] = (actions[offset+i], prediction_id, origin, offset+i,
                         {'alignment': policy, 'scheduled_step': target,
                          'elapsed_steps': elapsed, 'selected_offset': offset})
    return offset, overwritten, {
        'policy': policy, 'elapsed_control_steps': elapsed,
        'selected_prediction_offset': offset,
        'retained_prediction_offset_interval': [offset, offset+remaining],
        'discarded_prefix_actions': offset,
        'discarded_tail_actions': len(actions)-offset-remaining,
        'remaining_actions': remaining, 'scheduled_interval_steps': [step, step+remaining],
        'original_horizon_end_step': origin+len(actions),
        'tracking_gain_m_per_command': gain, 'best_waypoint_distance_m': distance,
        'fallback_distance_threshold_m': 0.02,
    }


def commit_blended(queue, actions, origin, step, prediction_id, beta=1.2, arm_only=False):
    """Same absolute-step alignment as baseline; change only overlap update."""
    from perceptbudget.action_blend import blend_actions
    begin = max(step, origin)
    count = min(buffer_count(queue, begin), max(0, origin + len(actions) - begin))
    old = [[float(v) for v in queue[begin + i][0]] for i in range(count)]
    new = [[float(v) for v in actions[begin - origin + i]] for i in range(count)]
    sources = [{'prediction_id': queue[begin + i][1], 'origin_step': queue[begin + i][2],
                'offset': queue[begin + i][3], 'already_blended': len(queue[begin + i]) > 4}
               for i in range(count)]
    merged, weights = blend_actions(old, new, beta=beta)
    if arm_only:
        for row, latest in zip(merged, new):
            row[6:] = latest[6:]
    expired, overwritten = commit(queue, actions, origin, step, prediction_id)
    for i in range(count):
        target = begin + i
        item = queue[target]
        metadata = {'old_source': sources[i], 'new_prediction_id': prediction_id,
                    'weight_old': weights[i], 'overlap_index': i, 'overlap_length': count}
        queue[target] = (merged[i], item[1], item[2], item[3], metadata)
    return expired, overwritten, {
        'overlap_actions': count, 'global_interval_steps': [begin, begin + count],
        'new_offset_interval': [begin - origin, begin - origin + count],
        'beta': beta, 'weights_old': weights, 'old_actions': old,
        'new_actions': new, 'merged_actions': merged, 'old_sources': sources,
        'blended_dimensions': 6 if arm_only else len(actions[0]) if len(actions) else 0,
        'gripper_policy': 'latest_prediction' if arm_only else 'same_blend_as_arm',
        'single_overlap_policy': 'new action directly; paper only defines N>=2',
    }


def commit_rtc_fused(queue, raw_chunks, actions, origin, step, prediction_id,
                     decay=0.1, max_chunks=3):
    """Fuse the six arm dimensions with FlashRT's host-side RTC equations."""
    import math
    if not math.isfinite(decay) or decay < 0:
        raise ValueError('RTC decay must be finite and non-negative')
    if type(max_chunks) is not int or max_chunks < 1:
        raise ValueError('RTC max_chunks must be a positive integer')
    if len(actions) == 0 or len(actions[0]) == 0:
        raise ValueError('RTC fusion requires non-empty action vectors')
    if any(len(action) != 7 for action in actions):
        raise ValueError('LIBERO RTC fusion requires 7D actions')

    latest = {'prediction_id': prediction_id, 'origin_step': origin,
              'actions': [[float(v) for v in action] for action in actions]}
    chunks = [chunk for chunk in raw_chunks
              if chunk['origin_step'] + len(chunk['actions']) > step]
    chunks.append(latest)
    expired, overwritten = commit(queue, actions, origin, step, prediction_id)
    begin = max(step, origin)
    overlap_counts = {}
    fused_steps = 0
    for target in range(begin, origin + len(actions)):
        candidates = [chunk for chunk in chunks
                      if chunk['origin_step'] <= target <
                      chunk['origin_step'] + len(chunk['actions'])]
        candidates.sort(key=lambda chunk: chunk['prediction_id'], reverse=True)
        candidates = candidates[:max_chunks]
        latest_index = target - origin
        total_weight = 0.0
        fused_arm = [0.0] * 6
        weights = []
        for chunk in candidates:
            source_index = target - chunk['origin_step']
            weight = math.exp(-decay * abs(source_index - latest_index))
            weights.append(weight)
            total_weight += weight
            for dim in range(6):
                fused_arm[dim] += weight * chunk['actions'][source_index][dim]
        if not candidates or total_weight <= 0:
            raise RuntimeError('RTC fusion found no valid action source')
        if len(candidates) > 1:
            fused_steps += 1
        overlap_counts[len(candidates)] = overlap_counts.get(len(candidates), 0) + 1
        action = list(latest['actions'][latest_index])
        for dim in range(6):
            action[dim] = fused_arm[dim] / total_weight
        # The gripper uses the newest model prediction without temporal fusion.
        item = queue[target]
        metadata = {'prediction_id': prediction_id, 'source_prediction_ids':
                    [chunk['prediction_id'] for chunk in candidates],
                    'source_count': len(candidates), 'source_weights': weights,
                    'arm_dimensions': 6,
                    'gripper_policy': 'latest_prediction'}
        queue[target] = (action, item[1], item[2], item[3], metadata)
    raw_chunks[:] = chunks[-max_chunks:]
    return expired, overwritten, {
        'fused_steps': fused_steps, 'source_count_histogram': overlap_counts,
        'decay': decay, 'max_chunks': max_chunks,
        'arm_dimensions': 6,
        'gripper_policy': 'latest_prediction',
    }


def snapshot_scene(env, observation):
    import numpy as np
    sim = env.env.sim
    numeric = {}
    for key, value in observation.items():
        if 'image' in key:
            continue
        array = np.asarray(value)
        if array.dtype.kind in 'biuf':
            numeric[key] = array.tolist()
    contacts = []
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        contacts.append({'geom1': sim.model.geom_id2name(contact.geom1),
                         'geom2': sim.model.geom_id2name(contact.geom2),
                         'distance': float(contact.dist)})
    return {'raw_state': raw_libero_state(observation).tolist(),
            'observations': numeric, 'sim_time': float(sim.data.time),
            'sim_state': sim.get_state().flatten().tolist(), 'contacts': contacts}


def device_state():
    import glob
    patterns = ['/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq',
                '/sys/class/devfreq/*/cur_freq', '/sys/class/thermal/thermal_zone*/temp']
    result = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                result[path] = Path(path).read_text().strip()
            except OSError as exc:
                result[path] = str(exc)
    return result


def _drain_executor(executor, monitor, start, stop, *, terminal_low=None):
    """Keep sampling through worker completion; exclude subsequent artifact work."""
    try:
        executor.shutdown(wait=True)
        drain_stop = terminal_low['stop_monotonic'] if terminal_low else time.monotonic()
    finally:
        monitor.stop()
    return {
        'drain_start_monotonic': stop,
        'drain_stop_monotonic': drain_stop,
        'drain_seconds': drain_stop - stop,
        'drain_power': base._power_result(monitor, stop, drain_stop),
        'including_drain_start_monotonic': start,
        'including_drain_stop_monotonic': drain_stop,
        'including_drain_seconds': drain_stop - start,
        'including_drain_power': base._power_result(monitor, start, drain_stop),
        **({'terminal_low': dict(terminal_low,
            power=base._power_result(monitor, terminal_low['start_monotonic'], drain_stop),
            task_plus_inference_power=base._power_result(monitor, start, terminal_low['start_monotonic'])),
            'energy_scope': 'Task plus inference drain and common LOW tail; original-frequency restoration excluded'}
           if terminal_low else {}),
    }


def run_episode(env, task, q, model, stats, config, np, torch, run_dir, stream, *,
                trial_index=1, condition='crop_overwrite', action_blend_enabled=False,
                rtc_fusion_enabled=False, rtc_adapter=None, artifact_tag='',
                alignment_policy=None, refresh_policy='boundary', pending_arm_gain=1.0,
                synchronous=False, step_hook=None, prepared_start=None):
    if synchronous and (action_blend_enabled or rtc_fusion_enabled or alignment_policy is not None or (pending_arm_gain != 1.0)):
        raise ValueError('Synchronous recording must execute an unchanged fresh prefix')
    if alignment_policy is not None and (action_blend_enabled or rtc_fusion_enabled):
        raise ValueError('phase alignment is a separate diagnostic condition')
    if rtc_adapter is not None and (action_blend_enabled or rtc_fusion_enabled or alignment_policy is not None or synchronous):
        raise ValueError('RTC VJP rollout is an isolated asynchronous condition')
    if not 0 < pending_arm_gain <= 1:
        raise ValueError('pending arm gain must be in (0, 1]')
    def setup_memory(stage):
        if config.get('m3_ordinary_fast_context'):
            from perceptbudget.rtc_memory import memory_record
            from perceptbudget.experiment import _write_json_atomically
            row = dict(memory_record(torch, stage), monotonic=time.monotonic())
            _write_json_atomically(run_dir/'setup_memory.json', row)
            print('episode_setup='+json.dumps(row), flush=True)
    setup_memory('before_environment_reset')
    if prepared_start is None:
        observation, hz = base._prepare_env_episode(env, task, np)
    else:
        if config.get('m4_acceptance_group') not in ('joint','m3') or not config.get('m4_reuse_calibrated_start'):
            raise ValueError('Prepared start is restricted to explicit M3/M4 acceptance')
        from perceptbudget.m4_acceptance import calibrated_start
        observation,hz=calibrated_start(env,task,prepared_start,np)
    setup_memory('environment_reset_finished')
    assert hz == 20
    base._seed_process(7, np, torch)
    period = 1 / hz
    queue, requests, commits, rows, frames, raw_chunks = {}, [], [], [], [], []
    if config.get('disk_backed_frames'):
        from perceptbudget.trace_frames import DiskFrames
        frames = DiskFrames(run_dir / f'q{q}{artifact_tag}_frames_spool.npy', task['max_steps']+1, np)
    capture = None
    if config.get('m4_lean_continue'):
        from perceptbudget.continue_camera import ContinueCapture
        capture = ContinueCapture(request_only=bool(config.get('m4_request_only_images')))
    visual_monitor = None
    visual_images = []
    if config.get('visual_monitor'):
        from perceptbudget.visual_monitor import WristSlipMonitor
        settings=config['visual_monitor']
        visual_monitor=WristSlipMonitor(interval=settings['interval'],
            confirmation=settings['confirmation'],loss_fraction=settings['loss_fraction'])
    future = None
    previous_request_step = 0
    previous_commit_step = 0
    delay_queue = None
    if config.get('rtc_paper_schedule'):
        from perceptbudget.rtc_schedule import DelayQueue
        delay_queue = DelayQueue(config['rtc_initial_delay_steps'], config['rtc_delay_buffer_size'])
    mode_delay_queues=None
    if config.get('m2_split_delay'):
        if not (config.get('m2_expert_recompute') or config.get('m2_visual_alternation')) or delay_queue is None or rtc_adapter is None:
            raise ValueError('Split delay requires explicit expert-recompute RTC experiment')
        mode_delay_queues={mode:DelayQueue(config['rtc_initial_delays_by_mode'][mode],config['rtc_delay_buffer_size'])
                           for mode in config.get('m2_delay_modes', ('full','expert_only'))}
    if (config.get('m4_enabled') or config.get('m3_ordinary_fast_context')) and mode_delay_queues is not None:
        mode_delay_queues={key:DelayQueue(value,config['rtc_delay_buffer_size']) for key,value in config['rtc_initial_delays_by_compute'].items()}
    frequency_delay_queues=None
    if config.get('m3_low_full') and config.get('m3_low_full_policy')!='original_m3_slot_v3' and not config.get('m3_ordinary_fast_context'):
        if delay_queue is None or mode_delay_queues is not None:
            raise ValueError('Low Full requires paper RTC without visual mode delay queues')
        frequency_delay_queues=dict(baseline=delay_queue,
            gpu_emc=DelayQueue(config['m3_low_full_initial_delay'],config['rtc_delay_buffer_size']))
    current_request = None
    missed_triggers, underflows = [], []
    from perceptbudget.critical_wait import PhaseWait
    phase_wait = PhaseWait()
    prefix_rejection = None
    if config.get('m3_critical_wait') and config.get('controlled_delay_steps') is not None:
        raise ValueError('Critical wait requires actual async completion, not controlled step delay')
    if config.get('prefix_lock') or config.get('adaptive_prefix'):
        raise ValueError('Hard-prefix experiments were retired; use the RTC entrypoint')
    late_policy = 'discard'
    pauses = []
    monitor = PowerMonitor(poll_interval=0.1)
    scheduler = None
    fixed_clocks = None
    request_watchdog = None
    if config.get('m3_watchdog_path'):
        from perceptbudget.watchdog import RequestWatchdog
        request_watchdog = RequestWatchdog(config['m3_watchdog_path'], config['m3_request_timeout_seconds'])
    if config.get('m3_scheduler_mode') in ('deadline_high','deadline_dvfs'):
        from perceptbudget.closed_loop import DeadlineRuntime
        if config.get('m4_enabled'):
            from perceptbudget.m4_runtime import M4Runtime
            scheduler=M4Runtime(config,rtc_adapter)
        elif config.get('m3_three_frequency'):
            from perceptbudget.three_frequency import ThreeFullRuntime
            scheduler=ThreeFullRuntime(config)
        elif config.get('m3_low_full'):
            from perceptbudget.m3_low_full import LowFullRuntime
            scheduler=LowFullRuntime(config)
        else:
            scheduler = DeadlineRuntime(config)
        scheduler.start_clocks(run_dir)
    if config.get('m3_hold_high_clocks'):
        from perceptbudget.closed_loop import RuntimeClocks
        fixed_clocks=RuntimeClocks(run_dir)
        fixed_clocks.__enter__()
    executor = ThreadPoolExecutor(max_workers=1)
    monitor.start()
    start = time.monotonic()
    base._trace(stream, 'episode_start', task_id=task['task_id'], q=q,
                trial_index=trial_index, condition=condition,
                task_name=task['task_name'], start_monotonic=start,
                execution_mode=config.get('execution_mode', 'actual_delay_diagnostic'))

    def submit(obs, step):
        if refresh_policy == 'post_commit' and step and step != previous_commit_step + q:
            raise RuntimeError('Post-commit refresh must follow exactly q executed control steps')
        observation_started = time.monotonic()
        acquisition = None
        if capture is not None:
            obs, acquisition = capture.for_request(env, obs, step)
            frames.append(np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=1))
        snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
        request = {
            'prediction_id': len(requests),
            'observation_step': step,
            'refresh_anchor': 'commit' if refresh_policy == 'post_commit' else 'request',
            'refresh_anchor_step': previous_commit_step if refresh_policy == 'post_commit' else previous_request_step,
            'target_handoff_step': None,
            'observation_monotonic': observation_started,
            'input_ready_monotonic': time.monotonic(),
            'model_state': snapshot.get('state'),
            'observation_raw_state': snapshot['raw_state'].tolist(),
            'queue_before_submit_actions': buffer_count(queue, step),
        }
        if capture is not None:
            request['camera_acquisition'] = acquisition
            request['input_frame_step'] = step
        if config.get('m3_fast_enabled') or config.get('m3_ordinary_fast_context'):
            decision=dict(scheduler.plan or {})
            formula=decision.get('m3_fast_formula')
            if step and formula is None and (config.get('m3_fast_enabled') or config.get('m3_ordinary_fast')):raise RuntimeError('Full/Fast request has no evaluated deadline/energy plan')
            request['m3_fast_decision']=decision
            request['m3_fast_choice']=(dict(mode=formula['mode'],reason=formula['reason'])
                if step and formula else dict(mode='Full',reason='bootstrap_full' if not step else 'full_control'))
            rtc_adapter.select_next_mode(request['m3_fast_choice']['mode'])
        if scheduler is not None:
            scheduler.requested(request)
        selected_delay_queue=delay_queue
        if frequency_delay_queues is not None:
            selected_delay_queue=frequency_delay_queues[request['m4_compute_profile']]
            request['full_delay_profile']=request['m4_compute_profile']
        if mode_delay_queues is not None:
            mode=rtc_adapter.context_mode(task['instruction'],step,guided=step>0)
            request['planned_context_mode' if config.get('m2_visual_alternation') else 'planned_kv_mode']=mode
            delay_key=mode+':'+request['m4_compute_profile'] if config.get('m4_enabled') or config.get('m3_ordinary_fast_context') else mode
            request['planned_delay_key']=delay_key
            selected_delay_queue=mode_delay_queues[delay_key]
        rtc_previous_raw = None
        if rtc_adapter is not None and step > 0:
            executed = step - previous_request_step
            recent_rows = rows[-16:]
            periods = [b['action_end_monotonic'] - a['action_end_monotonic'] - b.get('critical_wait_seconds', 0.)
                       for a, b in zip(recent_rows, recent_rows[1:])]
            period_estimate = float(np.median(periods)) if periods else period
            delay = config.get('controlled_delay_steps', rtc_adapter.estimate_delay_steps(period_estimate))
            if selected_delay_queue is not None:
                delay = selected_delay_queue.estimate
            if config.get('rtc_fixed_delay_estimate') is not None:
                delay=config['rtc_fixed_delay_estimate']
            if step not in queue:
                raise RuntimeError(f'RTC old-action target missing at control step {step}')
            uncapped_delay = delay
            if config.get('rtc_critical_wait_prefix_policy') == 'committed_stop_v1':
                from perceptbudget.critical_wait import rtc_prefix_plan, wait_target
                review = dict(wait=wait_target(request) is not None, reason='point_event')
                if review['wait'] and config.get('m3_critical_wait_phases'):
                    owner = requests[queue[step][1]]
                    review = phase_wait.review(request, owner['actions'],
                        owner['observation_step'], scheduler.events_config)
                request['rtc_wait_prefix'] = rtc_prefix_plan(request, step, delay, review)
                delay = request['rtc_wait_prefix']['selected_delay_steps']
            feasible = (0 <= delay <= config['H'] - executed if delay_queue is not None
                        else delay <= executed <= config['H'] - delay)
            if not feasible:
                raise RuntimeError(f'RTC infeasible before inference: d={delay}, '
                                   f's={executed}, H={config["H"]}')
            old_origin = queue[step][2]
            if executed != step - old_origin:
                raise RuntimeError('RTC executed-prefix length does not match queued chunk origin: '
                                   f's={executed}, queue_origin={old_origin}, step={step}')
            target_len = config['H'] - executed
            try:
                rtc_previous_raw = np.asarray(
                    [queue[step + index][4] for index in range(target_len)],
                    dtype=np.float32)
            except (KeyError, IndexError) as exc:
                raise RuntimeError('RTC old-action target is incomplete or lacks raw actions') from exc
            if rtc_previous_raw.shape != (target_len, 32):
                raise RuntimeError(f'invalid RTC old-action target shape {rtc_previous_raw.shape}')
            request['rtc_guidance'] = {
                'delay_estimate_steps': delay,
                'executed_steps': executed,
                'control_period_estimate_seconds': period_estimate,
                'predicted_inference_seconds': rtc_adapter.latency_estimate_seconds,
                'old_chunk_origin_step': old_origin,
                'delay_estimate_mode': ('fixed' if config.get('rtc_fixed_delay_estimate') is not None else 'frequency_history_max' if frequency_delay_queues is not None else 'mode_history_max' if mode_delay_queues is not None else 'history_max' if delay_queue is not None else 'latency'),
            }
            if 'rtc_wait_prefix' in request:
                request['rtc_guidance']['uncapped_delay_estimate_steps'] = uncapped_delay
            if delay_queue is not None:
                request['rtc_guidance']['delay_history_steps'] = list(selected_delay_queue.values)
                if mode_delay_queues is not None:
                    request['rtc_guidance']['delay_history_mode']=mode
                    request['rtc_guidance']['delay_history_key']=request.get('planned_delay_key',mode)
                request['rtc_guidance']['initialization_d_exceeds_s'] = delay > executed
        requests.append(request)
        if request_watchdog is not None:
            request_watchdog.arm(request, run_dir)

        def predict():
            if rtc_adapter is not None and step == 0:
                if hasattr(rtc_adapter, 'bootstrap'):
                    actions = rtc_adapter.bootstrap(snapshot, task['instruction'], request)
                else:
                    actions = base._infer_actions(model, snapshot, task['instruction'],
                                                  config['H'], request)
                    request['actions_raw'] = model._pipe._noise_out.float().cpu().numpy().copy().tolist()
            elif rtc_adapter is not None:
                actions = rtc_adapter.predict(
                    snapshot, task['instruction'], rtc_previous_raw,
                    delay=request['rtc_guidance']['delay_estimate_steps'],
                    executed=request['rtc_guidance']['executed_steps'], request=request)
            else:
                actions = base._infer_actions(model, snapshot, task['instruction'],
                                              config['H'], request)
            if config.get('m3_watchdog_case'):
                from perceptbudget.watchdog import inject_request_fault
                inject_request_fault(request, config, run_dir)
            request['actions'] = actions.tolist()
            request['ready_monotonic'] = time.monotonic()
            return actions
        return executor.submit(predict), request

    def harvest(step, completed_row=None):
        nonlocal future, current_request, previous_commit_step
        if future is not None and request_watchdog is not None:
            # A completed request is judged by its ready timestamp, not the
            # later control tick which first observes completion.
            request_watchdog.check(current_request.get('ready_monotonic') if future.done() else None)
        controlled = config.get('controlled_delay_steps')
        if future is not None and controlled is not None and current_request['observation_step'] > 0:
            target_step = current_request['observation_step'] + controlled
            if step < target_step:
                return
            wait_started = time.monotonic()
            future.result()  # Controlled simulator experiment, NOT real-time execution.
            current_request['controlled_wait_seconds'] = time.monotonic() - wait_started
        if future is None or not future.done():
            return
        ready_actions = future.result()
        if request_watchdog is not None:
            request_watchdog.complete(current_request['ready_monotonic'])
        if rtc_adapter is not None and current_request.get('rtc_guidance'):
            guidance = current_request['rtc_guidance']
            guidance['actual_delay_steps'] = max(0, step - current_request['observation_step'])
            if delay_queue is not None and not current_request.get('delay_queue_observed'):
                observed_queue=delay_queue
                if frequency_delay_queues is not None:
                    observed_queue=frequency_delay_queues[current_request['m4_compute_profile']]
                if mode_delay_queues is not None:
                    mode=current_request['context_refresh' if config.get('m2_visual_alternation') else 'kv_recompute']['mode']
                    if mode!=current_request['planned_context_mode' if config.get('m2_visual_alternation') else 'planned_kv_mode']:
                        raise RuntimeError('Delay observation mode differs from planned inference')
                    key=current_request.get('planned_delay_key',mode)
                    observed_queue=mode_delay_queues[key]
                current_request['delay_queue_observed'] = observed_queue.observe(
                    guidance['actual_delay_steps'],
                    observation_step=current_request['observation_step'])
                if current_request['observation_step'] == 0:
                    guidance['delay_history_excluded_reason'] = 'bootstrap A_init is not an inference-loop delay sample'
            guidance['observed_inference_seconds'] = (
                current_request.get('inference_ended_monotonic', time.monotonic())
                - current_request.get('inference_started_monotonic',
                                      current_request['input_ready_monotonic']))
        current_request.setdefault('ready_observed_step', step)
        actions = ready_actions
        before = buffer_count(queue, step)
        origin = current_request['observation_step']
        reference_remaining = reference_prefix_count(queue, step, q)
        blend = None
        rtc_fusion = None
        alignment = None
        merge_start = time.monotonic()
        if alignment_policy is not None:
            expired, overwritten, alignment = commit_aligned(
                queue, actions, origin, step, current_request['prediction_id'],
                current_request['observation_raw_state'][:3],
                raw_libero_state(observation)[:3],
                rows if completed_row is None else rows + [completed_row], alignment_policy)
        elif action_blend_enabled:
            expired, overwritten, blend = commit_blended(
                queue, actions, origin, step, current_request['prediction_id'],
                beta=config['action_blend']['beta'],
                arm_only=config['action_blend'].get('gripper_policy') == 'latest_prediction')
        elif rtc_fusion_enabled:
            expired, overwritten, rtc_fusion = commit_rtc_fused(
                queue, raw_chunks, actions, origin, step,
                current_request['prediction_id'],
                decay=config['rtc_action_fusion']['decay'],
                max_chunks=config['rtc_action_fusion']['max_chunks'])
        else:
            expired, overwritten = commit(queue, actions, origin, step,
                                          current_request['prediction_id'], current_request.get('actions_raw'))
        result = {'prediction_id': current_request['prediction_id'],
                  'observation_step': origin, 'commit_step': step,
                  'commit_monotonic': time.monotonic(),
                  'old_queue_remaining_actions': before,
                  'old_queue_remaining_nominal_seconds': before * period,
                  'old_reference_prefix_remaining_actions': reference_remaining,
                  'B_safe_seconds': None,
                  'control_steps_during_inference': current_request['ready_observed_step'] - origin,
                  'control_steps_until_commit': step - origin,
                  'prefix_exact': current_request.get('prefix_exact'),
                  'expired_new_actions': expired, 'overwritten_old_actions': overwritten,
                  'new_prediction_remaining_actions': alignment['remaining_actions'] if alignment else len(actions) - expired,
                  'new_prediction_remaining_nominal_seconds': (alignment['remaining_actions'] if alignment else len(actions) - expired) * period,
                  'new_prediction_remaining_interval_steps': [max(origin, step), max(step, origin + len(actions))],
                  'new_queue_actions': buffer_count(queue, step)}
        if alignment is not None:
            result['action_alignment'] = alignment
        if blend is not None:
            result['blend'] = blend
            result['merge_seconds'] = time.monotonic() - merge_start
        if rtc_fusion is not None:
            result['rtc_action_fusion'] = rtc_fusion
            result['merge_seconds'] = time.monotonic() - merge_start
        commits.append(result)
        if scheduler is not None:
            scheduler.committed(current_request, result)
        previous_commit_step = step
        base._trace(stream, 'prediction_committed', task_id=task['task_id'], q=q,
                    trial_index=trial_index, condition=condition,
                    request=current_request, **result)
        future, current_request = None, None
    stop = None
    status = 'invalid'
    env_context = None
    env_audit = None
    try:
        if config.get('reuse_control_forward'):
            from perceptbudget.env_forward import reuse_control_forward
            # reset() above replaces MjSim; install only on the live instance.
            env_context = reuse_control_forward(env)
            env_audit = env_context.__enter__()
        setup_memory('before_bootstrap')
        future, current_request = submit(observation, 0)
        future.result(timeout=config.get('m3_request_timeout_seconds'))  # Bootstrap only; no env steps yet.
        harvest(0)
        setup_memory('bootstrap_finished')
        deadline = time.monotonic()
        for step in range(task['max_steps']):
            if fixed_clocks is not None:
                fixed_clocks.poll()
            if step_hook is not None:
                observation, hook_event = step_hook(env, observation, step)
                if hook_event is not None:
                    base._trace(stream, 'external_disturbance', task_id=task['task_id'],
                                q=q, step=step, details=hook_event)
            tick = time.monotonic()
            before = snapshot_scene(env, observation)
            if capture is None:
                frames.append(np.concatenate((observation['agentview_image'],
                                             observation['robot0_eye_in_hand_image']), axis=1))
            harvest(step)
            anchor_step = previous_commit_step if refresh_policy == 'post_commit' else previous_request_step
            due = refresh_due(step, anchor_step, q, refresh_policy)
            if scheduler is not None:
                rtc_delay=delay_queue.estimate if delay_queue is not None else None
                if frequency_delay_queues is not None:
                    rtc_delay=frequency_delay_queues['baseline'].estimate
                    scheduler.low_delay_steps=frequency_delay_queues['gpu_emc'].estimate
                if (config.get('m4_enabled') or config.get('m3_fast_enabled')) and mode_delay_queues:
                    rtc_delay=max(q.estimate for q in mode_delay_queues.values())
                if config.get('m3_ordinary_fast_context'):
                    rtc_delay=mode_delay_queues['full:baseline'].estimate
                    scheduler.low_delay_steps=mode_delay_queues['full:gpu_emc'].estimate
                    fast_mode=config.get('m3_ordinary_fast_visual_mode','selective320')
                    scheduler.fast_delays={p:mode_delay_queues[fast_mode+':'+p].estimate for p in ('baseline','gpu_emc')}
                if config.get('rtc_fixed_delay_estimate') is not None:
                    rtc_delay=config['rtc_fixed_delay_estimate']
                    if frequency_delay_queues is not None:scheduler.low_delay_steps=rtc_delay
                decision = scheduler.decide(now=time.monotonic(),step=step,queue=queue,
                                            rows=rows,inflight=future is not None,rtc_delay_steps=rtc_delay)
                decision = scheduler.control_clocks(decision,inflight=future is not None)
                due = decision['start_now'] and future is None and step-previous_request_step >= q
                base._trace(stream, 'm3_decision', step=step, **decision)
            if (rtc_adapter is not None and config.get('rtc_wait_for_delay_feasibility')
                    and future is None and step > 0 and due):
                recent_rows = rows[-16:]
                periods = [b['action_end_monotonic'] - a['action_end_monotonic'] - b.get('critical_wait_seconds', 0.)
                           for a, b in zip(recent_rows, recent_rows[1:])]
                measured_period = float(np.median(periods)) if periods else period
                # Preserve the audited d<=s constraint; q is a minimum request
                # interval, not a promise to overlap two inference requests.
                due = step - previous_request_step >= rtc_adapter.estimate_delay_steps(measured_period)
            if refresh_policy == 'post_commit' and future is not None:
                due = False  # Next q-step window starts only after this result commits.
            if config.get('m3_watchdog_case') and len(commits)>=2:
                due=False  # The short fault window ends after this commit.
            if due:
                if future is None:
                    future, current_request = submit(observation, step)
                    previous_request_step = step
                    if future is None:
                        prefix_rejection = current_request
                        frames.pop()  # No action was executed for this tick.
                        status = 'infeasible_prefix_plan'
                        break
                    if synchronous:
                        future.result()  # Explicit sync diagnostic: no env steps while waiting.
                        harvest(step)
                else:
                    missed_triggers.append(step)
                    base._trace(stream, 'busy_trigger', task_id=task['task_id'], q=q,
                                trial_index=trial_index, condition=condition, step=step)
            critical_wait_seconds = 0.0
            if config.get('m3_critical_wait') and future is not None:
                from perceptbudget.critical_wait import should_wait, wait_target
                if should_wait(current_request, step):
                    review = dict(wait=True, reason='point_event')
                    active_item = queue.get(step)
                    if config.get('m3_critical_wait_phases') and active_item is not None:
                        owner = requests[active_item[1]]
                        review = phase_wait.review(current_request, owner['actions'],
                            owner['observation_step'], scheduler.events_config)
                    current_request['critical_wait_review'] = review
                else:
                    review = dict(wait=False)
                if review['wait']:
                    waiting_request = current_request
                    wait_started = time.monotonic()
                    target = wait_target(waiting_request)
                    # No env.step and no queue consumption while waiting. Poll
                    # through harvest so the normal request watchdog stays live.
                    while future is not None:
                        harvest(step)
                        if future is not None:
                            time.sleep(.002)
                    critical_wait_seconds = time.monotonic() - wait_started
                    phase_wait.complete(review)
                    waiting_request['critical_wait'] = dict(step=step, target_step=target,
                        seconds=critical_wait_seconds, simulation_frozen=True)
                    base._trace(stream, 'critical_wait', step=step,
                        prediction_id=waiting_request['prediction_id'], target_step=target,
                        seconds=critical_wait_seconds)
                    deadline += critical_wait_seconds
            queued = buffer_count(queue, step)
            item = queue.pop(step, None)
            if scheduler is not None and not scheduler.admitted(item, step, time.monotonic()):
                item = None
            if item is None:
                underflows.append(step)
                action = np.zeros(7, dtype=np.float32)
                source = None
            else:
                if rtc_adapter is not None and hasattr(rtc_adapter,'validate_execution'):
                    rtc_adapter.validate_execution(requests[item[1]],step)
                action = np.asarray(item[0], dtype=np.float32)
                source = {'prediction_id': item[1], 'origin_step': item[2],
                          'offset': item[3], 'beyond_sync_q_prefix': item[3] >= q}
                if action_blend_enabled and len(item) > 4:
                    source['blend'] = item[4]
                elif rtc_fusion_enabled and len(item) > 4:
                    source['rtc_action_fusion'] = item[4]
                elif alignment_policy and len(item) > 4:
                    source['action_alignment'] = item[4]
            action_start = time.monotonic()
            planned_action = action.copy()
            inference_active_at_action = future is not None
            inference_prediction_id_at_action=(current_request['prediction_id'] if inference_active_at_action else None)
            inference_mode_at_action=(current_request.get('m3_fast_choice',{}).get('mode','Full')
                                      if inference_active_at_action else None)
            arm_gain = pending_arm_gain if future is not None else 1.0
            action = action.copy()
            action[:6] *= arm_gain
            if capture is None:
                observation, reward, done, info = env.step(action)
            else:
                observation, reward, done, info = capture.step(env, action,
                    control_step=step, inference_active=inference_active_at_action)
            action_end = time.monotonic()
            after = snapshot_scene(env, observation)
            success = bool(env.check_success())
            harvest(step + 1, {'step': step, 'action': action.tolist(),
                               'before': before, 'after': after})
            terminal = success or done or step + 1 == task['max_steps']
            deadline += period
            overrun = max(0.0, time.monotonic() - deadline)
            row = {
                'step': step,
                'trial_index': trial_index,
                'condition': condition,
                'tick_monotonic': tick,
                'critical_wait_seconds': critical_wait_seconds,
                'action_start_monotonic': action_start,
                'action_end_monotonic': action_end,
                'before': before,
                'after': after,
                'action': action.tolist(),
                'planned_action': planned_action.tolist(),
                'applied_arm_gain': arm_gain,
                'source': source,
                'queue_before_pop_actions': queued,
                'late_hold': False,
                'queue_after_step_actions': buffer_count(queue, step + 1),
                'inference_active': future is not None,
                'inference_active_at_action': inference_active_at_action,
                'inference_prediction_id_at_action': inference_prediction_id_at_action,
                'inference_mode_at_action': inference_mode_at_action,
                **({'camera_render_skipped': capture.skips_render(inference_active_at_action)} if capture is not None else {}),
                'deadline_overrun_seconds': overrun,
                'success': success,
                'env_done': bool(done),
                'reward': float(reward),
            }
            rows.append(row)
            base._trace(stream, 'control_step', task_id=task['task_id'], q=q, **row)
            if visual_monitor is not None:
                from perceptbudget.visual_monitor import continuation_eligible,fresh_wrist_image
                visual_monitor.on_control(action)
                eligible=continuation_eligible(decision,inflight=inference_active_at_action or future is not None)
                if (not terminal and eligible and action[6]>=.2 and visual_monitor.due(step+1)):
                    image,acquisition=fresh_wrist_image(env)
                    sample=visual_monitor.observe(image,after['raw_state'],action,
                        step=step+1,eligible=True)
                    sample.update(sample_id=len(visual_monitor.samples)-1,acquisition=acquisition,
                        source_prediction_id=source['prediction_id'] if source else None,
                        decision_step=step,inference_active=False)
                    visual_images.append(visual_monitor.cv.cvtColor(image,visual_monitor.cv.COLOR_RGB2GRAY))
                    base._trace(stream,'visual_monitor',**sample)
                    scheduler.arm_visual_alarm(sample)
            if step % 20 == 0:
                base._trace(stream, 'device_state', q=q, trial_index=trial_index,
                            condition=condition, step=step,
                            timestamp_monotonic=time.monotonic(), values=device_state())
            if config.get('m3_watchdog_case') and len(commits)>=2:
                from perceptbudget.watchdog import fault_window_recovered
                if fault_window_recovered(row,commits):
                    status = 'fault_window_completed'
                    break
            if terminal:
                status = 'success' if success else 'timeout' if step + 1 == task['max_steps'] else 'failure'
                break
            if time.monotonic() > deadline:
                deadline = time.monotonic()
            else:
                time.sleep(max(0, deadline - time.monotonic()))
        stop = time.monotonic()
    except BaseException as exc:
        if request_watchdog is not None:
            abort_time=time.monotonic()
            base._trace(stream,'watchdog_abort',error=type(exc).__name__,
                timestamp_monotonic=abort_time,request=request_watchdog.record,executed_steps=len(rows))
            base._write_json_atomic(run_dir/'watchdog_abort.json',dict(
                status='aborted',error=type(exc).__name__,request=request_watchdog.record,
                executed_steps=len(rows),underflow_steps=underflows,requests=requests,
                power_until_abort=base._power_result(monitor,start,abort_time),
                scope='Measured through local abort; outer process termination/cleanup energy not included'))
        raise
    finally:
        if stop is None:
            stop = time.monotonic()
        try:
            if request_watchdog is not None and future is not None and status!='invalid':
                from perceptbudget.watchdog import finish_watched_request
                finish_watched_request(future,current_request,request_watchdog)
        finally:
            terminal_low = None
            try:
                if config.get('m4_common_low_tail_seconds') and status != 'invalid':
                    endpoint_clocks = scheduler.clocks if scheduler is not None else fixed_clocks
                    if endpoint_clocks is None:
                        raise RuntimeError('Common LOW endpoint requires runtime clocks')
                    executor.shutdown(wait=True)
                    from perceptbudget.terminal_energy import settle_low
                    terminal_low = settle_low(endpoint_clocks, seconds=config['m4_common_low_tail_seconds'])
            finally:
                try:
                    if scheduler is not None:
                        scheduler.stop_clocks()
                finally:
                    try:
                        if fixed_clocks is not None:
                            fixed_clocks.close()
                    finally:
                        drain = _drain_executor(executor, monitor, start, stop, terminal_low=terminal_low)
                        if request_watchdog is not None and current_request is not None:
                            request_watchdog.complete(current_request.get('ready_monotonic',time.monotonic()))
        base._trace(stream, 'episode_drain', task_id=task['task_id'], q=q,
                    trial_index=trial_index, condition=condition, **drain)

        if env_audit is not None:
            env_context.__exit__(None, None, None)
        if rtc_adapter is not None and config.get('release_rtc_graph_before_artifacts'):
            released = rtc_adapter.release_graph()
            base._trace(stream, 'rtc_graph_released_before_artifacts', details=released)
        base._write_json_atomic(run_dir / f'q{q}{artifact_tag}_late_hold.json', {
            'late_policy': late_policy, 'pauses': pauses,
            'count': len(pauses), 'steps': sum(p['steps'] for p in pauses),
            'wall_seconds': sum(p['wall_seconds'] for p in pauses)})
        for request in requests:
            if 'actions' in request and (not any((c['prediction_id'] == request['prediction_id'] for c in commits))):
                base._trace(stream, 'uncommitted_prediction', task_id=task['task_id'], q=q, trial_index=trial_index, condition=condition, request=request, reason='completed after final harvest; not executed')
        base._write_json_atomic(run_dir / f'q{q}{artifact_tag}_requests.json', requests)
        if visual_monitor is not None:
            base._write_json_atomic(run_dir/f'q{q}{artifact_tag}_visual_monitor.json',visual_monitor.samples)
            if visual_images:
                np.savez_compressed(run_dir/f'q{q}{artifact_tag}_visual_monitor_frames.npz',
                    frames=np.asarray(visual_images),frame_steps=np.asarray([s['step'] for s in visual_monitor.samples]))
        if frames:
            if capture is None:
                frames.append(np.concatenate((observation['agentview_image'],
                                             observation['robot0_eye_in_hand_image']), axis=1))
            np.savez_compressed(run_dir / f'q{q}{artifact_tag}_frames.npz',
                                frames=np.asarray(frames), control_period_s=period,
                                **({'frame_steps': np.asarray(capture.frame_steps),
                                    'note': 'raw LIBERO request images only; sparse control-step indices; rotate180 for viewing'}
                                   if capture is not None else
                                   {'note': 'raw LIBERO images; rotate180 for viewing; final frame included'}))
    busy = sum(max(0, min(stop, r['inference_ended_monotonic']) -
                   max(start, r['inference_started_monotonic'])) for r in requests)
    intervals = [b['observation_step'] - a['observation_step'] for a, b in zip(requests, requests[1:])]
    result = {
        'task_id': task['task_id'],
        'task_name': task['task_name'],
        'q': q,
        'status': status,
        'steps': len(rows),
        'simulated_seconds': len(rows) * period,
        'real_seconds': stop - start,
        'critical_wait_count': sum(r.get('critical_wait_seconds', 0) > 0 for r in rows),
        'critical_wait_seconds': sum(r.get('critical_wait_seconds', 0) for r in rows),
        'start_monotonic': start,
        'stop_monotonic': stop,
        'inference_busy_seconds': busy,
        'inference_duty': busy / (stop - start),
        'busy_trigger_steps': missed_triggers,
        'actual_refresh_intervals_steps': intervals,
        'prefix_lock': False,
        'prefix_len': 0,
        'adaptive_prefix': False,
        'trial_index': trial_index,
        'condition': condition,
        'action_blend': action_blend_enabled,
        'rtc_action_fusion': rtc_fusion_enabled,
        'rtc_vjp_guidance': rtc_adapter is not None,
        'seam_vls_guidance': False,
        'alignment_policy': alignment_policy,
        'refresh_policy': refresh_policy,
        'advance_handoff': False,
        'prefix_rtc_merge': False,
        'pending_arm_gain': pending_arm_gain,
        'synchronous_recording': synchronous,
        'prefix_plan_rejection': prefix_rejection,
        'late_discarded_predictions': sum((bool(r.get('late_discarded')) for r in requests)),
        'late_policy': late_policy,
        'late_hold_pauses': pauses,
        'late_hold_count': len(pauses),
        'late_hold_steps': sum((p['steps'] for p in pauses)),
        'late_hold_wall_seconds': sum((p['wall_seconds'] for p in pauses)),
        'underflow_steps': underflows,
        'commits': commits,
        'tail_execution_steps': sum((bool(r['source'] and r['source']['beyond_sync_q_prefix']) for r in rows)),
        'post_bootstrap_control_hz': (len(rows) - 1) / (rows[-1]['action_end_monotonic'] - rows[0]['action_end_monotonic']) if len(rows) > 1 else None,
        'control_deadline_overruns': sum((r['deadline_overrun_seconds'] > 0 for r in rows)),
        'initial_sim_state_sha256': hashlib.sha256(np.asarray(rows[0]['before']['sim_state'], dtype=np.float64).tobytes()).hexdigest() if rows else None,
        'initial_frames_sha256': hashlib.sha256(np.asarray(frames[0]).tobytes()).hexdigest() if frames else None,
        'initial_raw_state': rows[0]['before']['raw_state'] if rows else None,
        'underflow_nominal_seconds': len(underflows) * period,
        'underflow_env_step_real_seconds': sum((r['action_end_monotonic'] - r['action_start_monotonic'] for r in rows if r['source'] is None)),
        'power': base._power_result(monitor, start, stop),
        **drain,
        'field_checks': {'full_predictions': all((len(r.get('actions', [])) == config['H'] for r in requests)), 'pre_post_state': all(('raw_state' in r['before'] and 'raw_state' in r['after'] for r in rows)), 'scene_contacts': all(('contacts' in r['after'] for r in rows)), 'frames': (len(frames) == len(requests) and capture.frame_steps == [r['observation_step'] for r in requests]) if capture is not None else len(frames) == len(rows) + 1},
    }
    if capture is not None:
        result['continue_capture'] = capture.summary()
    if visual_monitor is not None:
        samples=visual_monitor.samples
        if capture is not None:
            result['continue_capture']['scope']='VLA request images and separate sparse wrist-monitor images; no physics/state step skipped'
        result['visual_monitor']=dict(policy=config['visual_monitor'],samples=len(samples),
            alarms=sum(s['alarm'] for s in samples),responses=sum('visual_monitor_response' in r for r in requests),
            acquisition_seconds=sum(s['acquisition']['stop_monotonic']-s['acquisition']['start_monotonic'] for s in samples),
            detector_seconds=sum(s['detector_seconds'] for s in samples),
            cost_scope='Included in whole-task wall time and board energy; request images remain separate')
    if env_audit is not None:
        result['env_forward_optimization'] = env_audit
    print(json.dumps({k: result[k] for k in ('q', 'status', 'steps', 'real_seconds', 'post_bootstrap_control_hz', 'tail_execution_steps', 'field_checks')}, ensure_ascii=False), flush=True)
    return result
