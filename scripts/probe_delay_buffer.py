#!/usr/bin/env python3
"""Actual-delay LIBERO diagnostic episodes; not a validated async policy."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_q as base
from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import load_libero_norm_stats, raw_libero_state
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


def advance_prefix_due(step, previous_commit_step, q, prefix_len):
    """Request early enough to target the next q-step handoff (no concurrency)."""
    if q <= prefix_len:
        raise ValueError('handoff interval must exceed locked prefix length')
    return step >= previous_commit_step + q - prefix_len


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


def prefix_hold_due(request, step, policy):
    return (policy == 'hold' and request is not None
            and 'scheduled_commit_step' in request
            and step >= request['scheduled_commit_step'])


def commit_prefix_hold(queue, actions, origin, step, prediction_id, raw_actions, length):
    """Resume after the executed prefix; consume delay from the tail, not H."""
    remaining = origin + len(actions) - step
    if remaining <= 0:
        raise RuntimeError('late prefix result fully expired: original origin+H deadline reached')
    if step < origin + length:
        raise ValueError('cannot commit before locked prefix ends')
    tail = len(actions) - length - remaining
    overwritten = sum(target in queue for target in range(step, origin + len(actions)))
    # No old tail may survive the handoff, including beyond the original deadline.
    for target in list(queue):
        if target >= step:
            del queue[target]
    for index in range(remaining):
        offset = length + index
        queue[step + index] = (actions[offset], prediction_id, origin, offset,
                               raw_actions[offset])
    return length, overwritten, {'discarded_prefix_actions': length,
                                'discarded_tail_actions': tail,
                                'remaining_actions': remaining,
                                'original_deadline_step': origin + len(actions),
                                'source_offset': length}


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


def _drain_executor(executor, monitor, start, stop):
    """Keep sampling through worker completion; exclude subsequent artifact work."""
    try:
        executor.shutdown(wait=True)
        drain_stop = time.monotonic()
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
    }


def run_episode(env, task, q, model, stats, config, np, torch, run_dir, stream,
                prefix_adapter=None, *, trial_index=1, condition='crop_overwrite',
                action_blend_enabled=False, rtc_fusion_enabled=False, rtc_adapter=None,
                seam_adapter=None,
                artifact_tag='', alignment_policy=None, refresh_policy='boundary',
                pending_arm_gain=1.0, synchronous=False, step_hook=None):
    if synchronous and (prefix_adapter or action_blend_enabled or rtc_fusion_enabled or
                        alignment_policy is not None or pending_arm_gain != 1.0):
        raise ValueError('Synchronous recording must execute an unchanged fresh prefix')
    if alignment_policy is not None and (prefix_adapter or action_blend_enabled or rtc_fusion_enabled):
        raise ValueError('phase alignment is a separate diagnostic condition')
    if rtc_adapter is not None and (prefix_adapter or action_blend_enabled or rtc_fusion_enabled
                                    or alignment_policy is not None or synchronous):
        raise ValueError('RTC VJP rollout is an isolated asynchronous condition')
    if seam_adapter is not None and (prefix_adapter or action_blend_enabled or rtc_fusion_enabled
                                     or rtc_adapter or alignment_policy is not None
                                     or synchronous or pending_arm_gain != 1.0):
        raise ValueError('SEAM async uses ordinary absolute-step cropping and full-speed actions')
    if not 0 < pending_arm_gain <= 1 or (prefix_adapter is not None and pending_arm_gain != 1):
        raise ValueError('invalid pending gain; committed prefixes must remain exact')
    observation, hz = base._prepare_env_episode(env, task, np)
    assert hz == 20
    base._seed_process(7, np, torch)
    period = 1 / hz
    queue, requests, commits, rows, frames, raw_chunks = {}, [], [], [], [], []
    if config.get('disk_backed_frames'):
        from perceptbudget.trace_frames import DiskFrames
        frames = DiskFrames(run_dir / f'q{q}{artifact_tag}_frames_spool.npy', task['max_steps']+1, np)
    future = None
    previous_request_step = 0
    previous_commit_step = 0
    delay_queue = None
    if config.get('rtc_paper_schedule'):
        from perceptbudget.rtc_schedule import DelayQueue
        delay_queue = DelayQueue(config['rtc_initial_delay_steps'], config['rtc_delay_buffer_size'])
    current_request = None
    missed_triggers, underflows = [], []
    selector = None
    prefix_rejection = None
    late_policy = config.get('prefix_lock', {}).get('late_policy', 'discard')
    advance_handoff = config.get('prefix_lock', {}).get('advance_handoff', False)
    prefix_fusion = config.get('prefix_lock', {}).get('rtc_merge', False)
    if prefix_fusion and (not advance_handoff or late_policy != 'hold'):
        raise ValueError('prefix RTC merge requires advance handoff and late hold')
    if advance_handoff and (prefix_adapter is None or q <= prefix_adapter.prefix_len):
        raise ValueError('advance handoff requires prefix locking and q > prefix_len')
    if late_policy not in ('discard', 'hold') or (late_policy == 'hold' and prefix_adapter is None):
        raise ValueError('late hold requires prefix locking')
    pauses, active_pause = [], None
    if config.get('adaptive_prefix'):
        from perceptbudget.prefix_timing import PrefixTiming
        selector = PrefixTiming(latency_seconds=config['adaptive_prefix']['initial_latency_s'])
    monitor = PowerMonitor(poll_interval=0.1)
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
        snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
        request = {'prediction_id': len(requests), 'observation_step': step,
                   'refresh_anchor': 'advance_handoff' if advance_handoff else 'commit' if refresh_policy == 'post_commit' else 'request',
                   'refresh_anchor_step': previous_commit_step if advance_handoff or refresh_policy == 'post_commit' else previous_request_step,
                   'target_handoff_step': previous_commit_step + q if advance_handoff and step else None,
                   'observation_monotonic': observation_started,
                   'input_ready_monotonic': time.monotonic(),
                   'model_state': snapshot.get('state'),
                   'observation_raw_state': snapshot['raw_state'].tolist(),
                   'queue_before_submit_actions': buffer_count(queue, step)}
        seam_tail, seam_executed, seam_origin, seam_delay = None, 0, None, 0
        if seam_adapter is not None and step > 0:
            from perceptbudget.seam_rollout import queued_raw_tail
            seam_tail, seam_executed, seam_origin = queued_raw_tail(queue, step, config['H'])
            recent_rows = rows[-16:]
            periods = [b['action_end_monotonic'] - a['action_end_monotonic']
                       for a, b in zip(recent_rows, recent_rows[1:])]
            observed_period = float(np.median(periods)) if periods else period
            if seam_tail is not None:
                seam_delay = seam_adapter.estimate_delay_steps(observed_period)
            request['seam_async'] = {
                'old_chunk_origin_step': seam_origin,
                'control_period_estimate_seconds': observed_period,
                'estimated_delay_steps': seam_delay,
                'guidance_timebase': 'observation step; preserve full old-tail window',
                'prior_unavailable': seam_tail is None,
                'reason': 'old queue exhausted' if seam_tail is None else 'valid old tail',
            }
        rtc_previous_raw = None
        if rtc_adapter is not None and step > 0:
            executed = step - previous_request_step
            recent_rows = rows[-16:]
            periods = [b['action_end_monotonic'] - a['action_end_monotonic']
                       for a, b in zip(recent_rows, recent_rows[1:])]
            period_estimate = float(np.median(periods)) if periods else period
            delay = config.get('controlled_delay_steps', rtc_adapter.estimate_delay_steps(period_estimate))
            if delay_queue is not None:
                delay = delay_queue.estimate
            if config.get('rtc_fixed_delay_estimate') is not None:
                delay=config['rtc_fixed_delay_estimate']
            feasible = (0 <= delay <= config['H'] - executed if delay_queue is not None
                        else delay <= executed <= config['H'] - delay)
            if not feasible:
                raise RuntimeError(f'RTC infeasible before inference: d={delay}, '
                                   f's={executed}, H={config["H"]}')
            if step not in queue:
                raise RuntimeError(f'RTC old-action target missing at control step {step}')
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
                'delay_estimate_mode': ('fixed' if config.get('rtc_fixed_delay_estimate') is not None else 'history_max' if delay_queue is not None else 'latency'),
            }
            if delay_queue is not None:
                request['rtc_guidance']['delay_history_steps'] = list(delay_queue.values)
                request['rtc_guidance']['initialization_d_exceeds_s'] = delay > executed
        raw_prefix = None
        prefix_len = None
        if prefix_adapter is not None and step > 0:
            prefix_len = prefix_adapter.prefix_len
            if selector is not None:
                decision = selector.choose(q=q, buffer_actions=buffer_count(queue, step))
                request['prefix_decision'] = decision
                prefix_len = decision['selected']
                if prefix_len is None:
                    base._trace(stream, 'prefix_plan_infeasible', task_id=task['task_id'],
                                q=q, step=step, decision=decision)
                    return None, request
            if buffer_count(queue, step) < prefix_len:
                raise RuntimeError(f'insufficient queued actions for committed prefix: step={step}, '
                                   f'available={buffer_count(queue, step)}, required={prefix_len}; no horizon extension')
            raw_prefix = np.asarray([queue[step + i][4] for i in range(prefix_len)])
            request.update(committed_prefix_raw=raw_prefix.tolist(),
                           committed_prefix_actions=[np.asarray(queue[step + i][0]).tolist() for i in range(prefix_len)],
                           committed_interval_steps=[step, step + prefix_len],
                           prefix_len=prefix_len,
                           scheduled_commit_step=step + prefix_len)
        requests.append(request)

        def predict():
            if seam_adapter is not None:
                actions = seam_adapter.predict(
                    snapshot, task['instruction'], seam_tail,
                    # Keep the synchronous VLS window at the observation origin.
                    # Committed old actions also constrain the new continuation;
                    # only execution at commit discards the actual expired rows.
                    executed=seam_executed, delay_estimate=0, request=request)
            elif rtc_adapter is not None and step == 0:
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
            elif prefix_adapter is None:
                actions = base._infer_actions(model, snapshot, task['instruction'],
                                              config['H'], request)
            else:
                actions, raw = prefix_adapter.predict(snapshot, task['instruction'], request, raw_prefix,
                                                     prefix_len=prefix_len)
                request['actions_raw'] = raw.tolist()
            request['actions'] = actions.tolist()
            request['ready_monotonic'] = time.monotonic()
            return actions

        return executor.submit(predict), request

    def harvest(step, completed_row=None):
        nonlocal future, current_request, previous_commit_step, active_pause
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
        if seam_adapter is not None:
            guidance = current_request['seam_guidance']
            actual_delay = max(0, step - current_request['observation_step'])
            guidance['actual_delay_steps'] = actual_delay
            guidance['surviving_guided_actions'] = max(
                0, guidance['guided_start'] + guidance['guided_window']
                - max(actual_delay, guidance['guided_start']))
        if rtc_adapter is not None and current_request.get('rtc_guidance'):
            guidance = current_request['rtc_guidance']
            guidance['actual_delay_steps'] = max(0, step - current_request['observation_step'])
            if delay_queue is not None and not current_request.get('delay_queue_observed'):
                current_request['delay_queue_observed'] = delay_queue.observe(
                    guidance['actual_delay_steps'],
                    observation_step=current_request['observation_step'])
                if current_request['observation_step'] == 0:
                    guidance['delay_history_excluded_reason'] = 'bootstrap A_init is not an inference-loop delay sample'
            guidance['observed_inference_seconds'] = (
                current_request.get('inference_ended_monotonic', time.monotonic())
                - current_request.get('inference_started_monotonic',
                                      current_request['input_ready_monotonic']))
        if selector is not None and not current_request.get('latency_observed'):
            selector.observe_latency(current_request['ready_monotonic'] - current_request['observation_monotonic'])
            current_request['latency_observed'] = True
        current_request.setdefault('ready_observed_step', step)
        scheduled = current_request.get('scheduled_commit_step', 0)
        if step < scheduled:
            return  # Never replace a prefix already committed for execution.
        if prefix_adapter is not None and scheduled and step > scheduled and late_policy == 'discard':
            future.result()  # Propagate model errors even for a late result.
            base._trace(stream, 'late_prefix_result_discarded', task_id=task['task_id'], q=q,
                        trial_index=trial_index, condition=condition,
                        request=current_request, step=step, scheduled=scheduled)
            current_request['late_discarded'] = True
            future, current_request = None, None
            return
        actions = ready_actions
        before = buffer_count(queue, step)
        origin = current_request['observation_step']
        reference_remaining = reference_prefix_count(queue, step, q)
        blend = None
        rtc_fusion = None
        alignment = None
        merge_start = time.monotonic()
        if late_policy == 'hold' and scheduled:
            old_queue = dict(queue) if prefix_fusion else None
            expired, overwritten, alignment = commit_prefix_hold(
                queue, actions, origin, step, current_request['prediction_id'],
                current_request['actions_raw'], current_request['prefix_len'])
            if prefix_fusion:
                from perceptbudget.prefix_fusion import fuse_prefix_tail
                from flash_rt.core.utils.actions import unnormalize_actions
                fused = fuse_prefix_tail(old_queue, queue, beta=1.2)
                records = []
                for target, data in fused.items():
                    raw = np.asarray(data['raw'], dtype=np.float32)
                    actual = unnormalize_actions(raw, prefix_adapter.model._pipe.norm_stats)[:7]
                    item = queue[target]
                    record = {**data['metadata'], 'target_step': target,
                              'raw': raw.tolist(), 'action': actual.tolist()}
                    queue[target] = (actual, item[1], item[2], item[3], raw, record)
                    records.append(record)
                rtc_fusion = {'variant': 'paper_eq8_9_arm6_clipped_bf16',
                              'fused_steps': sum(r['source_count'] == 2 for r in records),
                              'recorded_tail_steps': len(records), 'records': records,
                              'beta': 1.2, 'arm_dimensions': 6,
                              'gripper_policy': 'latest_prediction'}
        elif alignment_policy is not None:
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
        if active_pause is not None:
            active_pause.update(end_step=step, end_monotonic=time.monotonic(),
                                resumed=True)
            active_pause['wall_seconds'] = active_pause['end_monotonic'] - active_pause['start_monotonic']
            base._trace(stream, 'late_hold_end', q=q, **active_pause)
            active_pause = None
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
        future, current_request = submit(observation, 0)
        future.result()  # Only bootstrap blocks; no control ticks yet.
        harvest(0)
        deadline = time.monotonic()
        for step in range(task['max_steps']):
            if step_hook is not None:
                observation, hook_event = step_hook(env, observation, step)
                if hook_event is not None:
                    base._trace(stream, 'external_disturbance', task_id=task['task_id'],
                                q=q, step=step, details=hook_event)
            tick = time.monotonic()
            if selector is not None and rows:
                selector.observe_tick(tick - rows[-1]['tick_monotonic'])
            before = snapshot_scene(env, observation)
            frames.append(np.concatenate((observation['agentview_image'],
                                         observation['robot0_eye_in_hand_image']), axis=1))
            harvest(step)
            anchor_step = previous_commit_step if refresh_policy == 'post_commit' else previous_request_step
            due = refresh_due(step, anchor_step, q, refresh_policy)
            if (rtc_adapter is not None and config.get('rtc_wait_for_delay_feasibility')
                    and future is None and step > 0 and due):
                recent_rows = rows[-16:]
                periods = [b['action_end_monotonic'] - a['action_end_monotonic']
                           for a, b in zip(recent_rows, recent_rows[1:])]
                measured_period = float(np.median(periods)) if periods else period
                # Preserve the audited d<=s constraint; q is a minimum request
                # interval, not a promise to overlap two inference requests.
                due = step - previous_request_step >= rtc_adapter.estimate_delay_steps(measured_period)
            if advance_handoff:
                due = (future is None and advance_prefix_due(
                    step, previous_commit_step, q, prefix_adapter.prefix_len))
            if refresh_policy == 'post_commit' and future is not None:
                due = False  # Next q-step window starts only after this result commits.
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
            queued = buffer_count(queue, step)
            holding = prefix_hold_due(current_request, step, late_policy)
            if holding and active_pause is None:
                active_pause = {'prediction_id': current_request['prediction_id'],
                                'start_step': step, 'start_monotonic': time.monotonic(),
                                'steps': 0, 'resumed': False}
                pauses.append(active_pause)
                base._trace(stream, 'late_hold_start', q=q, **active_pause)
            item = queue.pop(step, None)
            if holding:
                active_pause['steps'] += 1
                action = np.zeros(7, dtype=np.float32)
                source = None
            elif item is None:
                underflows.append(step)
                # Explicit zero-translation, zero-rotation, neutral gripper command.
                action = np.zeros(7, dtype=np.float32)
                source = None
            else:
                action = np.asarray(item[0], dtype=np.float32)
                source = {'prediction_id': item[1], 'origin_step': item[2],
                          'offset': item[3], 'beyond_sync_q_prefix': item[3] >= q}
                if prefix_fusion and len(item) > 5:
                    source['prefix_rtc_merge'] = item[5]
                elif action_blend_enabled and len(item) > 4:
                    source['blend'] = item[4]
                elif rtc_fusion_enabled and len(item) > 4:
                    source['rtc_action_fusion'] = item[4]
                elif alignment_policy and len(item) > 4:
                    source['action_alignment'] = item[4]
            action_start = time.monotonic()
            planned_action = action.copy()
            inference_active_at_action = future is not None
            arm_gain = pending_arm_gain if future is not None else 1.0
            action = action.copy()
            action[:6] *= arm_gain
            observation, reward, done, info = env.step(action)
            action_end = time.monotonic()
            after = snapshot_scene(env, observation)
            success = bool(env.check_success())
            # Harvest only on control thread, also after a step, never block it.
            harvest(step + 1, {'step': step, 'action': action.tolist(),
                               'before': before, 'after': after})
            terminal = success or done or step + 1 == task['max_steps']
            deadline += period
            overrun = max(0.0, time.monotonic() - deadline)
            row = {'step': step, 'trial_index': trial_index, 'condition': condition,
                   'tick_monotonic': tick,
                   'action_start_monotonic': action_start, 'action_end_monotonic': action_end,
                   'before': before, 'after': after, 'action': action.tolist(),
                   'planned_action': planned_action.tolist(), 'applied_arm_gain': arm_gain,
                   'source': source, 'queue_before_pop_actions': queued,
                   'late_hold': holding,
                   'queue_after_step_actions': buffer_count(queue, step + 1),
                   'inference_active': future is not None,
                   'inference_active_at_action': inference_active_at_action,
                   'deadline_overrun_seconds': overrun,
                   'success': success, 'env_done': bool(done), 'reward': float(reward)}
            rows.append(row)
            base._trace(stream, 'control_step', task_id=task['task_id'], q=q, **row)
            if step % 20 == 0:
                base._trace(stream, 'device_state', q=q, trial_index=trial_index,
                            condition=condition, step=step,
                            timestamp_monotonic=time.monotonic(), values=device_state())
            if terminal:
                status = 'success' if success else 'timeout' if step + 1 == task['max_steps'] else 'failure'
                break
            # No catch-up bursts. Slow simulation is recorded, not called real 20Hz.
            if time.monotonic() > deadline:
                deadline = time.monotonic()
            else:
                time.sleep(max(0, deadline - time.monotonic()))
        stop = time.monotonic()
    finally:
        if stop is None:
            stop = time.monotonic()
        if active_pause is not None:
            active_pause.update(end_step=len(rows), end_monotonic=stop,
                                wall_seconds=stop - active_pause['start_monotonic'])
        drain = _drain_executor(executor, monitor, start, stop)
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
            if 'actions' in request and not any(c['prediction_id'] == request['prediction_id'] for c in commits):
                base._trace(stream, 'uncommitted_prediction', task_id=task['task_id'], q=q,
                            trial_index=trial_index, condition=condition,
                            request=request, reason=('not committed before task termination; may be ready before committed prefix ends'
                                                     if prefix_adapter else 'completed after final harvest; not executed'))
        base._write_json_atomic(run_dir / f'q{q}{artifact_tag}_requests.json', requests)
        if frames:
            frames.append(np.concatenate((observation['agentview_image'],
                                         observation['robot0_eye_in_hand_image']), axis=1))
            np.savez_compressed(run_dir / f'q{q}{artifact_tag}_frames.npz',
                                frames=np.asarray(frames), control_period_s=period,
                                note='raw LIBERO images; rotate180 for viewing; final frame included')
    busy = sum(max(0, min(stop, r['inference_ended_monotonic']) -
                   max(start, r['inference_started_monotonic'])) for r in requests)
    intervals = [b['observation_step'] - a['observation_step'] for a, b in zip(requests, requests[1:])]
    result = {'task_id': task['task_id'], 'task_name': task['task_name'], 'q': q,
              'status': status, 'steps': len(rows), 'simulated_seconds': len(rows) * period,
              'real_seconds': stop - start, 'start_monotonic': start, 'stop_monotonic': stop,
              'inference_busy_seconds': busy, 'inference_duty': busy / (stop - start),
              'busy_trigger_steps': missed_triggers, 'actual_refresh_intervals_steps': intervals,
              'prefix_lock': prefix_adapter is not None,
              'prefix_len': None if selector else prefix_adapter.prefix_len if prefix_adapter else 0,
              'adaptive_prefix': selector is not None,
              'trial_index': trial_index, 'condition': condition,
              'action_blend': action_blend_enabled,
              'rtc_action_fusion': rtc_fusion_enabled,
              'rtc_vjp_guidance': rtc_adapter is not None,
              'seam_vls_guidance': seam_adapter is not None,
              'alignment_policy': alignment_policy,
              'refresh_policy': refresh_policy,
              'advance_handoff': advance_handoff,
              'prefix_rtc_merge': prefix_fusion,
              'pending_arm_gain': pending_arm_gain,
              'synchronous_recording': synchronous,
              'prefix_plan_rejection': prefix_rejection,
              'late_discarded_predictions': sum(bool(r.get('late_discarded')) for r in requests),
              'late_policy': late_policy, 'late_hold_pauses': pauses,
              'late_hold_count': len(pauses),
              'late_hold_steps': sum(p['steps'] for p in pauses),
              'late_hold_wall_seconds': sum(p['wall_seconds'] for p in pauses),
              'underflow_steps': underflows, 'commits': commits,
              'tail_execution_steps': sum(bool(r['source'] and r['source']['beyond_sync_q_prefix']) for r in rows),
              'post_bootstrap_control_hz': (len(rows) - 1) / (rows[-1]['action_end_monotonic'] - rows[0]['action_end_monotonic']) if len(rows) > 1 else None,
              'control_deadline_overruns': sum(r['deadline_overrun_seconds'] > 0 for r in rows),
              'initial_sim_state_sha256': hashlib.sha256(np.asarray(rows[0]['before']['sim_state'], dtype=np.float64).tobytes()).hexdigest() if rows else None,
              'initial_frames_sha256': hashlib.sha256(np.asarray(frames[0]).tobytes()).hexdigest() if frames else None,
              'initial_raw_state': rows[0]['before']['raw_state'] if rows else None,
              'underflow_nominal_seconds': len(underflows) * period,
              'underflow_env_step_real_seconds': sum(r['action_end_monotonic'] - r['action_start_monotonic'] for r in rows if r['source'] is None),
              'power': base._power_result(monitor, start, stop),
              **drain,
              'field_checks': {'full_predictions': all(len(r.get('actions', [])) == config['H'] for r in requests),
                               'pre_post_state': all('raw_state' in r['before'] and 'raw_state' in r['after'] for r in rows),
                               'scene_contacts': all('contacts' in r['after'] for r in rows),
                               'frames': len(frames) == len(rows) + 1}}
    if env_audit is not None:
        result['env_forward_optimization'] = env_audit
    print(json.dumps({k: result[k] for k in ('q', 'status', 'steps', 'real_seconds', 'post_bootstrap_control_hz', 'tail_execution_steps', 'field_checks')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix-lock', action='store_true')
    parser.add_argument('--prefix-len', type=int, choices=(7, 8), default=8)
    parser.add_argument('--late-policy', choices=('discard', 'hold'), default='discard')
    parser.add_argument('--advance-handoff', action='store_true',
                        help='q is handoff interval; request q-prefix_len steps after previous commit')
    parser.add_argument('--prefix-rtc-merge', action='store_true',
                        help='six-arm-dimension two-source RTC-style merge after prefix handoff')
    parser.add_argument('--adaptive-prefix', action='store_true')
    parser.add_argument('--action-blend', action='store_true')
    parser.add_argument('--blend-arm-only', action='store_true',
                        help='paper overlap weights on arm dims 0:6; gripper from new prediction')
    parser.add_argument('--rtc-action-fusion', action='store_true',
                        help='use RTC temporal fusion on arm dims; take gripper from newest prediction')
    parser.add_argument('--rtc-vjp', action='store_true',
                        help='run one task with full RTC denoising guidance')
    parser.add_argument('--horizon', type=int, choices=(20, 30),
                        help='action chunk horizon override for the opt-in RTC rollout')
    parser.add_argument('--denoising-steps', type=int,
                        help='override denoising steps for the opt-in RTC rollout')
    parser.add_argument('--rtc-attention-backend', choices=('math', 'fa2-seqused'),
                        default='math',
                        help='RTC expert attention backend; FA2 seqused masks state padding')
    parser.add_argument('--q', type=int, nargs='+', choices=(5, 8, 10, 12, 15),
                        help='control-step refresh interval(s); default: 8 12')
    parser.add_argument('--compare-action-blend-q8', action='store_true',
                        help='run paired q=8 baseline/blending episodes in one loaded model process')
    parser.add_argument('--pairs', type=int, default=None,
                        help='number of q=8 baseline/blending pairs; default: 3')
    parser.add_argument('--interface-only', action='store_true')
    args = parser.parse_args()
    if args.blend_arm_only and not args.action_blend:
        parser.error('--blend-arm-only requires --action-blend')
    if args.adaptive_prefix:
        args.prefix_lock = True
    if (args.late_policy != 'discard' or args.prefix_len != 8) and not args.prefix_lock:
        parser.error('--late-policy/--prefix-len require --prefix-lock')
    if args.adaptive_prefix and (args.late_policy != 'discard' or args.prefix_len != 8):
        parser.error('fixed late-hold diagnostic excludes adaptive prefix')
    if args.action_blend and args.prefix_lock:
        parser.error('action blending is tested separately from prefix locking')
    if args.rtc_action_fusion and (args.action_blend or args.prefix_lock or args.compare_action_blend_q8):
        parser.error('RTC action fusion is a separate controller-side condition')
    if args.rtc_vjp and (args.action_blend or args.prefix_lock or args.compare_action_blend_q8
                         or args.rtc_action_fusion or args.interface_only):
        parser.error('--rtc-vjp is an isolated task rollout')
    if args.horizon is not None and not args.rtc_vjp:
        parser.error('--horizon is currently limited to the RTC rollout')
    if args.denoising_steps is not None and not args.rtc_vjp:
        parser.error('--denoising-steps is currently limited to the RTC rollout')
    if args.denoising_steps is not None and args.denoising_steps < 1:
        parser.error('--denoising-steps must be a positive integer')
    if args.rtc_attention_backend != 'math' and not args.rtc_vjp:
        parser.error('--rtc-attention-backend is currently limited to the RTC rollout')
    if args.compare_action_blend_q8 and (args.action_blend or args.prefix_lock or args.interface_only):
        parser.error('--compare-action-blend-q8 already runs baseline and blending; do not combine modes')
    if args.compare_action_blend_q8 and args.q is not None:
        parser.error('--compare-action-blend-q8 fixes q=8')
    if args.pairs is not None and not args.compare_action_blend_q8:
        parser.error('--pairs requires --compare-action-blend-q8')
    if args.compare_action_blend_q8 and args.pairs is not None and args.pairs < 1:
        parser.error('--pairs must be at least 1')
    if args.interface_only and not args.prefix_lock:
        parser.error('--interface-only requires --prefix-lock')
    pair_count = args.pairs if args.pairs is not None else 3
    q_values = ([8] if args.compare_action_blend_q8 else
                args.q if args.rtc_vjp and args.q is not None else
                [12 if args.horizon == 30 else 10] if args.rtc_vjp else
                args.q or [8, 12])
    if args.advance_handoff and (not args.prefix_lock or args.adaptive_prefix or
                                args.late_policy != 'hold' or min(q_values) <= args.prefix_len):
        parser.error('--advance-handoff requires fixed prefix, late hold and q > prefix-len')
    if args.prefix_rtc_merge and not args.advance_handoff:
        parser.error('--prefix-rtc-merge requires --advance-handoff')
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
    config = load_config(repo_root=ROOT)
    if args.rtc_vjp and args.horizon is not None:
        config['H'] = args.horizon
    if args.rtc_vjp and args.denoising_steps is not None:
        config['N'] = args.denoising_steps
    if args.rtc_vjp and len(q_values) != 1:
        parser.error('--rtc-vjp accepts exactly one q value per rollout')
    config.update(candidate_q=q_values,
                  execution_mode=('q8_paired_action_blend_comparison' if args.compare_action_blend_q8
                                  else 'rtc_vjp_guidance_single_task' if args.rtc_vjp
                                  else 'prefix_lock_diagnostic' if args.prefix_lock
                                  else 'actual_delay_diagnostic'))
    if args.prefix_lock:
        config['prefix_lock'] = {'length': args.prefix_len, 'full_rtc': False,
                                 'advance_handoff': args.advance_handoff,
                                 'rtc_merge': args.prefix_rtc_merge,
                                 'fusion_method': 'paper_eq8_9_arm6_clipped_bf16' if args.prefix_rtc_merge else None,
                                 'q_semantics': 'target_commit_interval' if args.advance_handoff else 'request_boundary',
                                 'late_policy': args.late_policy,
                                 'early_result': 'wait until committed prefix ends',
                                 'late_result': ('zero arm and gripper command; tick until ready; resume after prefix; crop tail at origin+H'
                                                 if args.late_policy == 'hold' else 'discard and record; never silently claim prefix agreement')}
    if args.adaptive_prefix:
        config['adaptive_prefix'] = {'lengths': [5, 6, 7, 8], 'margin_s': 0.005,
                                     'latency_estimator': 'recent16 nearest-rank p80',
                                     'tick_estimator': 'recent16 nearest-rank p20; at least nominal50ms',
                                     'no_feasible_choice': 'stop episode explicitly; no silent clamp'}
        config['execution_mode'] = 'adaptive_prefix_diagnostic'
    if args.action_blend or args.compare_action_blend_q8:
        config['action_blend'] = {
            'beta': 1.2, 'source': 'https://arxiv.org/html/2609.22587v1#S4.SS3',
            'equations': [8, 9], 'action_dimensions': list(range(6 if args.blend_arm_only else 7)),
            'gripper_policy': 'latest_prediction' if args.blend_arm_only else 'same_blend_as_arm',
            'alignment': 'absolute observation-step grid, expired new prefix discarded',
            'event_estimator': False, 'prefix_lock': False,
            'N1_policy': 'direct new action (paper formula requires N>=2)',
            'claim': 'blending mechanism transplant, not complete paper reproduction'}
        config['execution_mode'] = 'action_blend_diagnostic'
    if args.rtc_action_fusion:
        config['rtc_action_fusion'] = {
            'source': 'FlashRT host-side rtc_temporal_fusion equations',
            'weight': 'exp(-decay * abs(old_index - latest_index))',
            'combine': 'normalized weighted mean over raw overlapping chunks',
            'decay': 0.1, 'max_chunks': 3,
            'arm_dimensions': list(range(6)),
            'gripper_policy': 'latest_prediction',
            'model_vjp': False,
        }
        config['execution_mode'] = 'rtc_action_fusion_diagnostic'
    if args.compare_action_blend_q8:
        config['execution_mode'] = 'q8_paired_action_blend_comparison'
        config['comparison'] = {
            'q': 8, 'pairs': pair_count,
            'conditions': ['crop_overwrite', 'paper_action_blend_beta_1.2'],
            'order': 'alternate AB and BA by pair',
            'fixed_task_seed_init': [0, 7, 0],
        }
    config['screen'] = {**config['screen'], 'task_ids': [0],
                        'trials_per_task': pair_count * 2 if args.compare_action_blend_q8 else 1}
    purpose = ('rtc-vjp-rollout' if args.rtc_vjp else
               'q8-blend-ab' if args.compare_action_blend_q8 else
               'rtc-action-fusion-6d' if args.rtc_action_fusion else
               'action-blend-arm6' if args.blend_arm_only else 'action-blend-pilot' if args.action_blend else 'adaptive-prefix-pilot'
               if args.adaptive_prefix else 'advance-handoff-rtc-merge' if args.prefix_rtc_merge else 'advance-handoff-pilot' if args.advance_handoff else 'prefix-lock-pilot' if args.prefix_lock else 'delay-buffer-pilot')
    run_dir = create_run(config, 'M1', purpose, repo_root=ROOT)
    print('run_dir=' + str(run_dir), flush=True)
    for name in ('scripts/probe_delay_buffer.py', 'scripts/eval_q.py', 'perceptbudget/model.py',
                 'perceptbudget/libero_input.py', 'perceptbudget/power.py', 'perceptbudget/prefix_model.py') + (
                     ('perceptbudget/prefix_fusion.py',) if args.prefix_rtc_merge else ()) + (
                     ('perceptbudget/prefix_timing.py',) if args.adaptive_prefix else ()) + (
                     ('perceptbudget/action_blend.py',)
                     if args.action_blend or args.compare_action_blend_q8 or args.prefix_rtc_merge else ()):
        source = ROOT / name
        target = run_dir / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    if args.rtc_action_fusion:
        rtc_source = Path(config['flashrt_root']) / 'flash_rt/runtime/rtc_temporal_fusion.py'
        rtc_target = run_dir / 'source/flash_rt/runtime/rtc_temporal_fusion.py'
        rtc_target.parent.mkdir(parents=True, exist_ok=True)
        rtc_target.write_bytes(rtc_source.read_bytes())
    if args.rtc_vjp:
        rtc_sources = [
            (ROOT, name) for name in ('perceptbudget/rtc_rollout.py',
                                      'perceptbudget/rtc_expert.py',
                                      'perceptbudget/rtc_guidance.py')
        ]
        rtc_sources.extend([
            (Path(config['flashrt_root']), 'flash_rt/subgraphs/pi05/rtc_action_vjp.py'),
            (Path(config['flashrt_root']), 'flash_rt/subgraphs/pi05/context_action.py'),
        ])
        for repo, name in rtc_sources:
            source = repo / name
            target = run_dir / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    summary = {'status': 'running',
               'purpose': ('single-task full RTC VJP rollout; diagnostic only' if args.rtc_vjp
                           else 'paired q=8 baseline/action-blend comparison' if args.compare_action_blend_q8
                           else 'single-task action-blend pilot; diagnostic only'
                           if args.action_blend
                           else 'single-task six-dimensional arm RTC temporal fusion; diagnostic only'
                           if args.rtc_action_fusion
                           else 'actual-delay episode; feasibility diagnostic only'),
               'task_ids': [0], 'q_values': q_values, 'episodes': [],
               'limitations': ['no RTC or fusion; expired-prefix crop/overwrite explicitly logged',
                              'queued buffer is not verified safe buffer',
                              'same q means requested trigger interval; skipped busy triggers reported',
                              'actual control speed measured; not assumed real 20Hz',
                              'no additional injected latency; instrumentation overhead included'],
               'state_protocol': base.STATE_PROTOCOL_NOTE,
               'norm_stats_sha256': hashlib.sha256(Path(config['norm_stats_path']).read_bytes()).hexdigest(),
               'device_before': device_state()}
    if args.action_blend or args.compare_action_blend_q8:
        summary['action_blend'] = dict(config['action_blend'])
        summary['limitations'][0] = ('paired comparison of crop/overwrite and paper overlap blending; '
                                     'no prefix locking, RTC or event estimator'
                                     if args.compare_action_blend_q8 else
                                     'paper overlap blending only; no prefix locking, RTC or event estimator')
    if args.rtc_action_fusion:
        summary['rtc_action_fusion'] = dict(config['rtc_action_fusion'])
        summary['limitations'][0] = ('host-side RTC temporal fusion on arm dimensions; gripper uses latest prediction; '
                                     'no denoiser VJP or model changes')
    if args.rtc_vjp:
        summary['limitations'][0] = ('single task/seed/init exploratory trial; '
                                     'not an estimated success rate; FlashRT itself is unchanged')
        summary['rtc_vjp'] = {
            'backend': ('project-side differentiable expert + FlashRT FA2 seqused split-KV CUDA graph'
                        if args.rtc_attention_backend == 'fa2-seqused' else
                        'project-side differentiable expert + math SDPA CUDA graph'),
            'flashrt_modified': False,
            'q': q_values[0], 'H': config['H'], 'N': config['N'], 'beta': 5.0,
            'attention_backend': args.rtc_attention_backend,
            'latency_prior_seconds': 0.5994,
            'delay_selection': 'ceil(previous full context+RTC latency / measured control period)',
            'handoff': 'crop elapsed new-action prefix at actual ready control step; keep original H deadline',
        }
    if args.compare_action_blend_q8:
        summary['comparison'] = dict(config['comparison'])
        summary['limitations'].append('single task and fixed seed/init; diagnostic pairs do not estimate LIBERO-wide success rate')
    env = None
    prefix_adapter = None
    rtc_adapter = None
    try:
        base._save_run_status(run_dir, 'running')
        summary['egl_cleanup'] = base._patch_egl_cleanup()
        torch, cuda = base._check_cuda_and_native(config)
        summary['cuda'] = cuda
        import numpy as np
        stats = load_libero_norm_stats(config['norm_stats_path'])
        task = base._load_tasks([0], config['screen']['suites'])[0]
        summary['task'] = {key: task[key] for key in
                           ('task_id', 'suite', 'suite_task_id', 'task_name',
                            'instruction', 'bddl_path', 'max_steps')}
        summary['seed'] = 7
        summary['init_state_index'] = 0
        _, env_class = base._libero_api()
        model = base._load_and_check_model(config, torch)
        if args.rtc_vjp:
            from flash_rt.subgraphs.pi05.context_action import enable
            enable(model)
        if args.prefix_lock:
            from perceptbudget.prefix_model import PrefixModel
            prefix_adapter = PrefixModel(model, prefix_len=args.prefix_len)
            summary['limitations'][0] = 'prefix locking, not full RTC; no VJP guidance or blending'
        env = base._make_env(env_class, task)
        obs, _ = base._prepare_env_episode(env, task, np)
        warm_snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
        if prefix_adapter is None:
            base._infer_actions(model, warm_snapshot, task['instruction'], config['H'], {})
        else:
            gate_timing = {}
            _, raw = prefix_adapter.predict(warm_snapshot, task['instruction'], {})
            if args.adaptive_prefix:
                prefix_adapter.prepare_variants(config['adaptive_prefix']['lengths'])
                gates = []
                for length in config['adaptive_prefix']['lengths']:
                    timing = {}
                    output, raw_output = prefix_adapter.predict(warm_snapshot, task['instruction'], timing,
                                                                raw[8:8 + length], prefix_len=length)
                    gates.append({'prefix_len': length, 'prefix_exact': timing['prefix_exact'],
                                  'finite_raw': bool(np.isfinite(raw_output).all()),
                                  'inference_seconds': timing['inference_ended_monotonic'] - timing['inference_started_monotonic']})
                summary['variant_gates'] = gates
                from statistics import median
                config['adaptive_prefix']['initial_latency_s'] = median(g['inference_seconds'] for g in gates)
                summary['adaptive_prefix'] = dict(config['adaptive_prefix'])
            gate_actions, gate_raw = prefix_adapter.predict(warm_snapshot, task['instruction'],
                                                           gate_timing, raw[8:8 + args.prefix_len])
            summary['interface_gate'] = {'prefix_exact': gate_timing['prefix_exact'],
                                         'finite_raw': bool(np.isfinite(gate_raw).all()),
                                         'finite_actions': bool(np.isfinite(gate_actions).all()),
                                         'shape': list(gate_raw.shape),
                                         'inference_seconds': gate_timing['inference_ended_monotonic'] - gate_timing['inference_started_monotonic']}
            base._write_json_atomic(run_dir / 'summary.json', summary)
            print('prefix_interface_gate=' + json.dumps(summary['interface_gate']), flush=True)
        if args.interface_only:
            summary['status'] = 'interface_only_completed'
            base._save_run_status(run_dir, 'interface_only_completed', completed_episodes=0)
            return
        if args.rtc_vjp:
            from perceptbudget.rtc_rollout import RTCGuidanceAdapter
            rtc_adapter = RTCGuidanceAdapter(
                model, torch, horizon=config['H'], num_steps=config['N'],
                latency_prior_seconds=summary['rtc_vjp']['latency_prior_seconds'],
                attention_backend=args.rtc_attention_backend)
            summary['rtc_vjp']['graph_capture_seconds'] = rtc_adapter.capture_seconds
            base._write_json_atomic(run_dir / 'summary.json', summary)
        with (run_dir / 'trace.jsonl').open('w', encoding='utf-8') as stream:
            if args.compare_action_blend_q8:
                schedule = []
                for pair_index in range(pair_count):
                    pair = [('crop_overwrite', False), ('paper_action_blend', True)]
                    if pair_index % 2:
                        pair.reverse()
                    schedule.extend((pair_index + 1, condition, enabled)
                                    for condition, enabled in pair)
                for trial_index, (pair_index, condition, enabled) in enumerate(schedule, 1):
                    episode = run_episode(env, task, 8, model, stats, config, np, torch,
                                          run_dir, stream, prefix_adapter,
                                          trial_index=trial_index, condition=condition,
                                          action_blend_enabled=enabled,
                                          artifact_tag=f'_trial{trial_index:02d}')
                    episode['pair_index'] = pair_index
                    summary['episodes'].append(episode)
                    base._write_json_atomic(run_dir / 'summary.json', summary)
            else:
                condition = ('full_rtc_vjp' if args.rtc_vjp else
                             'advance_prefix_handoff_rtc_merge' if args.prefix_rtc_merge else
                             'advance_prefix_handoff' if args.advance_handoff else
                             'rtc_temporal_fusion_6d' if args.rtc_action_fusion else
                             'paper_action_blend' if args.action_blend else 'crop_overwrite')
                for trial_index, q in enumerate(q_values, 1):
                    summary['episodes'].append(run_episode(
                        env, task, q, model, stats, config, np, torch, run_dir, stream,
                        prefix_adapter, trial_index=trial_index, condition=condition,
                        action_blend_enabled=args.action_blend,
                        rtc_fusion_enabled=args.rtc_action_fusion,
                        rtc_adapter=rtc_adapter))
                    base._write_json_atomic(run_dir / 'summary.json', summary)
        summary['status'] = 'completed'
        summary['device_after'] = device_state()
        base._save_run_status(run_dir, 'completed', completed_episodes=len(summary['episodes']))
    except BaseException as exc:
        summary.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        base._save_run_status(run_dir, 'invalid', error=summary['error'])
        raise
    finally:
        base._write_json_atomic(run_dir / 'summary.json', summary)
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
