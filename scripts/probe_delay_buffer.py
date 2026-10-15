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


def commit_blended(queue, actions, origin, step, prediction_id, beta=1.2,
                   blend_dims=7):
    """Same absolute-step alignment as baseline; change only overlap update."""
    from perceptbudget.action_blend import blend_actions
    if blend_dims not in (6, 7):
        raise ValueError('blend_dims must be 6 or 7 for LIBERO actions')
    if blend_dims == 6 and any(len(action) != 7 for action in actions):
        raise ValueError('six-dimensional arm blend requires 7D LIBERO actions')
    begin = max(step, origin)
    count = min(buffer_count(queue, begin), max(0, origin + len(actions) - begin))
    old = [[float(v) for v in queue[begin + i][0]] for i in range(count)]
    new = [[float(v) for v in actions[begin - origin + i]] for i in range(count)]
    sources = [{'prediction_id': queue[begin + i][1], 'origin_step': queue[begin + i][2],
                'offset': queue[begin + i][3], 'already_blended': len(queue[begin + i]) > 4}
               for i in range(count)]
    merged, weights = blend_actions(old, new, beta=beta)
    if blend_dims == 6:
        for index, action in enumerate(merged):
            action[6] = new[index][6]
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
        'blended_dimensions': blend_dims,
        'gripper_policy': 'new_prediction' if blend_dims == 6 else 'same_blend_as_arm',
        'single_overlap_policy': 'new action directly; paper only defines N>=2',
    }


def commit_rtc_fused(queue, raw_chunks, actions, origin, step, prediction_id,
                     decay=0.1, max_chunks=3):
    """Apply FlashRT's host-side temporal-fusion equations to arm dims only."""
    import math
    if not math.isfinite(decay) or decay < 0:
        raise ValueError('RTC decay must be finite and non-negative')
    if type(max_chunks) is not int or max_chunks < 1:
        raise ValueError('RTC max_chunks must be a positive integer')
    if any(len(action) != 7 for action in actions):
        raise ValueError('RTC six-dimensional fusion requires 7D LIBERO actions')

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
        arm = [0.0] * 6
        weights = []
        for chunk in candidates:
            source_index = target - chunk['origin_step']
            weight = math.exp(-decay * abs(source_index - latest_index))
            weights.append(weight)
            total_weight += weight
            for dim in range(6):
                arm[dim] += weight * chunk['actions'][source_index][dim]
        if not candidates or total_weight <= 0:
            raise RuntimeError('RTC fusion found no valid action source')
        if len(candidates) > 1:
            fused_steps += 1
        overlap_counts[len(candidates)] = overlap_counts.get(len(candidates), 0) + 1
        action = list(latest['actions'][latest_index])
        for dim in range(6):
            action[dim] = arm[dim] / total_weight
        # Gripper dim 6 stays exactly equal to the latest model prediction.
        item = queue[target]
        metadata = {'prediction_id': prediction_id, 'source_prediction_ids':
                    [chunk['prediction_id'] for chunk in candidates],
                    'source_count': len(candidates), 'source_weights': weights,
                    'arm_dimensions': 6, 'gripper_policy': 'latest_prediction'}
        queue[target] = (action, item[1], item[2], item[3], metadata)
    raw_chunks[:] = chunks[-max_chunks:]
    return expired, overwritten, {
        'fused_steps': fused_steps, 'source_count_histogram': overlap_counts,
        'decay': decay, 'max_chunks': max_chunks, 'arm_dimensions': 6,
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


def run_episode(env, task, q, model, stats, config, np, torch, run_dir, stream,
                prefix_adapter=None, *, trial_index=1, condition='crop_overwrite',
                action_blend_enabled=False, rtc_fusion_enabled=False,
                artifact_tag=''):
    observation, hz = base._prepare_env_episode(env, task, np)
    assert hz == 20
    base._seed_process(7, np, torch)
    period = 1 / hz
    queue, requests, commits, rows, frames, raw_chunks = {}, [], [], [], [], []
    future = None
    current_request = None
    missed_triggers, underflows = [], []
    selector = None
    prefix_rejection = None
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
        snapshot = base._make_snapshot(obs, stats, config['state_enabled'])
        request = {'prediction_id': len(requests), 'observation_step': step,
                   'observation_monotonic': time.monotonic(),
                   'model_state': snapshot.get('state'),
                   'queue_before_submit_actions': buffer_count(queue, step)}
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
                raise RuntimeError('insufficient queued actions for committed prefix')
            raw_prefix = np.asarray([queue[step + i][4] for i in range(prefix_len)])
            request.update(committed_prefix_raw=raw_prefix.tolist(),
                           committed_prefix_actions=[np.asarray(queue[step + i][0]).tolist() for i in range(prefix_len)],
                           committed_interval_steps=[step, step + prefix_len],
                           prefix_len=prefix_len,
                           scheduled_commit_step=step + prefix_len)
        requests.append(request)

        def predict():
            if prefix_adapter is None:
                actions = base._infer_actions(model, snapshot, task['instruction'],
                                              config['H'], request)
            else:
                actions, raw = prefix_adapter.predict(snapshot, task['instruction'], request, raw_prefix,
                                                     prefix_len=prefix_len)
                request['actions_raw'] = raw.tolist()
            request['ready_monotonic'] = time.monotonic()
            request['actions'] = actions.tolist()
            return actions

        return executor.submit(predict), request

    def harvest(step):
        nonlocal future, current_request
        if future is None or not future.done():
            return
        ready_actions = future.result()
        if selector is not None and not current_request.get('latency_observed'):
            selector.observe_latency(current_request['ready_monotonic'] - current_request['observation_monotonic'])
            current_request['latency_observed'] = True
        current_request.setdefault('ready_observed_step', step)
        scheduled = current_request.get('scheduled_commit_step', 0)
        if step < scheduled:
            return  # Never replace a prefix already committed for execution.
        if prefix_adapter is not None and scheduled and step > scheduled:
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
        merge_start = time.monotonic()
        if action_blend_enabled:
            expired, overwritten, blend = commit_blended(
                queue, actions, origin, step, current_request['prediction_id'],
                beta=config['action_blend']['beta'],
                blend_dims=config['action_blend']['blend_dims'])
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
                  'new_queue_actions': buffer_count(queue, step)}
        if blend is not None:
            result['blend'] = blend
            result['merge_seconds'] = time.monotonic() - merge_start
        if rtc_fusion is not None:
            result['rtc_action_fusion'] = rtc_fusion
            result['merge_seconds'] = time.monotonic() - merge_start
        commits.append(result)
        base._trace(stream, 'prediction_committed', task_id=task['task_id'], q=q,
                    trial_index=trial_index, condition=condition,
                    request=current_request, **result)
        future, current_request = None, None

    stop = None
    status = 'invalid'
    try:
        future, current_request = submit(observation, 0)
        future.result()  # Only bootstrap blocks; no control ticks yet.
        harvest(0)
        deadline = time.monotonic()
        for step in range(task['max_steps']):
            tick = time.monotonic()
            if selector is not None and rows:
                selector.observe_tick(tick - rows[-1]['tick_monotonic'])
            before = snapshot_scene(env, observation)
            frames.append(np.concatenate((observation['agentview_image'],
                                         observation['robot0_eye_in_hand_image']), axis=1))
            harvest(step)
            if step and step % q == 0:
                if future is None:
                    future, current_request = submit(observation, step)
                    if future is None:
                        prefix_rejection = current_request
                        frames.pop()  # No action was executed for this tick.
                        status = 'infeasible_prefix_plan'
                        break
                else:
                    missed_triggers.append(step)
                    base._trace(stream, 'busy_trigger', task_id=task['task_id'], q=q,
                                trial_index=trial_index, condition=condition, step=step)
            queued = buffer_count(queue, step)
            item = queue.pop(step, None)
            if item is None:
                underflows.append(step)
                # Explicit zero-translation, zero-rotation, neutral gripper command.
                action = np.zeros(7, dtype=np.float32)
                source = None
            else:
                action = np.asarray(item[0], dtype=np.float32)
                source = {'prediction_id': item[1], 'origin_step': item[2],
                          'offset': item[3], 'beyond_sync_q_prefix': item[3] >= q}
                if action_blend_enabled and len(item) > 4:
                    source['blend'] = item[4]
                elif rtc_fusion_enabled and len(item) > 4:
                    source['rtc_action_fusion'] = item[4]
            action_start = time.monotonic()
            observation, reward, done, info = env.step(action)
            action_end = time.monotonic()
            after = snapshot_scene(env, observation)
            success = bool(env.check_success())
            # Harvest only on control thread, also after a step, never block it.
            harvest(step + 1)
            terminal = success or done or step + 1 == task['max_steps']
            deadline += period
            overrun = max(0.0, time.monotonic() - deadline)
            row = {'step': step, 'trial_index': trial_index, 'condition': condition,
                   'tick_monotonic': tick,
                   'action_start_monotonic': action_start, 'action_end_monotonic': action_end,
                   'before': before, 'after': after, 'action': action.tolist(),
                   'source': source, 'queue_before_pop_actions': queued,
                   'queue_after_step_actions': buffer_count(queue, step + 1),
                   'inference_active': future is not None,
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
        monitor.stop()
        executor.shutdown(wait=True)
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
              'prefix_plan_rejection': prefix_rejection,
              'late_discarded_predictions': sum(bool(r.get('late_discarded')) for r in requests),
              'underflow_steps': underflows, 'commits': commits,
              'tail_execution_steps': sum(bool(r['source'] and r['source']['beyond_sync_q_prefix']) for r in rows),
              'post_bootstrap_control_hz': (len(rows) - 1) / (rows[-1]['action_end_monotonic'] - rows[0]['action_end_monotonic']) if len(rows) > 1 else None,
              'control_deadline_overruns': sum(r['deadline_overrun_seconds'] > 0 for r in rows),
              'power': base._power_result(monitor, start, stop),
              'field_checks': {'full_predictions': all(len(r.get('actions', [])) == config['H'] for r in requests),
                               'pre_post_state': all('raw_state' in r['before'] and 'raw_state' in r['after'] for r in rows),
                               'scene_contacts': all('contacts' in r['after'] for r in rows),
                               'frames': len(frames) == len(rows) + 1}}
    print(json.dumps({k: result[k] for k in ('q', 'status', 'steps', 'real_seconds', 'post_bootstrap_control_hz', 'tail_execution_steps', 'field_checks')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix-lock', action='store_true')
    parser.add_argument('--adaptive-prefix', action='store_true')
    parser.add_argument('--action-blend', action='store_true')
    parser.add_argument('--rtc-action-fusion', action='store_true',
                        help='use controller-side RTC temporal fusion on arm dims only')
    parser.add_argument('--blend-dims', type=int, choices=(6, 7), default=7,
                        help='blend 6 arm dimensions and use new gripper (6), or blend all 7')
    parser.add_argument('--q', type=int, nargs='+', choices=(5, 8, 10, 12, 15),
                        help='control-step refresh interval(s); default: 8 12')
    parser.add_argument('--compare-action-blend-q8', action='store_true',
                        help='run paired q=8 baseline/blending episodes in one loaded model process')
    parser.add_argument('--pairs', type=int, default=None,
                        help='number of q=8 baseline/blending pairs; default: 3')
    parser.add_argument('--interface-only', action='store_true')
    args = parser.parse_args()
    if args.adaptive_prefix:
        args.prefix_lock = True
    if args.action_blend and args.prefix_lock:
        parser.error('action blending is tested separately from prefix locking')
    if args.rtc_action_fusion and (args.action_blend or args.prefix_lock or args.compare_action_blend_q8):
        parser.error('RTC action fusion is a separate controller-side condition')
    if args.blend_dims != 7 and not args.action_blend:
        parser.error('--blend-dims requires --action-blend')
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
    q_values = [8] if args.compare_action_blend_q8 else args.q or [8, 12]
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
    config = load_config(repo_root=ROOT)
    config.update(candidate_q=q_values,
                  execution_mode=('q8_paired_action_blend_comparison' if args.compare_action_blend_q8
                                  else 'prefix_lock_diagnostic' if args.prefix_lock
                                  else 'actual_delay_diagnostic'))
    if args.prefix_lock:
        config['prefix_lock'] = {'length': 8, 'full_rtc': False,
                                 'early_result': 'wait until committed prefix ends',
                                 'late_result': 'discard and record; never silently claim prefix agreement'}
    if args.adaptive_prefix:
        config['adaptive_prefix'] = {'lengths': [5, 6, 7, 8], 'margin_s': 0.005,
                                     'latency_estimator': 'recent16 nearest-rank p80',
                                     'tick_estimator': 'recent16 nearest-rank p20; at least nominal50ms',
                                     'no_feasible_choice': 'stop episode explicitly; no silent clamp'}
        config['execution_mode'] = 'adaptive_prefix_diagnostic'
    if args.action_blend or args.compare_action_blend_q8:
        config['action_blend'] = {
            'beta': 1.2, 'source': 'https://arxiv.org/html/2609.22587v1#S4.SS3',
            'equations': [8, 9], 'blend_dims': args.blend_dims,
            'blended_dimensions': args.blend_dims,
            'gripper_policy': 'new_prediction' if args.blend_dims == 6 else 'same_blend_as_arm',
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
            'arm_dimensions': [0, 1, 2, 3, 4, 5],
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
    purpose = ('q8-blend-ab' if args.compare_action_blend_q8 else
               'rtc-action-fusion-6d' if args.rtc_action_fusion else
               'action-blend-6d-pilot' if args.action_blend and args.blend_dims == 6 else
               'action-blend-pilot' if args.action_blend else 'adaptive-prefix-pilot'
               if args.adaptive_prefix else 'prefix-lock-pilot' if args.prefix_lock else 'delay-buffer-pilot')
    run_dir = create_run(config, 'M1', purpose, repo_root=ROOT)
    print('run_dir=' + str(run_dir), flush=True)
    for name in ('scripts/probe_delay_buffer.py', 'scripts/eval_q.py', 'perceptbudget/model.py',
                 'perceptbudget/libero_input.py', 'perceptbudget/power.py', 'perceptbudget/prefix_model.py') + (
                     ('perceptbudget/prefix_timing.py',) if args.adaptive_prefix else ()) + (
                     ('perceptbudget/action_blend.py',)
                     if args.action_blend or args.compare_action_blend_q8 else ()):
        source = ROOT / name
        target = run_dir / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    if args.rtc_action_fusion:
        rtc_source = Path(config['flashrt_root']) / 'flash_rt/runtime/rtc_temporal_fusion.py'
        rtc_target = run_dir / 'source/flash_rt/runtime/rtc_temporal_fusion.py'
        rtc_target.parent.mkdir(parents=True, exist_ok=True)
        rtc_target.write_bytes(rtc_source.read_bytes())
    summary = {'status': 'running',
               'purpose': ('paired q=8 baseline/action-blend comparison' if args.compare_action_blend_q8
                           else 'single-task six-dimensional arm blend; diagnostic only'
                           if args.action_blend and args.blend_dims == 6
                           else 'single-task action-blend pilot; diagnostic only'
                           if args.action_blend
                           else 'single-task six-dimensional RTC temporal fusion; diagnostic only'
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
        summary['limitations'][0] = ('host-side RTC temporal fusion on arm dimensions only; '
                                     'no denoiser VJP or model changes')
    if args.compare_action_blend_q8:
        summary['comparison'] = dict(config['comparison'])
        summary['limitations'].append('single task and fixed seed/init; diagnostic pairs do not estimate LIBERO-wide success rate')
    env = None
    prefix_adapter = None
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
        if args.prefix_lock:
            from perceptbudget.prefix_model import PrefixModel
            prefix_adapter = PrefixModel(model, prefix_len=8)
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
                                                           gate_timing, raw[8:16])
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
                condition = ('rtc_temporal_fusion_6d' if args.rtc_action_fusion else
                             'paper_action_blend' if args.action_blend else 'crop_overwrite')
                for trial_index, q in enumerate(q_values, 1):
                    summary['episodes'].append(run_episode(
                        env, task, q, model, stats, config, np, torch, run_dir, stream,
                        prefix_adapter, trial_index=trial_index, condition=condition,
                        action_blend_enabled=args.action_blend,
                        rtc_fusion_enabled=args.rtc_action_fusion))
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
