#!/usr/bin/env python3
"""Read-only, stdlib-only audit of ordinary async q8 seven-dimensional Eq8/9.

Usage: python3 scripts/verify_action_blend.py RUN_DIR
JSON goes to stdout; exits 0=verified, 1=violation, 2=incomplete evidence.
No runner/model imports, writes, or GPU access. N=1 explicitly uses new action.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def audit(root):
    errors = []
    missing = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    def equal(actual, expected, label, tolerance=1e-6):
        if isinstance(expected, (list, tuple)):
            check(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + ': shape')
            if isinstance(actual, (list, tuple)) and len(actual) == len(expected):
                for i, (a, b) in enumerate(zip(actual, expected)):
                    equal(a, b, f'{label}[{i}]', tolerance)
        elif isinstance(expected, dict):
            check(isinstance(actual, dict), label + ': object')
            if isinstance(actual, dict):
                for key, value in expected.items():
                    equal(actual.get(key), value, label + '.' + key, tolerance)
        elif isinstance(expected, float):
            check(isinstance(actual, (int, float)) and math.isfinite(actual)
                  and abs(actual - expected) <= tolerance, label + ': value')
        else:
            check(actual == expected, label + ': value')

    def read(name):
        path = root / name
        if not path.is_file():
            missing.append(name)
            return None
        return json.loads(path.read_text())

    run = read('run.json')
    summary = read('summary.json')
    requests = read('q8_requests.json')
    trace_path = root / 'trace.jsonl'
    if not trace_path.is_file():
        missing.append('trace.jsonl')
    if missing:
        return {'status': 'incomplete', 'missing': missing}, 2
    config = run['config_snapshot']
    blend_config = config['action_blend']
    equal(blend_config['action_dimensions'], list(range(7)), 'config.dimensions')
    equal(blend_config['beta'], 1.2, 'config.beta')
    equal(blend_config['event_estimator'], False, 'config.event_estimator')
    equal(blend_config['prefix_lock'], False, 'config.prefix_lock')
    check(not config.get('prefix_lock'), 'config enables prefix_lock')
    check(not config.get('rtc_action_fusion'), 'config enables rtc_action_fusion')
    episodes = [e for e in summary['episodes'] if e['q'] == 8]
    if not episodes:
        return {'status': 'incomplete', 'run_status': run['status'],
                'summary_status': summary['status'],
                'reason': 'no completed q8 episode; no task execution claimed'}, 2
    check(len(episodes) == 1, 'requires exactly one q8 episode')
    if len(episodes) != 1:
        return {'status': 'failed', 'errors': errors}, 1
    episode = episodes[0]
    for key, value in {'prefix_lock': False, 'advance_handoff': False,
                       'pending_arm_gain': 1.0, 'action_blend': True,
                       'rtc_action_fusion': False, 'alignment_policy': None,
                       'synchronous_recording': False}.items():
        equal(episode.get(key), value, 'episode.' + key)
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    events = [e for e in events if e.get('q') == 8]
    starts = [e for e in events if e['event'] == 'episode_start']
    check(len(starts) == 1, 'requires one q8 trace episode')
    commits = [e for e in events if e['event'] == 'prediction_committed']
    controls = [e for e in events if e['event'] == 'control_step']
    busy = [e['step'] for e in events if e['event'] == 'busy_trigger']
    equal([e['step'] for e in controls], list(range(len(controls))), 'trace.control order')
    check(all(a['commit_monotonic'] <= b['commit_monotonic']
              for a, b in zip(commits, commits[1:])), 'trace.commit order')
    ids = [r['prediction_id'] for r in requests]
    check(len(ids) == len(set(ids)), 'duplicate request id')
    by_id = {r['prediction_id']: r for r in requests}
    commit_by_id = {e['prediction_id']: e for e in commits}
    for a, b in zip(requests, requests[1:]):
        previous = commit_by_id.get(a['prediction_id'])
        check(previous is not None and previous['commit_monotonic'] <= b['observation_monotonic'],
              'concurrent or unharvested requests')
    expected_busy = []
    for row in controls:
        step, timestamp = row['step'], row['action_start_monotonic']
        active = [r for r in requests if r['observation_monotonic'] <= timestamp
                  and (r['prediction_id'] not in commit_by_id or
                       commit_by_id[r['prediction_id']]['commit_monotonic'] > timestamp)]
        check(len(active) <= 1, f'step {step}: concurrent requests')
        equal(row['inference_active_at_action'], bool(active), f'step {step}.inference active')
        if step and step % 8 == 0:
            submitted = [r for r in requests if r['observation_step'] == step]
            if not submitted:
                expected_busy.append(step)
                check(bool(active), f'step {step}: skipped boundary without busy request')
            else:
                check(len(submitted) == 1, f'step {step}: duplicate submission')
    equal(busy, expected_busy, 'independent busy replay')
    horizon = config['H']
    for r in requests:
        actions = r.get('actions')
        if actions is None:
            missing.append(f'request {r["prediction_id"]}: actions')
        else:
            check(len(actions) == horizon and all(len(a) == 7 and all(
                isinstance(v, (float, int)) and math.isfinite(v) for v in a)
                for a in actions), f'request {r["prediction_id"]}: Hx7 finite actions')
    # Row logging happens AFTER the post-action harvest. Reorder by actual
    # control-thread times, never by JSONL line order or commit_step alone.
    timeline = [(e['commit_monotonic'], 0, e) for e in commits]
    timeline += [(e['action_start_monotonic'], 1, e) for e in controls]
    timeline.sort(key=lambda item: (item[0], item[1]))
    queue = {}
    committed = set()
    underflow = []
    overlap_counts = Counter()
    executed_blends = 0
    next_step = 0

    def count(begin):
        end = begin
        while end in queue:
            end += 1
        return end - begin

    def source(item):
        return {'prediction_id': item['id'], 'origin_step': item['origin'],
                'offset': item['offset'], 'already_blended': item['blend'] is not None}

    def alive(item, step):
        check(item['origin'] + item['offset'] == step, f'step {step}: source alignment')
        check(step < item['origin'] + len(by_id[item['id']]['actions']),
              f'step {step}: expired source')
        for ancestor in item['ancestors']:
            check(step < by_id[ancestor]['observation_step'] + len(by_id[ancestor]['actions']),
                  f'step {step}: expired blend ancestor {ancestor}')

    for _, kind, event in timeline:
        if kind == 0:
            pid = event['prediction_id']
            r = by_id[pid]
            equal(event['request'], r, f'commit {pid}.request')
            check(pid not in committed, f'duplicate commit {pid}')
            committed.add(pid)
            step, origin = event['commit_step'], r['observation_step']
            equal(step, next_step, f'commit {pid}.control time')
            equal(event['observation_step'], origin, f'commit {pid}.origin')
            check(r['ready_monotonic'] <= event['commit_monotonic'], f'commit {pid}: before ready')
            actions = r['actions']
            begin = max(step, origin)
            n = min(count(begin), max(0, origin + len(actions) - begin))
            overlap_counts[n] += 1
            old_items = [queue[begin + i] for i in range(n)]
            for i, item in enumerate(old_items):
                alive(item, begin + i)
            old = [item['action'] for item in old_items]
            new = [actions[begin - origin + i] for i in range(n)]
            beta = 1.2
            weights = [(math.exp(-beta * i / (n - 1)) - math.exp(-beta)) /
                       (1 - math.exp(-beta)) if n > 1 else 0.0 for i in range(n)]
            merged = [[w * a + (1 - w) * b for a, b in zip(x, y)]
                      for w, x, y in zip(weights, old, new)]
            expected = {'overlap_actions': n, 'global_interval_steps': [begin, begin + n],
                        'new_offset_interval': [begin - origin, begin - origin + n],
                        'beta': beta, 'weights_old': weights, 'old_actions': old,
                        'new_actions': new, 'merged_actions': merged,
                        'old_sources': [source(item) for item in old_items],
                        'blended_dimensions': 7, 'gripper_policy': 'same_blend_as_arm'}
            equal(event.get('blend'), expected, f'commit {pid}.blend', 1e-10)
            equal(event['old_queue_remaining_actions'], count(step), f'commit {pid}.old queue')
            overwritten = sum(origin + i in queue for i in range(len(actions)) if origin + i >= step)
            expired = min(len(actions), max(0, step - origin))
            equal(event['expired_new_actions'], expired, f'commit {pid}.expired')
            equal(event['overwritten_old_actions'], overwritten, f'commit {pid}.overwritten')
            # Replace the old tail, with no slots beyond the new origin+H.
            queue = {target: item for target, item in queue.items() if target < step}
            for offset, action in enumerate(actions):
                target = origin + offset
                if target >= step:
                    queue[target] = {'id': pid, 'origin': origin, 'offset': offset,
                                     'action': action, 'blend': None, 'ancestors': {pid}}
            for i, item in enumerate(old_items):
                target = begin + i
                queue[target].update(action=merged[i], ancestors=item['ancestors'] | {pid},
                                     blend={'old_source': source(item), 'new_prediction_id': pid,
                                            'weight_old': weights[i], 'overlap_index': i,
                                            'overlap_length': n})
            equal(event['new_queue_actions'], count(step), f'commit {pid}.new queue')
        else:
            step = event['step']
            equal(step, next_step, 'control sequence')
            next_step = step + 1
            equal(event['queue_before_pop_actions'], count(step), f'step {step}.queue')
            item = queue.pop(step, None)
            equal(event.get('late_hold', False), False, f'step {step}.hold')
            equal(event['applied_arm_gain'], 1.0, f'step {step}.gain')
            if item is None:
                underflow.append(step)
                equal(event['source'], None, f'step {step}.underflow source')
                action = [0.0] * 7
            else:
                alive(item, step)
                s = {'prediction_id': item['id'], 'origin_step': item['origin'],
                     'offset': step - item['origin'], 'beyond_sync_q_prefix': item['offset'] >= 8}
                if item['blend'] is not None:
                    s['blend'] = item['blend']
                    executed_blends += 1
                equal(event['source'], s, f'step {step}.source')
                check(('blend' in event['source']) == (item['blend'] is not None),
                      f'step {step}: unexpected blend metadata')
                action = [f32(v) for v in item['action']]
            equal(event['planned_action'], action, f'step {step}.planned', 1e-7)
            equal(event['action'], action, f'step {step}.executed', 1e-7)
    intervals = [b['observation_step'] - a['observation_step'] for a, b in zip(requests, requests[1:])]
    equal(episode['actual_refresh_intervals_steps'], intervals, 'summary.intervals')
    equal(episode['busy_trigger_steps'], busy, 'summary.busy')
    equal(episode['underflow_steps'], underflow, 'summary.underflow')
    equal(episode['steps'], len(controls), 'summary.steps')
    check(bool(controls) and bool(commits), 'no executed controls/commits')
    equal(episode['commits'], [{k: v for k, v in e.items() if k in c}
                             for e, c in zip(commits, episode['commits'])], 'summary.commits')
    check(len(episode['commits']) == len(commits), 'summary.commit count')
    # Busy means a due boundary with an unharvested request, not merely
    # inference_ended being later than the boundary.
    for step in busy:
        check(step > 0 and step % 8 == 0, f'busy {step}: not q8 boundary')
    for r in requests:
        check(r['observation_step'] % 8 == 0, f'request {r["prediction_id"]}: not q8 boundary')
    uncommitted = []
    terminal = episode.get('stop_monotonic')
    trace_uncommitted = {e['request']['prediction_id']: e['request'] for e in events
                         if e['event'] == 'uncommitted_prediction'}
    for r in requests:
        if r['prediction_id'] not in committed:
            ready = r.get('ready_monotonic')
            state = ('not_recorded_ready' if ready is None else
                     'ready_after_episode_stop' if terminal is not None and ready > terminal else
                     'ready_before_stop_but_not_harvested')
            uncommitted.append({'prediction_id': r['prediction_id'], 'state': state,
                                'trace_recorded': r['prediction_id'] in trace_uncommitted})
            if r['prediction_id'] in trace_uncommitted:
                equal(trace_uncommitted[r['prediction_id']], r, 'uncommitted.request')
    complete = summary['status'] == 'completed' and run['status'] == 'completed'
    if not complete:
        missing.append('completed run/summary status')
    result = {'status': 'failed' if errors else 'incomplete' if missing else 'verified',
              'scope': 'ordinary_async_paper_eq8_9_7d; no event model; not original RTC',
              'steps': len(controls), 'commits': len(commits), 'executed_blend_steps': executed_blends,
              'overlap_lengths': dict(sorted(overlap_counts.items())),
              'requests': len(requests), 'actual_request_intervals_steps': dict(sorted(Counter(intervals).items())),
              'busy_count': len(busy), 'busy_steps': busy,
              'underflow_count': len(underflow), 'underflow_steps': underflow,
              'uncommitted_requests': uncommitted, 'error_count': len(errors), 'errors': errors[:30]}
    if missing:
        result['missing'] = missing
    return result, 1 if errors else 2 if missing else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    try:
        result, code = audit(args.run_dir)
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        result, code = {'status': 'incomplete', 'error': f'{type(exc).__name__}: {exc}'}, 2
    print(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
