"""CPU-only, incremental analysis of paired asynchronous q experiments.

Reads raw records without changing them; writes only analysis.json/analysis.md.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

QS = (5, 8, 10, 12, 15)
SEED = 7
BOOTSTRAPS = 4000
H = 20
DT = 0.05
EVENT_RULES = {
    'gripper_switch': 'nonzero command sign changes; zeros retain previous sign',
    'deceleration': 'adjacent command norms: previous>=0.1, current<=0.5*previous, drop>=0.05',
    'turn': 'adjacent translation command vectors: both norms>=0.05, angle>=60 degrees',
    'matching': 'same type, one-to-one nearest pair within +/-2 control steps; boundaries right-censored',
}


def describe(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {'n': 0, 'mean': None, 'min': None, 'median': None, 'p95': None, 'max': None}
    return dict(n=len(values), mean=float(np.mean(values)), min=min(values),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)), max=max(values))


def bootstrap_mean(values):
    """Resample tasks, never individual timesteps (which are correlated)."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0, 'mean': None, 'ci95': None}
    if len(values) == 1:
        return {'n': 1, 'mean': float(values[0]), 'ci95': None,
                'reason': 'insufficient tasks: fewer than 2',
                'constant_sample': True}
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
    return {'n': len(values), 'mean': float(values.mean()),
            'ci95': np.percentile(means, [2.5, 97.5]).tolist(),
            'method': 'paired-task percentile bootstrap, seed=7, 4000 draws',
            'constant_sample': bool(np.all(values == values[0]))}


def clopper_pearson(successes, total, alpha=0.05):
    if not total:
        return None
    return [0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total-successes+1)),
            1.0 if successes == total else float(beta.ppf(1-alpha / 2, successes+1, total-successes))]


def paired_success(baseline, candidate):
    n = len(baseline)
    if n != len(candidate):
        raise ValueError('paired outcomes must have equal length')
    if not n:
        return {'n': 0, 'difference': None, 'ci95': None}
    gain = sum(not a and b for a, b in zip(baseline, candidate))
    loss = sum(a and not b for a, b in zip(baseline, candidate))
    # Two 97.5% marginal intervals => >=95% simultaneous coverage (Bonferroni).
    g = clopper_pearson(gain, n, alpha=0.025)
    l = clopper_pearson(loss, n, alpha=0.025)
    return {'n': n, 'gain': gain, 'loss': loss, 'difference': (gain-loss)/n,
            'ci95': [g[0]-l[1], g[1]-l[0]],
            'method': 'paired gain/loss: simultaneous 97.5% Clopper-Pearson bounds; Bonferroni >=95%'}


def command_events(rows):
    events = []
    previous = None
    sign = None
    for row in sorted(rows, key=lambda r: r['step']):
        step, action = row['step'], np.asarray(row['action'], dtype=float)
        current_sign = np.sign(action[6]) or sign
        if previous is not None and step != previous[0]+1:
            previous, sign = None, None
            current_sign = np.sign(action[6]) or None
        if sign is not None and current_sign is not None and current_sign != sign:
            events.append({'type': 'gripper_switch', 'step': step})
        if previous is not None:
            a, b = previous[1][:3], action[:3]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na >= 0.1 and nb <= na*0.5 and na-nb >= 0.05:
                events.append({'type': 'deceleration', 'step': step})
            if min(na, nb) >= 0.05:
                angle = math.degrees(math.acos(float(np.clip(np.dot(a, b)/(na*nb), -1, 1))))
                if angle >= 60-1e-10:
                    events.append({'type': 'turn', 'step': step})
        previous, sign = (step, action), current_sign
    return events


def prediction_events(rows, requests, commits):
    """Compare committed forecasts with later executed COMMAND events, not contact."""
    actual = command_events(rows)
    observed_end = max((r['step'] for r in rows), default=-1)+1
    by_id = {r['prediction_id']: r for r in requests}
    output = []
    for commit in commits:
        request = by_id.get(commit['prediction_id'])
        if not request or not request.get('actions'):
            continue
        origin, start = request['observation_step'], commit['commit_step']
        end = origin+len(request['actions'])  # exclusive absolute-action boundary
        alignment = commit.get('action_alignment')
        if alignment:
            low, high = alignment['retained_prediction_offset_interval']
            end = start + alignment['remaining_actions']
            forecast_rows = [{'step': start+i-low, 'action': request['actions'][i]}
                             for i in range(high)]
        else:
            forecast_rows = [{'step': origin+i, 'action': a}
                             for i, a in enumerate(request['actions'])]
        # Both forecast and actual events must have their complete matching window
        # observed inside this forecast's actionable interval. Tail is censored.
        hi = min(end, observed_end)-1
        forecast = command_events(forecast_rows)
        pred = [e for e in forecast if start <= e['step'] <= hi]
        obs = [e for e in actual if start <= e['step'] <= hi]
        pairs = sorted((abs(p['step']-a['step']), i, j)
                       for i, p in enumerate(pred) for j, a in enumerate(obs)
                       if p['type'] == a['type'] and abs(p['step']-a['step']) <= 2)
        used_p, used_a, matches = set(), set(), []
        for _, i, j in pairs:
            if i not in used_p and j not in used_a:
                used_p.add(i); used_a.add(j)
                matches.append({'type': pred[i]['type'], 'predicted_step': pred[i]['step'],
                                'actual_step': obs[j]['step'],
                                'error_steps_actual_minus_predicted': obs[j]['step']-pred[i]['step']})
        unmatched_p = [e for i, e in enumerate(pred) if i not in used_p and start+2 <= e['step'] <= hi-2]
        unmatched_a = [e for j, e in enumerate(obs) if j not in used_a and start+2 <= e['step'] <= hi-2]
        output.append({'prediction_id': request['prediction_id'], 'commit_step': start,
                       'forecast_end_exclusive': end, 'observed_end_exclusive': observed_end,
                       'matches': matches, 'unmatched_predictions': unmatched_p,
                       'unmatched_actual_commands': unmatched_a,
                       'boundary_censored_predictions': [e for i, e in enumerate(pred) if i not in used_p and e not in unmatched_p],
                       'boundary_censored_actual': [e for j, e in enumerate(obs) if j not in used_a and e not in unmatched_a],
                       'action_alignment': alignment,
                       'right_censored_forecast_events': [e for e in forecast if e['step'] >= observed_end and e['step'] >= start]})
    return output


def buffer_statistics(commits):
    records = []
    for c in commits:
        expired = c.get('expired_new_actions')
        new = c.get('new_prediction_remaining_actions', c.get('new_queue_actions'))
        elapsed = c.get('control_steps_until_commit', expired)
        records.append({'prediction_id': c.get('prediction_id'),
                        'observation_step': c.get('observation_step'),
                        'commit_step': c.get('commit_step'),
                        'elapsed_control_steps': c.get('control_steps_until_commit'),
                        'expired_new_actions': expired, 'new_valid_actions': new,
                        'action_alignment': c.get('action_alignment'),
                        'new_queue_actions': c.get('new_queue_actions'),
                        'new_prediction_remaining_interval_steps': c.get('new_prediction_remaining_interval_steps'),
                        'new_valid_seconds': None if new is None else new*DT,
                        'old_queue_actions': c.get('old_queue_remaining_actions'),
                        'old_queue_seconds': None if c.get('old_queue_remaining_actions') is None else c['old_queue_remaining_actions']*DT,
                        'h20_consistent': None if elapsed is None or new is None else new == max(0, H-elapsed)})
    steady = [r for r in records if (isinstance(r['prediction_id'], (int, float)) and r['prediction_id'] > 0)
              or (r['observation_step'] is not None and r['observation_step'] > 0)]
    startup = [r for r in records if r not in steady]
    return {'records': records,
            'new_valid_seconds': describe([r['new_valid_seconds'] for r in records]),
            'old_queue_seconds': describe([r['old_queue_seconds'] for r in records]),
            'steady_state_new_valid_seconds': describe([r['new_valid_seconds'] for r in steady]),
            'steady_state_old_queue_seconds': describe([r['old_queue_seconds'] for r in steady]),
            'bootstrap_new_valid_seconds': describe([r['new_valid_seconds'] for r in startup]),
            'bootstrap_old_queue_seconds': describe([r['old_queue_seconds'] for r in startup])}


def summarize_event_types(events):
    """Counts are forecast-event opportunities, not unique physical events."""
    return {
        kind: {'matched': sum(m['type'] == kind for p in events for m in p['matches']),
               'error_steps': describe([m['error_steps_actual_minus_predicted'] for p in events for m in p['matches'] if m['type'] == kind]),
               'absolute_error_steps': describe([abs(m['error_steps_actual_minus_predicted']) for p in events for m in p['matches'] if m['type'] == kind]),
               'unmatched_predicted': sum(e['type'] == kind for p in events for e in p['unmatched_predictions']),
               'unmatched_actual': sum(e['type'] == kind for p in events for e in p['unmatched_actual_commands']),
               'right_censored_forecast': sum(e['type'] == kind for p in events for e in p['right_censored_forecast_events'])}
        for kind in ('gripper_switch', 'deceleration', 'turn')}


def stage_proxy(row):
    scene = row.get('after', {})
    if 'contacts' not in scene:
        return 'unknown'
    def finger(name):
        return any(s in str(name).lower() for s in ('finger', 'gripper'))
    def robot(name):
        return any(s in str(name).lower() for s in ('robot', 'panda', 'gripper', 'finger'))
    if any(c.get('distance', 0) <= 0 and
           ((finger(c.get('geom1')) and c.get('geom2') and not robot(c['geom2'])) or
           (finger(c.get('geom2')) and c.get('geom1') and not robot(c['geom1'])))
           for c in scene['contacts']):
        return 'finger_external_contact_proxy'
    before, after = row.get('before', {}).get('raw_state'), scene.get('raw_state')
    if before and after:
        if math.dist(before[:3], after[:3])/DT >= 0.01:
            return 'noncontact_motion_proxy'
        return 'noncontact_low_motion_proxy'
    return 'unknown'


def compare_rows(base, candidate):
    a, b = {r['step']: r for r in base}, {r['step']: r for r in candidate}
    common = []
    # An actual common prefix: stop at first missing control step.
    for i in range(min(max(a, default=-1), max(b, default=-1))+1):
        if i not in a or i not in b:
            break
        common.append(i)
    xyz, rot, ee, grip, phases = [], [], [], [], {}
    for i in common:
        ra, rb = a[i], b[i]
        aa, ab = np.asarray(ra['action']), np.asarray(rb['action'])
        xyz.append(float(np.linalg.norm(aa[:3]-ab[:3])))
        rot.append(float(np.linalg.norm(aa[3:6]-ab[3:6])))
        grip.append(float(np.sign(aa[6]) != np.sign(ab[6])))
        sa, sb = ra.get('before', {}).get('raw_state'), rb.get('before', {}).get('raw_state')
        if sa and sb:
            distance = math.dist(sa[:3], sb[:3]); ee.append(distance)
            phases.setdefault(stage_proxy(ra), []).append(distance)
    # Shape diagnostic only, restricted to the SAME common prefix.
    from scripts.analyze_sync_quality import resample
    pa = [a[i]['before']['raw_state'][:3] for i in common if a[i].get('before', {}).get('raw_state')]
    pb = [b[i]['before']['raw_state'][:3] for i in common if b[i].get('before', {}).get('raw_state')]
    progress = None
    if len(pa) >= 2 and len(pb) >= 2 and len(pa) == len(pb):
        progress = float(np.mean([math.dist(x, y) for x, y in zip(resample(pa), resample(pb))]))
    return {'common_prefix_steps': len(common),
            'baseline_steps': len(a), 'candidate_steps': len(b),
            'baseline_coverage': len(common)/len(a) if a else None,
            'candidate_coverage': len(common)/len(b) if b else None,
            'translation_command_l2': describe(xyz), 'rotation_command_l2': describe(rot),
            'gripper_sign_disagreement_fraction': float(np.mean(grip)) if grip else None,
            'eef_xyz_distance_m': describe(ee), 'path_progress_distance_m': progress,
            'baseline_stage_eef_distance_m': {k: describe(v) for k, v in phases.items()}}


def _read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _sync(value):
    return isinstance(value, str) and ('sync' in value.lower() and 'async' not in value.lower()
                                       and 'asynchron' not in value.lower())


def analyze(run_dir):
    root = Path(run_dir)
    summary = _read(root/'summary.json', {})
    run = _read(root/'run.json', {})
    config = summary.get('config') or run.get('config') or run
    if any(_sync(d.get('execution_mode')) for d in (summary, config, run)):
        raise ValueError('synchronous run cannot provide an asynchronous q=5 baseline')
    episodes, exclusions, warnings = {}, [], []
    traces = {}
    for ep in summary.get('episodes', []):
        if _sync(ep.get('execution_mode')) or ep.get('q') not in QS:
            exclusions.append({'task_id': ep.get('task_id'), 'q': ep.get('q'), 'reason': 'wrong mode/q'})
            continue
        if ep.get('status') not in ('success', 'timeout', 'failed', 'failure'):
            exclusions.append({'task_id': ep.get('task_id'), 'q': ep.get('q'), 'reason': 'incomplete/invalid episode'})
            continue
        key = (ep.get('suite', ''), ep['task_id'], ep['q'])
        if key in episodes:
            raise ValueError('duplicate task/q episodes: '+str(key))
        artifact = root/ep.get('artifact_dir', f"tasks/task{ep['task_id']:02d}")
        if artifact not in traces:
            rows = []
            trace = artifact/'trace.jsonl'
            if trace.exists():
                for line in trace.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        warnings.append(str(trace)+': incomplete JSONL record ignored')
            else:
                warnings.append(str(trace)+': missing trace')
            traces[artifact] = rows
        trace = [r for r in traces[artifact] if r.get('q') == ep['q']]
        if any(_sync(r.get('execution_mode')) for r in trace if r.get('event') == 'episode_start'):
            exclusions.append({'task_id': ep['task_id'], 'q': ep['q'], 'reason': 'synchronous episode_start'})
            continue
        rows = [r for r in trace if r.get('event') == 'control_step']
        requests = _read(artifact/f"q{ep['q']}_requests.json", [])
        commits = ep.get('commits', [])
        episode = dict(ep)
        task_summary = _read(artifact/'summary.json', {})
        if 'problem_metadata' not in episode and 'problem_metadata' in task_summary:
            episode['problem_metadata'] = task_summary['problem_metadata']
        episode['rows'] = rows
        episode['event_comparisons'] = prediction_events(rows, requests, commits)
        planned_rows = [dict(r, action=r.get('planned_action', r['action'])) for r in rows]
        episode['planned_command_event_comparisons'] = prediction_events(planned_rows, requests, commits)
        episode['command_events_before_after_gain'] = {
            'planned': command_events(planned_rows), 'applied': command_events(rows),
            'note': 'Applied command events include pending arm gain changes; not isolated model prediction error.'}
        complete_requests = [r for r in requests if len(r.get('actions', [])) == H]
        episode['raw_prediction_event_comparisons'] = prediction_events(
            rows, complete_requests,
            [{'prediction_id': r['prediction_id'], 'commit_step': r['observation_step']}
             for r in complete_requests])
        # Uncommitted predictions are diagnostic only, not mixed with actionable
        # forecasts; there is no actual commit and therefore no handoff claim.
        committed_ids = {c['prediction_id'] for c in commits}
        uncommitted = [r for r in requests if r['prediction_id'] not in committed_ids]
        episode['uncommitted_raw_prediction_events'] = prediction_events(
            rows, uncommitted,
            [{'prediction_id': r['prediction_id'], 'commit_step': r['observation_step']}
             for r in uncommitted])
        episode['buffer'] = buffer_statistics(commits)
        phase_counts = {}
        for row in rows:
            phase = stage_proxy(row); phase_counts[phase] = phase_counts.get(phase, 0)+1
        episode['stage_proxy_steps'] = phase_counts
        episode['stage_gripper_state_diagnostics'] = {
            phase: {'raw_finger_qpos_sum_abs': describe([
                sum(abs(x) for x in r['after']['raw_state'][6:8]) for r in rows
                if stage_proxy(r) == phase and len(r.get('after', {}).get('raw_state', [])) >= 8]),
                'positive_gripper_command_steps': sum(r['action'][6] > 0 for r in rows if stage_proxy(r) == phase),
                'negative_gripper_command_steps': sum(r['action'][6] < 0 for r in rows if stage_proxy(r) == phase)}
            for phase in phase_counts}
        episodes[key] = episode
    by_q, paired = {}, {}
    for q in QS:
        eps = [e for k, e in episodes.items() if k[2] == q]
        n, successes = len(eps), sum(e['status'] == 'success' for e in eps)
        commits = [c for e in eps for c in e.get('commits', [])]
        events = [p for e in eps for p in e['event_comparisons']]
        by_q[str(q)] = {'n': n, 'successes': successes,
                       'success_rate': successes/n if n else None,
                       'success_rate_ci95': clopper_pearson(successes, n),
                       'attempt_real_seconds': describe([e.get('real_seconds') for e in eps]),
                       'successful_completion_real_seconds': describe([
                           e.get('real_seconds') for e in eps if e['status'] == 'success']),
                       'buffer': buffer_statistics(commits),
                       'busy_trigger_count': sum(len(e.get('busy_trigger_steps', [])) for e in eps),
                       'underflow_steps': sum(len(e.get('underflow_steps', [])) for e in eps),
                       'underflow_nominal_seconds': sum(e.get('underflow_nominal_seconds', DT*len(e.get('underflow_steps', []))) for e in eps),
                       'underflow_env_step_real_seconds': sum(e.get('underflow_env_step_real_seconds', 0) for e in eps),
                       'actual_refresh_intervals_steps': describe([i for e in eps for i in e.get('actual_refresh_intervals_steps', [])]),
                       'event_error_steps': describe([m['error_steps_actual_minus_predicted'] for p in events for m in p['matches']]),
                       'unmatched_predicted_events': sum(len(p['unmatched_predictions']) for p in events),
                       'unmatched_actual_command_events': sum(len(p['unmatched_actual_commands']) for p in events),
                       'right_censored_forecast_events': sum(len(p['right_censored_forecast_events']) for p in events)}
        by_q[str(q)]['actionable_events_by_type'] = summarize_event_types(events)
        by_q[str(q)]['events_by_type'] = by_q[str(q)]['actionable_events_by_type']
        by_q[str(q)]['raw_events_by_type'] = summarize_event_types(
            [p for e in eps for p in e['raw_prediction_event_comparisons']])
        by_q[str(q)]['by_suite'] = {
            suite: {'n': sum(e.get('suite', '') == suite for e in eps),
                    'successes': sum(e.get('suite', '') == suite and e['status'] == 'success' for e in eps)}
            for suite in sorted({e.get('suite', '') for e in eps})}
        if q == 5:
            continue
        task_keys = sorted({k[:2] for k in episodes if k[2] in (5, q)})
        pairs, missing, comparisons = [], [], []
        for suite, task_id in task_keys:
            base, cand = episodes.get((suite, task_id, 5)), episodes.get((suite, task_id, q))
            if base is None or cand is None:
                missing.append({'suite': suite, 'task_id': task_id,
                                'missing_q': 5 if base is None else q})
                continue
            mismatch = [name for name in ('initial_sim_state_sha256', 'initial_raw_state')
                        if name in base and name in cand and base[name] != cand[name]]
            if base.get('initial_frames_sha256') != cand.get('initial_frames_sha256'):
                comparison = cand.get('initial_frame_comparison', {})
                if not (comparison.get('accepted') and
                        comparison.get('baseline_sha256') == base.get('initial_frames_sha256')):
                    mismatch.append('initial_frames_sha256')
            if mismatch:
                missing.append({'suite': suite, 'task_id': task_id,
                                'reason': 'initial conditions mismatch', 'fields': mismatch})
                continue
            pairs.append((base, cand))
            c = compare_rows(base['rows'], cand['rows'])
            c['initial_conditions_verified'] = all(name in base and name in cand for name in
                ('initial_sim_state_sha256', 'initial_frames_sha256', 'initial_raw_state'))
            comparisons.append({'suite': suite, 'task_id': task_id, **c})
        metrics = {}
        for name in ('translation_command_l2', 'rotation_command_l2', 'eef_xyz_distance_m'):
            metrics[name] = bootstrap_mean([c[name]['mean'] for c in comparisons if c[name]['mean'] is not None])
        for name in ('gripper_sign_disagreement_fraction', 'path_progress_distance_m'):
            metrics[name] = bootstrap_mean([c[name] for c in comparisons if c[name] is not None])
        metrics['real_seconds_difference'] = bootstrap_mean([
            b['real_seconds']-a['real_seconds'] for a, b in pairs
            if a['status'] == 'success' and b['status'] == 'success'
            and a.get('real_seconds') is not None and b.get('real_seconds') is not None])
        metrics['attempt_elapsed_difference'] = bootstrap_mean([
            b['real_seconds']-a['real_seconds'] for a, b in pairs
            if a.get('real_seconds') is not None and b.get('real_seconds') is not None])
        paired[str(q)] = {'success_difference': paired_success([a['status']=='success' for a, b in pairs],
                                                              [b['status']=='success' for a, b in pairs]),
                          'missing_pairs': missing, 'continuous_metrics': metrics, 'tasks': comparisons}
    clean_episodes = [{k: v for k, v in e.items() if k != 'rows'} for e in episodes.values()]
    return {'status': summary.get('status'), 'baseline': 'asynchronous q=5', 'config': config,
            'event_rules': EVENT_RULES, 'q': by_q, 'paired_vs_q5': paired,
            'episodes': clean_episodes, 'exclusions': exclusions, 'warnings': sorted(set(warnings)),
            'limitations': [
                '30 tasks x 1: task-sample uncertainty only; no within-task repeat randomness estimate.',
                'Task-level bootstrap can degenerate for constant samples; never establishes noninferiority.',
                'Fewer than 2 valid tasks: bootstrap CI unavailable. Completion times use successes only; paired completion differences use mutual successes only.',
                'No trajectory tolerance is specified; no trajectory pass/fail conclusion.',
                'Same-step distances are phase-sensitive; path-progress distances describe prefix shape, not truth error.',
                'Prediction comparisons count command events per committed forecast, not physical grasp/contact detection.',
                'Applied command events include deliberate pending arm gain changes; planned-command comparisons are separate, not independent ground truth.',
                'Unmatched events near either matching boundary are censored; forecast beyond observed termination is right-censored.',
                'Stage labels are transparent geom-name/contact/motion proxies; stable transport and fine placement remain unknown.',
                'External disturbance not tested; remaining valid prediction interval is not B_safe.',
                'All valid completed episodes retained, including failed/timeout episodes; pairs use available intersections.']}


def render(result):
    lines = ['# 异步 q 对比（基线 q=5）', '',
             f"运行状态：{result['status']}。增量分析只采用已经结束且有效的回合。", '',
             '| q | 完成数 | 成功数 | 成功率95%区间 | 相对q5成功率差95%区间 | 稳态新预测有效时长均值(s) | 稳态旧队列余量均值(s) | busy/断供步 |',
             '|---|---:|---:|---|---|---:|---:|---|']
    def fmt(value):
        if value is None:
            return '缺失'
        if isinstance(value, list):
            return '['+', '.join(f'{v:.4f}' for v in value)+']'
        return f'{value:.4f}'
    for q, row in result['q'].items():
        pair = result['paired_vs_q5'].get(q, {}).get('success_difference', {})
        lines.append(f"| {q} | {row['n']} | {row['successes']} | {fmt(row['success_rate_ci95'])} | {fmt(pair.get('ci95')) if q != '5' else '基线'} | {fmt(row['buffer']['steady_state_new_valid_seconds']['mean'])} | {fmt(row['buffer']['steady_state_old_queue_seconds']['mean'])} | {row['busy_trigger_count']}/{row['underflow_steps']} |")
    lines += ['', '上表缓冲采用稳态提交（prediction_id>0或observation_step>0），排除首次阻塞推理。all统计保留在JSON；bootstrap首轮缓冲单列如下（此处bootstrap指启动推理，不是统计重采样）。', '',
              '| q | 首轮提交数 | 首轮新预测有效时长均值(s) | 首轮旧队列余量均值(s) |',
              '|---|---:|---:|---:|']
    for q, row in result['q'].items():
        startup = row['buffer']
        lines.append(f"| {q} | {startup['bootstrap_new_valid_seconds']['n']} | {fmt(startup['bootstrap_new_valid_seconds']['mean'])} | {fmt(startup['bootstrap_old_queue_seconds']['mean'])} |")
    lines += ['', '成功率使用Clopper–Pearson区间；配对成功率差使用gain/loss两个97.5%精确区间的保守组合（Bonferroni覆盖率至少95%）。连续指标先逐任务求均值，再按任务bootstrap 4000次，seed=7。少于2个有效任务不计算bootstrap区间；至少2个但数值相同的样本保留constant_sample标记，其退化区间不能证明非劣。', '',
              '| q | 全尝试耗时均值(s) | 成功完成时间均值(s) | 成功任务数 |',
              '|---|---:|---:|---:|']
    for q, row in result['q'].items():
        lines.append(f"| {q} | {fmt(row['attempt_real_seconds']['mean'])} | {fmt(row['successful_completion_real_seconds']['mean'])} | {row['successful_completion_real_seconds']['n']} |")
    lines += ['', '| q对q5 | 共同成功任务数 | 完成时间差均值(s)及95%区间 | 全尝试终止耗时差均值(s)及95%区间 |',
              '|---|---:|---|---|']
    for q, pair in result['paired_vs_q5'].items():
        completion = pair['continuous_metrics']['real_seconds_difference']
        attempt = pair['continuous_metrics']['attempt_elapsed_difference']
        lines.append(f"| {q} | {completion['n']} | {fmt(completion['mean'])} {fmt(completion['ci95'])} | {fmt(attempt['mean'])} {fmt(attempt['ci95'])} |")
    lines += ['', 'attempt_real_seconds包含成功、失败及超时的终止耗时；successful_completion_real_seconds只包含成功任务。real_seconds_difference只计算双方共同成功的任务，attempt_elapsed_difference包含所有有效配对，仅表示终止耗时差，不能称完成时间改善。差值均为当前q减q5。', '',
              '| q对q5 | 配对任务 | xyz命令差 | 旋转命令差 | 末端xyz距离(m) | 夹爪符号不同占比 | 路程进度距离(m) |',
              '|---|---:|---|---|---|---|---|']
    for q, pair in result['paired_vs_q5'].items():
        cells = [f'{q}', str(pair['success_difference']['n'])]
        for name in ('translation_command_l2', 'rotation_command_l2', 'eef_xyz_distance_m',
                     'gripper_sign_disagreement_fraction', 'path_progress_distance_m'):
            metric = pair['continuous_metrics'][name]
            cells.append(fmt(metric['mean'])+' '+fmt(metric['ci95']))
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '| q对q5 | 共同前缀覆盖率：q5均值 | 当前q均值 | 缺失/初态不符配对数 |',
              '|---|---:|---:|---:|']
    for q, pair in result['paired_vs_q5'].items():
        a = describe([t['baseline_coverage'] for t in pair['tasks']])['mean']
        b = describe([t['candidate_coverage'] for t in pair['tasks']])['mean']
        lines.append(f"| {q} | {fmt(a)} | {fmt(b)} | {len(pair['missing_pairs'])} |")
    lines += ['', '动作和末端轨迹只比较相同控制步的共同前缀，逐任务coverage在analysis.json中。路程归一化仅诊断进度/相位敏感性，也只使用共同前缀。末端xyz取raw_state前三维；动作xyz/旋转分开，夹爪按符号。轨迹容忍阈值未定，不判通过。', '',
              '每次提交保留expired_new_actions、control_steps_until_commit、new_queue_actions和old_queue_remaining_actions；新预测有效时长为new_prediction_remaining_actions×0.05秒（旧记录回退new_queue_actions），旧队列余量单独列出，不能称B_safe。启动推理即使墙钟耗时较长，零控制步过去就不扣动作。', '',
              'raw_prediction_event_comparisons：每个完整预测从observation_step开始，与后续实际执行命令作后验对照，包含已裁掉的推理期间前缀及终止时未提交预测。它衡量完整预测与后来执行命令的差异，不是独立真值误差。event_comparisons（actionable）：仅从真正commit_step开始，衡量结果就绪后仍可使用部分的预测误差；这部分命令可能直接来自该预测，存在自我一致性偏差。两套结果分开报告，均右删失不可观测尾部，不能把较小actionable误差解释成更强预测能力。', '',
              '事件规则：'+json.dumps(EVENT_RULES, ensure_ascii=False), '',
              '| q | 事件误差均值(实际−预测步) | 未匹配预测 | 未匹配实际命令 | 右删失预测 |',
              '|---|---:|---:|---:|---:|']
    for q, row in result['q'].items():
        lines.append(f"| {q} | {fmt(row['event_error_steps']['mean'])} | {row['unmatched_predicted_events']} | {row['unmatched_actual_command_events']} | {row['right_censored_forecast_events']} |")
    lines += ['', '上表为提交后actionable汇总。下表按类型分别列完整raw和actionable；误报/漏报为各预测窗口的未匹配预测/实际命令次数，同一实际事件可在不同预测窗口重复参与比较。', '',
              '| q | 范围 | 命令事件 | 匹配数 | 误差均值(步) | 绝对误差均值(步) | 误报 | 漏报 | 右删失预测 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for q, row in result['q'].items():
        for scope in ('raw', 'actionable'):
            for kind, metric in row[scope+'_events_by_type'].items():
                lines.append(f"| {q} | {scope} | {kind} | {metric['matched']} | {fmt(metric['error_steps']['mean'])} | {fmt(metric['absolute_error_steps']['mean'])} | {metric['unmatched_predicted']} | {metric['unmatched_actual']} | {metric['right_censored_forecast']} |")
    lines += ['', '这些是执行命令事件，不能等同于抓住物体。接触阶段仅用手指/夹爪geom与非机器人geom接触作proxy；其余按末端运动分为非接触运动/低速，精细放置与稳定搬运无法可靠识别，标为unknown。外扰未测。', '',
              '| q对q5 | q5接触proxy末端差(m) | q5非接触运动proxy末端差(m) |',
              '|---|---|---|']
    for q, pair in result['paired_vs_q5'].items():
        cells = []
        for phase in ('finger_external_contact_proxy', 'noncontact_motion_proxy'):
            metric = bootstrap_mean([t['baseline_stage_eef_distance_m'][phase]['mean'] for t in pair['tasks']
                                     if phase in t['baseline_stage_eef_distance_m']])
            cells.append(fmt(metric['mean'])+' '+fmt(metric['ci95']))
        lines.append('| '+q+' | '+' | '.join(cells)+' |')
    lines += ['',
              '30任务各一次的区间只反映任务样本，不包含同一任务重复运行的随机性。缺失配对、阶段proxy、逐提交事件匹配及缓冲明细保留在analysis.json。', '',
              '记录警告：'+('；'.join(result['warnings']) or '无')]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    (args.run_dir/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (args.run_dir/'analysis.md').write_text(render(result))
    print(args.run_dir/'analysis.md')


if __name__ == '__main__':
    # Allow both python scripts/analyze_async_q.py and importing from tests.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
