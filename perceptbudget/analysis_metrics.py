"""Shared paired statistics and action-event metrics; no experiment CLI."""
import math
import numpy as np

SEED = 7

BOOTSTRAPS = 4000

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
