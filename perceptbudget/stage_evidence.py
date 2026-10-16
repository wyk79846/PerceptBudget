"""Conservative offline frame-review proxies, never grasp or success ground truth.

Mapping checked against historical q8 task00/task03 episode_start and scene
observations (20261016T035800.217970Z-async-q8-screen-3650b7d9).
Task3 starts on cookies_1; departure may be horizontal or downward.
"""

import math

TASK_OBJECTS = {0: ('akita_black_bowl_1', 'plate_1'),
                3: ('akita_black_bowl_1', 'plate_1')}
DEFAULT_THRESHOLDS = dict(window_steps=3, motion_m=0.006,
                          relative_tolerance_m=0.008, departure_m=0.015,
                          goal_xy_m=0.06, goal_z_m=0.09, stationary_m=0.004,
                          descent_m=0.002, adjustment_min_m=0.0005,
                          adjustment_max_m=0.004)


def _vector(value):
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        if all(isinstance(x, (int, float)) and math.isfinite(x) for x in value[:3]):
            return tuple(value[:3])
    return None


def _position(scene, name):
    return _vector(scene.get('observations', {}).get(name + '_pos'))


def _matches(geom, name):
    return geom == name or geom.startswith(name + '_')


def _contacts(scene, target):
    if scene.get('contacts') is None:
        return None, None
    finger, supports = False, set()
    for c in scene['contacts']:
        if c.get('distance', math.inf) > 0:
            continue
        a, b = c.get('geom1') or '', c.get('geom2') or ''
        other = b if _matches(a, target) else a if _matches(b, target) else ''
        if 'finger' in other:
            finger = True
        elif other and not other.startswith(('robot', 'gripper')):
            supports.add(other)
    return finger, supports


def annotate_episode(rows, target, goal, thresholds=None):
    """Return steps, inclusive segments and thresholds for one ordered episode.

    Each step has phase and independent evidence (unknown contact is None).
    Windows are trailing and require consecutive control step IDs. Candidate
    boundaries are confirmation steps, not inferred earlier grasp instants.
    Missing observations never get filled from other objects or sim indices.
    Placement requires transport on an earlier step. Approach reasons describe
    held-object descent within one control step or consecutive small adjustments
    near the goal; these are frame-review candidates, not fine-placement truth.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if set(t) != set(DEFAULT_THRESHOLDS):
        raise ValueError('unknown threshold')
    if (isinstance(t['window_steps'], bool) or
            not isinstance(t['window_steps'], int) or t['window_steps'] < 2):
        raise ValueError('window_steps must be an integer >= 2')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in t.values()):
        raise ValueError('thresholds must be positive finite numbers')
    if t['adjustment_min_m'] > t['adjustment_max_m']:
        raise ValueError('adjustment_min_m must not exceed adjustment_max_m')
    rows = [r for r in rows if r.get('event', 'control_step') == 'control_step']
    ids = [r['step'] for r in rows]
    if any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError('expected one episode with increasing step IDs')
    initial = rows[0].get('before', {}) if rows else {}
    origin = _position(initial, target)
    _, initial_support = _contacts(initial, target)
    samples, steps, segments = [], [], []
    transport_seen = False
    for row in rows:
        scene = row.get('after', {})
        p, g = _position(scene, target), _position(scene, goal)
        eef = _vector(scene.get('raw_state'))
        finger, support = _contacts(scene, target)
        relative = tuple(a - b for a, b in zip(p, eef)) if p and eef else None
        samples.append(dict(p=p, g=g, eef=eef, relative=relative, finger=finger))
        n = t['window_steps']
        window = samples[-n:]
        consecutive = len(window) == n and row['step'] - ids[len(samples)-n] == n-1
        motion_ok = consecutive and all(s['p'] and s['eef'] for s in window)
        object_motion = math.dist(window[0]['p'], p) if motion_ok else None
        eef_motion = math.dist(window[0]['eef'], eef) if motion_ok else None
        stable_relative = bool(motion_ok and all(
            math.dist(s['relative'], relative) <= t['relative_tolerance_m'] for s in window))
        departed = (bool(math.dist(origin, p) >= t['departure_m'] and
                         (not initial_support or not initial_support.intersection(support)))
                    if origin and p and support is not None else None)
        near_goal = (math.dist(p[:2], g[:2]) <= t['goal_xy_m'] and
                     abs(p[2] - g[2]) <= t['goal_z_m']) if p and g else None
        goal_contact = (any(_matches(s, goal) for s in support)
                        if support is not None else None)
        released = bool(consecutive and all(s['finger'] is False for s in window))
        stationary = bool(consecutive and all(s['p'] for s in window) and
                          all(math.dist(s['p'], p) <= t['stationary_m'] for s in window))
        transport = bool(departed and stable_relative and
                         object_motion >= t['motion_m'] and eef_motion >= t['motion_m'] and
                         all(s['finger'] is True for s in window))
        before = row.get('before', {})
        bp, bg = _position(before, target), _position(before, goal)
        be = _vector(before.get('raw_state'))
        held_step = bool(bp and be and relative and finger is True and
                         _contacts(before, target)[0] is True and
                         math.dist(tuple(a-b for a, b in zip(bp, be)), relative)
                         <= t['relative_tolerance_m'])
        descending = bool(held_step and bg and g and p[2] >= g[2] and
                          (bp[2]-bg[2]) - (p[2]-g[2]) >= t['descent_m'])
        adjustments = bool(stable_relative and all(s['finger'] is True and s['g'] and
            math.dist(s['p'][:2], s['g'][:2]) <= t['goal_xy_m'] and
            abs(s['p'][2]-s['g'][2]) <= t['goal_z_m'] for s in window) and all(
            t['adjustment_min_m'] <= math.dist(a['p'], b['p']) <= t['adjustment_max_m']
            for a, b in zip(window, window[1:])))
        reasons = (['target_descending_toward_goal'] if descending else [])
        if adjustments:
            reasons.append('continuous_small_adjustments_near_goal')
        approach = bool(transport_seen and near_goal and reasons)
        placement = bool(transport_seen and departed and near_goal and
                         goal_contact and released and stationary)
        phase = ('placement_candidate' if placement else
                 'placement_approach_candidate' if approach else
                 'transport_candidate' if transport else 'unknown')
        evidence = dict(finger_target_contact=finger, object_observed=p is not None,
                        departed_initial_support=departed, near_goal=near_goal,
                        goal_support_contact=goal_contact, released_window=released,
                        stationary_window=stationary, stable_relative_window=stable_relative,
                        object_motion_m=object_motion, eef_motion_m=eef_motion,
                        prior_transport_evidence=transport_seen,
                        placement_approach_reasons=reasons if approach else [])
        transport_seen = transport_seen or transport
        steps.append(dict(step=row['step'], phase=phase, evidence=evidence))
        if (not segments or segments[-1]['phase'] != phase or
                segments[-1]['end_step'] + 1 != row['step']):
            segments.append(dict(phase=phase, start_step=row['step'], end_step=row['step']))
        else:
            segments[-1]['end_step'] = row['step']
    return dict(target=target, goal=goal, evidence_kind='conservative_proxy',
                steps=steps, segments=segments, thresholds=t)
