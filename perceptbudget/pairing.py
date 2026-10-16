"""Numerical initial-state checks; retain hashes and audit every tolerated delta."""

import json
import math
from pathlib import Path

STATE_ATOL = 1e-7
RAW_ATOL = 1e-8
TIME_ATOL = 1e-12


def _sim_state(episode, run_dir):
    if 'initial_sim_state' in episode:
        return episode['initial_sim_state']
    if run_dir is not None and 'artifact_dir' in episode:
        path = Path(run_dir) / episode['artifact_dir'] / 'trace.jsonl'
        if path.is_file():
            with path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get('event') == 'control_step' and record.get('q') == episode.get('q'):
                        return record['before']['sim_state']
    raise RuntimeError('initial_sim_state differs but original state data is unavailable')


def _delta(a, b, label):
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not a.size or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise RuntimeError('Invalid initial state shape/nonfinite: ' + label)
    return float(np.max(np.abs(a-b)))


def validate_initial_state_pair(baseline, candidate, run_dir=None):
    same_sim = baseline['initial_sim_state_sha256'] == candidate['initial_sim_state_sha256']
    same_raw = baseline.get('initial_raw_state') == candidate.get('initial_raw_state')
    if same_sim and same_raw:
        return
    raw_a, raw_b = baseline.get('initial_raw_state'), candidate.get('initial_raw_state')
    if raw_a is None or raw_b is None or len(raw_a) != 8 or len(raw_b) != 8:
        raise RuntimeError('initial_sim_state differs without valid 8D initial_raw_state')
    raw_delta = _delta(raw_a, raw_b, 'initial_raw_state')
    delta = time_delta = 0.0
    if not same_sim:
        a, b = _sim_state(baseline, run_dir), _sim_state(candidate, run_dir)
        delta = _delta(a, b, 'initial_sim_state')
        time_delta = abs(float(a[0])-float(b[0]))
    comparison = {'baseline_sha256': baseline['initial_sim_state_sha256'],
                  'baseline_raw_state': list(raw_a), 'max_abs_difference': delta,
                  'raw_max_abs_difference': raw_delta, 'sim_time_abs_difference': time_delta,
                  'atol': STATE_ATOL, 'raw_atol': RAW_ATOL, 'time_atol': TIME_ATOL,
                  'rtol': 0.0,
                  'accepted': delta <= STATE_ATOL and raw_delta <= RAW_ATOL and time_delta <= TIME_ATOL}
    candidate['initial_state_comparison'] = comparison
    if not comparison['accepted']:
        raise RuntimeError('Initial conditions differ beyond initial_sim_state tolerance: ' + repr(comparison))


def initial_state_mismatches(baseline, candidate):
    """Offline analysis accepts only evidence tied to THIS exact baseline."""
    mismatch = [name for name in ('initial_sim_state_sha256', 'initial_raw_state')
                if name in baseline and name in candidate and baseline[name] != candidate[name]]
    if not mismatch:
        return []
    c = candidate.get('initial_state_comparison', {})
    if (not c.get('accepted') or c.get('baseline_sha256') != baseline.get('initial_sim_state_sha256') or
            c.get('baseline_raw_state') != baseline.get('initial_raw_state') or c.get('rtol') != 0):
        return mismatch
    for key, bound, tolerance in [('max_abs_difference', STATE_ATOL, 'atol'),
                                  ('raw_max_abs_difference', RAW_ATOL, 'raw_atol'),
                                  ('sim_time_abs_difference', TIME_ATOL, 'time_atol')]:
        value, claimed = c.get(key), c.get(tolerance)
        if (not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= bound or
                not isinstance(claimed, (int, float)) or not 0 <= claimed <= bound or value > claimed):
            return mismatch
    return []
