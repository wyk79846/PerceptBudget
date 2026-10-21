"""Offline schema-1 run_m1_h50 batch analysis. No model or device access."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_async_q import (bootstrap_mean, describe, EVENT_RULES,
                                    prediction_events, summarize_event_types)
from scripts.analyze_sync_quality import resample
from scripts.analyze_opportunity import _merge, _number

SUITES = ('libero_spatial', 'libero_object', 'libero_goal')
QS = (5, 10, 20, 30, 40)
MODES = ('sync_no_rtc', 'actual_history')
NOTES = [
    'Pairs with an explicitly saved exact_physics_render_lsb_v1 review have exact recorded physics but non-bit-identical images (at most4 pixels differ by1/255); these pairs are qualified separately, not called identical inputs.',
    'Energy summaries cover the currently indexed worker per cell; historical invalid/retried attempt energy is not accumulated and must be reported separately before claiming total retry cost.',
    'Main scan only; power calibration, six perturbations and physical phases are not delivered here. Phase=unknown; B_safe unverified.',
    'Trajectory: each complete observed pre-action EEF xyz path is resampled to 101 normalized cumulative-distance points; mean Euclidean deviation in metres versus same-task sync q5. Shape diagnostic, not ground-truth error; failed paths may end early.',
    'Events use absolute control step = observation_step + prediction index; report raw and committed forecasts separately. Command switches/deceleration/turns are not physical grasp events; committed forecasts have self-consistency bias. Counts are forecast opportunities, not independent physical events.',
    '95% intervals resample paired tasks, never timesteps; fewer than two pairs gives NA. Success differences are fractions (0.05 = 5 percentage points). Groups have at most 10 tasks; overall groups at most 30.',
    'Busy duty uses the union of saved inference intervals clipped to the task window, including bootstrap and discarded work. Gaps include task-window boundary gaps, exclude load/warmup and drain.',
    'Nominal coverage seconds use 0.05 s per step, not measured 20 Hz; remaining H-actual elapsed is not B_safe. VDD_IN energy is module input, not GPU-only energy.',
    'Missing or invalid evidence is NA (JSON null); descriptive means report their known sample count. Energy per success requires every observed attempt energy, including invalid attempts; missing cells are excluded and progress remains explicit.',
]


def read(path):
    return json.loads(path.read_text())


def relative(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def energy(ep, field):
    value = _number((ep.get(field) or {}).get('total_energy_mj'))
    return value / 1000 if value is not None else None


def total(values):
    return sum(values) if values and all(v is not None for v in values) else None


def timeline(requests, start, stop):
    if (requests is None or not requests or start is None or stop is None or stop <= start):
        return dict(busy_seconds=None, duty=None, busy_intervals=None, idle_gaps_seconds=None)
    intervals = []
    for r in requests:
        a, b = (_number(r.get(k)) for k in
                ('inference_started_monotonic', 'inference_ended_monotonic'))
        if a is None or b is None or b < a:
            return dict(busy_seconds=None, duty=None, busy_intervals=None, idle_gaps_seconds=None)
        intervals.append((a, b))
    busy = _merge(intervals, start, stop)
    cursor, gaps = start, []
    for a, b in busy:
        if a > cursor:
            gaps.append(a-cursor)
        cursor = b
    if cursor < stop:
        gaps.append(stop-cursor)
    seconds = sum(b-a for a, b in busy)
    return dict(busy_seconds=seconds, duty=seconds/(stop-start),
                busy_intervals=busy, idle_gaps_seconds=gaps)


def trajectory(a, b):
    paths = []
    for rows in (a, b):
        points = [r.get('before', {}).get('raw_state', [])[:3] for r in rows]
        if len(points) < 2 or any(len(p) != 3 or any(_number(x) is None for x in p) for p in points):
            return None
        paths.append(resample(points))
    return sum(math.dist(x, y) for x, y in zip(*paths))/101


def load_cell(record, batch):
    s, t, q, mode = record['cell']
    item = dict(suite=s, task_id=t, q=q, mode=mode, index_status=record.get('status'),
                valid=False, warnings=[], phase='unknown')
    if record.get('status') != 'completed':
        item['warnings'].append('cell not completed; excluded from outcomes')
        return item
    worker = relative(record['result']['run'], batch)
    item['worker'] = str(worker)
    metadata = read(worker/'run.json')
    cfg = metadata['config_snapshot']
    ep = read(worker/'summary.json')['episode']
    # Detect identity/config errors without importing the experiment launcher.
    for k, v in dict(H=50, N=5, candidate_q=[q], rtc_suite=s,
                     state_enabled=True, rtc_exact_flash_attention=True,
                     rtc_fused_norm=True, native_gemm_policy='autotune',
                     diagnostic_condition=mode,
                     checkpoint_profile='lerobot_h50_quantiles_v044',
                     checkpoint_provenance_revision='b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd').items():
        if k not in cfg:
            raise ValueError('missing frozen config: '+k)
        elif cfg[k] != v:
            raise ValueError('frozen config mismatch: '+k)
    if ep.get('q') != q or ep.get('task_id') != t or ep.get('status') not in ('success', 'timeout', 'failure'):
        raise ValueError('episode identity/outcome mismatch')
    item.update(valid=True, outcome=ep['status'], steps=_number(ep.get('steps')),
                real_seconds=_number(ep.get('real_seconds')),
                simulated_seconds=_number(ep.get('simulated_seconds')),
                task_energy_j=energy(ep, 'power'), drain_energy_j=energy(ep, 'drain_power'),
                including_drain_energy_j=energy(ep, 'including_drain_power'),
                drain_seconds=_number(ep.get('drain_seconds')),
                initial_hashes=record['result'].get('initial_hashes'),
                initial_pair_audit=record.get('initial_pair_audit'))
    trace_path, request_path = worker/'rtc/trace.jsonl', worker/f'rtc/q{q}_requests.json'
    rows, commits = None, None
    if trace_path.exists():
        events = []
        for line in trace_path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                item['warnings'].append('incomplete trace JSONL line')
        rows = [r for r in events if r.get('event') == 'control_step']
        if [r.get('step') for r in rows] != list(range(len(rows))) or len(rows) != ep.get('steps'):
            item['warnings'].append('trace control prefix/step count mismatch')
            rows = None
        commits = [r for r in events if r.get('event') == 'prediction_committed']
        if not commits:
            item['warnings'].append('missing commit records')
            commits = None
    else:
        item['warnings'].append('missing trace')
    requests = read(request_path) if request_path.exists() else None
    if not requests:
        item['warnings'].append('missing/empty requests')
        requests = None
    item['_rows'] = rows
    start, stop = (_number(ep.get(k)) for k in ('start_monotonic', 'stop_monotonic'))
    item['timeline'] = timeline(requests, start, stop)
    if item['timeline']['duty'] is None:
        item['warnings'].append('incomplete task/inference timeline')
    if rows is not None and all('source' in r for r in rows):
        underflows = [r for r in rows if r['source'] is None]
        item['underflow_steps'] = len(underflows)
        durations = [(_number(r.get('action_start_monotonic')), _number(r.get('action_end_monotonic'))) for r in underflows]
        item['underflow_env_seconds'] = (sum(b-a for a, b in durations)
            if all(a is not None and b is not None and b >= a for a, b in durations) else None)
    else:
        item['underflow_steps'] = None
        item['underflow_env_seconds'] = None
    item['buffer'] = None
    if commits is not None:
        item['buffer'] = []
        for c in commits:
            origin, at = _number(c.get('observation_step')), _number(c.get('commit_step'))
            elapsed = at-origin if origin is not None and at is not None else None
            remaining = 50-elapsed if elapsed is not None else None
            saved = _number(c.get('new_prediction_remaining_actions'))
            item['buffer'].append(dict(prediction_id=c.get('prediction_id'), actual_elapsed_steps=elapsed,
                remaining_steps=remaining, remaining_nominal_seconds=remaining*.05 if remaining is not None else None,
                old_queue_remaining_steps=_number(c.get('old_queue_remaining_actions')),
                recorded_remaining_matches=saved == remaining if saved is not None and remaining is not None else None))
    item['events'] = None
    if rows is not None and requests is not None and commits is not None:
        if all(isinstance(r.get('actions'), list) and len(r['actions']) == 50 for r in requests):
            raw = [dict(prediction_id=r['prediction_id'], commit_step=r['observation_step']) for r in requests]
            item['events'] = {}
            for name, cs in (('raw', raw), ('committed', commits)):
                comparisons = prediction_events(rows, requests, cs)
                item['events'][name] = dict(summary=summarize_event_types(comparisons), records=comparisons)
        else:
            item['warnings'].append('missing/incomplete H50 action predictions')
    for k in ('task_energy_j', 'drain_energy_j', 'including_drain_energy_j', 'drain_seconds'):
        if item[k] is None:
            item['warnings'].append('missing '+k)
    for k in ('steps', 'real_seconds', 'simulated_seconds', 'underflow_steps', 'underflow_env_seconds', 'initial_hashes'):
        if item.get(k) is None:
            item['warnings'].append('missing '+k)
    return item


def paired(items, baselines):
    differences, deviations, excluded, identities = [], [], [], []
    for e in items:
        b = baselines.get((e['suite'], e['task_id']))
        if b is None:
            excluded.append(dict(suite=e['suite'], task_id=e['task_id'], reason='missing sync q5'))
            continue
        a_hash, b_hash = e.get('initial_hashes'), b.get('initial_hashes')
        identity='verified' if a_hash and b_hash else 'unverified'
        if a_hash and b_hash and a_hash != b_hash:
            proof=e.get('initial_pair_audit') or {}
            accepted=(proof.get('passed') is True and proof.get('rule')=='exact_physics_render_lsb_v1'
                and proof.get('reference_hashes')==b_hash and proof.get('candidate_hashes')==a_hash
                and all(a_hash.get(k)==b_hash.get(k) and a_hash.get(k) is not None for k in
                        ('initial_sim_state_sha256','initial_raw_state_sha256'))
                and 0<proof.get('changed_pixels',999)<=4 and 0<proof.get('max_channel_delta',999)<=1)
            if not accepted:
                excluded.append(dict(suite=e['suite'], task_id=e['task_id'], reason='initial hashes mismatch'))
                continue
            identity='exact_physics_qualified_render_lsb'
        identities.append(identity)
        differences.append(int(e['outcome'] == 'success')-int(b['outcome'] == 'success'))
        value = trajectory(b['_rows'], e['_rows']) if b.get('_rows') is not None and e.get('_rows') is not None else None
        if value is not None:
            deviations.append(value)
    return dict(success_difference=bootstrap_mean(differences),
                trajectory_progress_deviation_m=bootstrap_mean(deviations),
                initial_identity=identities, excluded=excluded)


def aggregate(items, baselines, expected):
    valid = [e for e in items if e['valid']]
    success = [e for e in valid if e['outcome'] == 'success']
    pending = sum(e.get('index_status') in ('running', 'pending') for e in items)
    output = dict(expected=expected, observed=len(items), valid=len(valid), pending=pending,
                  invalid=len(items)-len(valid)-pending,
                  successes=len(success), success_rate=len(success)/len(valid) if valid else None,
                  success_steps=describe([e['steps'] for e in success]),
                  success_real_seconds=describe([e['real_seconds'] for e in success]),
                  success_simulated_seconds=describe([e['simulated_seconds'] for e in success]),
                  success_task_energy_j=describe([e['task_energy_j'] for e in success]),
                  success_drain_energy_j=describe([e['drain_energy_j'] for e in success]),
                  success_including_drain_energy_j=describe([e['including_drain_energy_j'] for e in success]),
                  paired_sync_q5=paired(valid, baselines))
    for field in ('task_energy_j', 'drain_energy_j', 'including_drain_energy_j'):
        value = total([e.get(field) for e in items])
        output['total_attempt_'+field] = value
        output['per_success_'+field] = value/len(success) if value is not None and success else None
    output['busy_seconds'] = describe([e['timeline']['busy_seconds'] for e in valid])
    output['duty'] = describe([e['timeline']['duty'] for e in valid])
    output['idle_gaps_seconds'] = describe([v for e in valid for v in (e['timeline']['idle_gaps_seconds'] or [])])
    output['idle_timeline_known_episodes'] = sum(e['timeline']['idle_gaps_seconds'] is not None for e in valid)
    output['underflow_steps'] = total([e['underflow_steps'] for e in valid])
    output['underflow_env_seconds'] = total([e['underflow_env_seconds'] for e in valid])
    output['remaining_steps'] = describe([c['remaining_steps'] for e in valid for c in (e['buffer'] or [])])
    output['actual_elapsed_steps'] = describe([c['actual_elapsed_steps'] for e in valid for c in (e['buffer'] or [])])
    output['old_queue_remaining_steps'] = describe([c['old_queue_remaining_steps'] for e in valid for c in (e['buffer'] or [])])
    output['drain_seconds'] = describe([e['drain_seconds'] for e in valid])
    output['event_evidence_known_episodes'] = sum(e['events'] is not None for e in valid)
    output['events'] = {}
    for scope in ('raw', 'committed'):
        records = [p for e in valid if e['events'] is not None for p in e['events'][scope]['records']]
        output['events'][scope] = summarize_event_types(records) if output['event_evidence_known_episodes'] else None
    return output


def analyze(batch):
    batch = Path(batch).resolve()
    index_path = batch/'index.json' if batch.is_dir() else batch
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    if index.get('schema') != 1:
        raise ValueError('expected run_m1_h50 schema 1')
    expected = set(itertools.product(SUITES, range(10), QS, MODES))
    episodes, seen = [], set()
    for record in index['cells'].values():
        cell = tuple(record['cell'])
        if cell not in expected or cell in seen:
            raise ValueError('unexpected/duplicate cell: '+str(cell))
        seen.add(cell)
        try:
            item = load_cell(record, index_path.parent)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            s, t, q, m = cell
            item = dict(suite=s, task_id=t, q=q, mode=m, valid=False,
                        index_status=record.get('status'), warnings=[str(exc)])
        episodes.append(item)
    baselines = {(e['suite'], e['task_id']): e for e in episodes
                 if e['valid'] and e['q'] == 5 and e['mode'] == 'sync_no_rtc'}
    groups = []
    for q, mode, suite in itertools.product(QS, MODES, (*SUITES, 'all')):
        items = [e for e in episodes if e['q'] == q and e['mode'] == mode and (suite == 'all' or e['suite'] == suite)]
        groups.append(dict(q=q, mode=mode, suite=suite,
                           **aggregate(items, baselines, 30 if suite == 'all' else 10)))
    count = sum(e['valid'] for e in episodes)
    pending = sum(e.get('index_status') in ('running', 'pending') for e in episodes)
    for e in episodes:
        e.pop('_rows', None)
    return dict(status='main_scan_complete' if count == 300 else 'partial', final_delivery=False,
                evidence_status='partial' if any(e['warnings'] for e in episodes) or count < 300 else 'available_for_review',
                expected=300, observed=len(episodes), valid=count, missing=300-len(seen),
                pending=pending, invalid=len(episodes)-count-pending, source_status=index.get('status'),
                index_path=str(index_path), index_sha256=hashlib.sha256(index_bytes).hexdigest(),
                frozen=index.get('provenance', {}).get('frozen'),
                event_rules=EVENT_RULES, limitations=NOTES, groups=groups, episodes=episodes)


def markdown(result):
    def fmt(v):
        return 'NA' if v is None else f'{v:.4g}'
    lines = [f"M1 H50: {result['status']}; evidence={result['evidence_status']}; valid {result['valid']}/300, missing {result['missing']}, pending/running {result['pending']}, invalid {result['invalid']}. Not final M1 delivery.", '',
             '|q|mode|suite|valid/expected|success rate|success steps mean|success wall s mean|task J / success|incl drain J / success|paired Δ success [95%]|path deviation m|',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for g in result['groups']:
        p = g['paired_sync_q5']
        d = p['success_difference']
        lines.append('|'+ '|'.join(map(str, [g['q'], g['mode'], g['suite'], f"{g['valid']}/{g['expected']}",
            fmt(g['success_rate']), fmt(g['success_steps']['mean']), fmt(g['success_real_seconds']['mean']),
            fmt(g['per_success_task_energy_j']), fmt(g['per_success_including_drain_energy_j']),
            f"{fmt(d['mean'])} [{', '.join(map(fmt, d['ci95'])) if d['ci95'] else 'NA'}] n={d['n']}",
            fmt(p['trajectory_progress_deviation_m']['mean'])]))+'|')
    lines += ['', *result['limitations'], '', 'Per-cell events, buffers, busy intervals, gaps, separate drain energy and warnings: analysis.json.']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.batch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous report or raw artifact.
    for name in ('analysis.json', 'analysis.md'):
        if (args.output_dir/name).exists():
            raise FileExistsError(args.output_dir/name)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'analysis.md').write_text(markdown(result))
    print(f"{result['status']}: {result['valid']}/300; {args.output_dir}")


if __name__ == '__main__':
    main()
