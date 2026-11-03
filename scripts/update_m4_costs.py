#!/usr/bin/env python3
"""Refresh current Full/Continue costs from audited lean-capture rollouts."""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import load_config, create_run, _write_json_atomically
from perceptbudget.m4_acceptance import request_path
from perceptbudget.power import PowerReading, _integrate_report
from perceptbudget.service_ledger import build_ledger


def history(age):
    return 'under_0.2s' if age < .2 else '0.2_to_1s' if age < 1. else 'at_least_1s'


def continue_segments(start, stop, clocks, ready_times):
    """Partition once by acknowledged clock and time since last busy/change."""
    changes = [(e['monotonic'], e['profile']) if e['status'] == 'READY' else
               (e['observed_match_monotonic'], e['profile']) for e in clocks
               if e.get('status') in ('READY', 'APPLIED')]
    points = {start, stop}
    origins = sorted(set(ready_times + [t for t, _ in changes]))
    for t in origins:
        points.update(x for x in (t, t+.2, t+1.) if start < x < stop)
    cuts = sorted(points)
    for a, b in zip(cuts, cuts[1:]):
        prior = [(t, p) for t, p in changes if t <= a+1e-9]
        if not prior:
            raise ValueError('No acknowledged frequency before Continue')
        profile = max(prior, key=lambda x: x[0])[1]
        origin = max((t for t in origins if t <= a+1e-9), default=a)
        yield a, b, profile, history(max(0., (a+b)/2-origin))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['mode'], row['profile'], row['from_profile'], row['history'],
               row.get('after_full_profile', 'unspecified')].append(row)
    cells = []
    for key, group in sorted(groups.items()):
        seconds = sum(r['seconds'] for r in group)
        energy = sum(r['energy_j'] for r in group)
        cells.append(dict(zip(('mode', 'profile', 'from_profile', 'history', 'after_full_profile'), key),
            observations=len(group), episodes=len({r['run'] for r in group}),
            total_seconds=seconds, total_energy_j=energy, mean_power_w=energy/seconds,
            mean_seconds=seconds/len(group),
            mean_request_energy_j=energy/len(group) if key[0] != 'Continue' else None,
            median_seconds=statistics.median(r['seconds'] for r in group),
            sources=sorted({r['run'] for r in group})))
    return cells


def extract(run):
    run = Path(run)
    config = json.loads((run/'run.json').read_text())['config_snapshot']
    record = json.loads((run/'summary.json').read_text())
    if record.get('status') != 'completed' or not config.get('m4_lean_continue'):
        raise ValueError('Require completed lean-capture episode: '+str(run))
    if config['H'] != 50 or config['N'] != 5 or not config['state_enabled']:
        raise ValueError('Unexpected model configuration')
    for name in ('system_audit.json', 'm4_clock_audit.json'):
        if not json.loads((run/name).read_text())['passed']:
            raise ValueError('Failed audit: '+str(run/name))
    episode = record['episode']
    requests = json.loads(request_path(run).read_text())
    if any(q.get('m3_fast_choice',{}).get('mode','Full')!='Full' for q in requests):
        raise ValueError('Full/Continue cost extraction cannot include Fast requests')
    clocks = []
    lines = (run/'rtc/runtime_clocks.jsonl').read_text().splitlines()
    if 'RESTORED' not in lines:
        raise ValueError('Clock restoration not confirmed')
    for line in lines:
        try:
            clocks.append(json.loads(line))
        except ValueError:
            pass
    readings = [PowerReading(x['timestamp_monotonic'], x['power_mw'], x.get('error'))
                for x in episode['including_drain_power']['raw_readings']]
    def energy(a, b):
        p = _integrate_report(readings, a, b, .3)
        if p.total_energy_mj is None or p.missing_count:
            raise ValueError('Incomplete power window')
        return p.total_energy_mj/1000
    ready = [r['ready_monotonic'] for r in requests]
    ledger = build_ledger(episode, requests)
    rows = []
    for window in ledger['windows']:
        if window['activity'] == 'Terminal LOW tail':
            continue
        a, b = window['start_monotonic'], window['stop_monotonic']
        if b <= a:
            continue
        if window['activity'] == 'Continue':
            for left, right, profile, age in continue_segments(a, b, clocks, ready):
                prior_full=max((q for q in requests if q['ready_monotonic']<=left+1e-9),
                               key=lambda q:q['ready_monotonic'], default=None)
                rows.append(dict(mode='Continue', profile=profile, from_profile=profile,
                    after_full_profile=prior_full.get('m4_compute_profile','baseline') if prior_full else 'none',
                    history=age, seconds=right-left, energy_j=energy(left, right),
                    start_monotonic=left, stop_monotonic=right, run=str(run)))
        else:
            q = next(q for q in requests if q['prediction_id'] == window['prediction_id'])
            changes = [(e['monotonic'], e['profile']) if e['status'] == 'READY' else
                       (e['observed_match_monotonic'], e['profile']) for e in clocks
                       if e.get('status') in ('READY', 'APPLIED')]
            origin_profile = max((x for x in changes if x[0] <= a), key=lambda x: x[0])[1]
            quiet_origin = max((t for t in ready if t < a), default=a)
            rows.append(dict(mode='Bootstrap' if q['prediction_id'] == 0 else 'Full',
                profile=q.get('m4_compute_profile', 'baseline'), from_profile=origin_profile,
                history='bootstrap' if q['prediction_id'] == 0 else history(a-quiet_origin),
                seconds=b-a, energy_j=energy(a, b), prediction_id=q['prediction_id'],
                observation_to_ready_seconds=q['ready_monotonic']-q['observation_monotonic'],
                start_monotonic=a, stop_monotonic=b, run=str(run)))
    expected = episode['terminal_low']['task_plus_inference_power']['total_energy_mj']/1000
    difference = sum(r['energy_j'] for r in rows)-expected
    if abs(difference) > 1e-6:
        raise ValueError('Full/Continue partition does not equal total task and drain energy')
    transitions = []
    previous = 'baseline'
    endpoint = episode['terminal_low']['start_monotonic']
    for event in clocks:
        if event.get('status') == 'APPLIED':
            if event['command_monotonic'] < endpoint:
                transitions.append(dict(from_profile=previous, profile=event['profile'],
                    milliseconds=1000*(event['observed_match_monotonic']-event['command_monotonic']),
                    switch_window_energy_j=energy(event['command_monotonic'],event['observed_match_monotonic']),
                    during_task=event['command_monotonic']<episode['stop_monotonic'],
                    run=str(run), separate_energy_j=None,
                    energy_scope='Already included in Full service or Continue; do not charge twice'))
            previous = event['profile']
    task = dict(run=str(run), q=config['candidate_q'][0], suite=config.get('rtc_suite'),
        task_id=episode['task_id'], outcome=episode['status'], steps=episode['steps'],
        seconds=episode['real_seconds'], task_plus_inference_j=expected,
        underflow_steps=len(episode['underflow_steps']), full_requests=len(requests),
        partition_difference_j=difference, startup_floor_ms=config.get('m3_start_floor_ms', 0))
    return rows, transitions, task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-runs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    sources = []
    for source in args.source_runs:
        data = json.loads((source/'summary.json').read_text())
        if data.get('status') != 'completed':
            raise ValueError('Incomplete source: '+str(source))
        children = [Path(e['run']) for e in data['episodes']] if 'episodes' in data else [source]
        for child in children:
            config = json.loads((child/'run.json').read_text())['config_snapshot']
            if config.get('m4_lean_continue') and child not in sources:
                sources.append(child)
    rows, transitions, tasks = [], [], []
    for source in sources:
        observed, switches, task = extract(source)
        rows.extend(observed); transitions.extend(switches); tasks.append(task)
    if not rows:
        raise ValueError('No current lean-capture data')
    cells = aggregate(rows)
    full_profiles=sorted({c['profile'] for c in cells if c['mode']=='Full'})
    hold_profiles=sorted({c['profile'] for c in cells if c['mode']=='Continue'})
    config = load_config(repo_root=ROOT)
    run = create_run(dict(config, cost_source_runs=list(map(str, args.source_runs))),
                     'M4', 'lean-cost-refresh', repo_root=ROOT)
    result = dict(status='completed', episodes=tasks, cells=cells, intervals=rows, transitions=transitions,
        coverage=dict(full_profiles=full_profiles, continue_profiles=hold_profiles,
            missing_full_profiles=sorted({'baseline','gpu_emc'}-set(full_profiles)),
            missing_continue_profiles=sorted({'baseline','gpu_emc','deep_idle'}-set(hold_profiles))),
        scope='Absolute module energy by time activity in real-async lean-capture rollouts; concurrent control/simulation included',
        switch_energy='Included in observed service/Continue windows; no separate switch-energy estimate',
        latency_scope='Observed complete service to commit/ready; small-sample means, not a new tail-latency calibration',
        wait_scope='History-dependent transient power during execution; not a steady idle or railgate claim',
        response_tail='Continue is conditioned on preceding Full frequency as well as elapsed quiet time; power-response tails remain in the measured whole-window sum',
        sources=list(map(str, args.source_runs)), runtime_defaults_changed=False)
    _write_json_atomically(run/'cost_table.json', result)
    for name, values in (('cost_table', cells), ('episode_rows', tasks), ('switch_timings', transitions)):
        with (run/(name+'.csv')).open('w') as out:
            if values:
                fields=list(values[0]); writer=csv.DictWriter(out, fields, lineterminator='\n')
                writer.writeheader(); writer.writerows(values)
    _write_json_atomically(run/'summary.json', dict(status='completed', sources=result['sources'],
        episode_count=len(tasks), cells=cells, result=str(run/'cost_table.json')))
    meta=json.loads((run/'run.json').read_text());meta['status']='completed';_write_json_atomically(run/'run.json',meta)
    print('cost_run='+str(run), flush=True)
    for cell in cells:
        print(json.dumps(cell, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
