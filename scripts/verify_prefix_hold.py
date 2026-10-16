#!/usr/bin/env python3
"""Read-only audit of one recorded prefix hold diagnostic (no model/GPU)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def verify(run):
    config = json.loads((run / 'run.json').read_text())['config_snapshot']
    summary = json.loads((run / 'summary.json').read_text())
    events = [json.loads(line) for line in (run / 'trace.jsonl').read_text().splitlines()]
    rows = [r for r in events if r['event'] == 'control_step']
    commits = [r for r in events if r['event'] == 'prediction_committed']
    assert len(summary['q_values']) == 1
    q = summary['q_values'][0]
    requests = json.loads((run / f'q{q}_requests.json').read_text())
    pauses = json.loads((run / f'q{q}_late_hold.json').read_text())
    by_id = {r['prediction_id']: r for r in requests}
    by_step = {r['step']: r for r in rows}
    horizon = config['H']
    assert config['prefix_lock']['late_policy'] == 'hold'
    assert summary['task_ids'] == [0]
    assert len(by_step) == len(rows)
    for row in rows:
        assert np.isclose(row['after']['sim_time'] - row['before']['sim_time'], .05)
        if row['late_hold']:
            assert row['action'] == [0.] * 7 and row['source'] is None
        elif row['source'] is not None:
            source = row['source']
            request = by_id[source['prediction_id']]
            merged = source.get('prefix_rtc_merge')
            expected = merged['action'] if merged else request['actions'][source['offset']]
            np.testing.assert_array_equal(row['action'], expected)
            if merged:
                from perceptbudget.prefix_fusion import round_to_bf16
                assert merged['target_step'] == row['step']
                np.testing.assert_array_equal(merged['raw'][6:],
                    request['actions_raw'][source['offset']][6:])
                np.testing.assert_array_equal(merged['new_raw'], request['actions_raw'][source['offset']])
                if merged['source_count'] == 2:
                    weight = np.exp(-.1 * abs(merged['old_offset'] - merged['new_offset']))
                    np.testing.assert_allclose(merged['weight_old'], weight, rtol=1e-14)
                    expected = (np.array(merged['old_raw'])[:6] * (weight / (1 + weight))
                                + np.array(merged['new_raw'])[:6] * (1 / (1 + weight)))
                    np.testing.assert_array_equal(merged['raw'][:6], round_to_bf16(expected))
            assert row['step'] < request['observation_step'] + horizon
    for request in requests:
        assert np.asarray(request['actions_raw']).shape == (horizon, 32)
        if 'prefix_len' not in request:
            continue
        origin, length = request['observation_step'], request['prefix_len']
        assert request['prefix_exact']
        np.testing.assert_array_equal(request['actions_raw'][:length], request['committed_prefix_raw'])
        for i, expected in enumerate(request['committed_prefix_actions']):
            if origin + i in by_step:
                row = by_step[origin + i]
                np.testing.assert_array_equal(row['action'], expected)
                source = row['source']
                merged = source.get('prefix_rtc_merge')
                np.testing.assert_array_equal(request['committed_prefix_raw'][i],
                    merged['raw'] if merged else by_id[source['prediction_id']]['actions_raw'][source['offset']])
    for commit in commits:
        request = by_id[commit['prediction_id']]
        origin, step = request['observation_step'], commit['commit_step']
        assert commit['new_prediction_remaining_interval_steps'] == [step, origin + horizon]
        assert commit['new_prediction_remaining_actions'] == origin + horizon - step
        if 'prefix_len' in request:
            length = request['prefix_len']
            assert step >= origin + length
            alignment = commit['action_alignment']
            assert alignment['source_offset'] == length
            assert alignment['discarded_tail_actions'] == step - origin - length
            if step in by_step and by_step[step]['source'] is not None:
                assert by_step[step]['source']['offset'] == length
    assert pauses['steps'] == sum(r['late_hold'] for r in rows)
    assert pauses['count'] == len(pauses['pauses'])
    if config['prefix_lock'].get('advance_handoff'):
        last_commit = 0
        commit_by_id = {c['prediction_id']: c for c in commits}
        for request in requests[1:]:
            assert request['refresh_anchor'] == 'advance_handoff'
            assert request['observation_step'] == last_commit + q - request['prefix_len']
            assert request['target_handoff_step'] == last_commit + q
            assert request['scheduled_commit_step'] == request['target_handoff_step']
            if request['prediction_id'] in commit_by_id:
                last_commit = commit_by_id[request['prediction_id']]['commit_step']
    return {'verified': True, 'status': summary['status'], 'control_steps': len(rows),
            'q': q,
            'commits': len(commits), 'raw32_requests': len(requests),
            'late_hold_count': pauses['count'], 'late_hold_steps': pauses['steps'],
            'late_hold_wall_seconds': pauses['wall_seconds'],
            'checks': ['zero hold command and simulation ticks', 'executed locked prefixes',
                       'raw32 queue lineage', 'prefix resume offset and tail crop',
                       'original origin+H intervals and action deadlines']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(json.dumps(verify(parser.parse_args().run), ensure_ascii=False, indent=2))
