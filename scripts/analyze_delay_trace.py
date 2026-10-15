"""Describe contact, lift, action changes and timing in the fixed pilot trace."""

import argparse
import json
import math
import statistics
from pathlib import Path


def target_contact(scene):
    for contact in scene['contacts']:
        a, b = contact['geom1'] or '', contact['geom2'] or ''
        if contact['distance'] <= 0 and (
            ('finger' in a and 'akita_black_bowl_1_' in b)
            or ('finger' in b and 'akita_black_bowl_1_' in a)
        ):
            return True
    return False


def describe(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'max': None}
    return {'count': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'max': max(values)}


def analyze(run_dir):
    events = [json.loads(line) for line in (run_dir / 'trace.jsonl').open()]
    result = {'kind': 'offline_analysis_no_new_trials',
              'contact_definition': 'post-step sampled target bowl / finger geom contact with distance <= 0',
              'lift_definition': 'target bowl center z > initial z + 0.02m; descriptive threshold, not grasp truth',
              'command_units': 'host output xyz control command, not meters or actual velocity',
              'causal_limit': 'one trial per q; correlations do not isolate cause', 'episodes': []}
    for q in (8, 12):
        rows = [r for r in events if r['event'] == 'control_step' and r['q'] == q]
        requests = json.loads((run_dir / f'q{q}_requests.json').read_text())
        by_step = {r['step']: r for r in rows}
        initial_z = rows[0]['before']['observations']['akita_black_bowl_1_pos'][2]
        rises = [r['after']['observations']['akita_black_bowl_1_pos'][2] - initial_z for r in rows]
        contacts = [r['step'] for r in rows if target_contact(r['after'])]
        lifts = [r['step'] for r, rise in zip(rows, rises) if rise > 0.02]
        boundary, interior, mismatches, gripper_switches = [], [], [], []
        for a, b in zip(rows, rows[1:]):
            if a['source'] is None or b['source'] is None:
                continue  # Underflow handling is not a model chunk boundary.
            change = math.dist(a['action'][:3], b['action'][:3])
            is_boundary = a['source']['prediction_id'] != b['source']['prediction_id']
            (boundary if is_boundary else interior).append(change)
            if a['action'][6] * b['action'][6] < 0:
                gripper_switches.append(b['step'])
        for request in requests[1:]:
            committed = next((e for e in events if e['event'] == 'prediction_committed'
                              and e['q'] == q and e['prediction_id'] == request['prediction_id']), None)
            if committed is None:
                continue
            for offset in range(committed['control_steps_during_inference']):
                actual = by_step[request['observation_step'] + offset]['action']
                mismatches.append(math.dist(actual[:3], request['actions'][offset][:3]))
        result['episodes'].append({
            'q': q, 'first_recorded_finger_target_contact_step': contacts[0] if contacts else None,
            'finger_target_contact_steps': contacts,
            'first_recorded_2cm_lift_step': lifts[0] if lifts else None,
            'target_max_z_rise_m': max(rises),
            'gripper_direction_switch_steps': gripper_switches,
            'underflow_steps': [r['step'] for r in rows if r['source'] is None],
            'boundary_xyz_command_change': describe(boundary),
            'interior_xyz_command_change': describe(interior),
            'boundary_to_interior_mean_change_ratio': (
                statistics.mean(boundary) / statistics.mean(interior)
                if boundary and interior and statistics.mean(interior) else None),
            'discarded_prefix_vs_executed_old_xyz_command_difference': describe(mismatches),
            'env_step_seconds': describe([r['action_end_monotonic'] - r['action_start_monotonic'] for r in rows]),
            'pre_action_seconds': describe([r['action_start_monotonic'] - r['tick_monotonic'] for r in rows]),
            'between_action_end_and_next_tick_seconds': describe([
                b['tick_monotonic'] - a['action_end_monotonic'] for a, b in zip(rows, rows[1:])]),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    output = args.run_dir / 'trace_analysis.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
