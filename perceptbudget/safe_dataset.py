"""Fixed training-suite quotas; validation/test trajectories never move."""
import random


TRAINING_QUOTAS = {'libero_spatial': 40, 'libero_object': 40,
                   'libero_goal': 40, 'libero_10': 120}


def select_training240(episodes, candidates, *, seed=37):
    rng = random.Random(seed)
    existing = {e['source_run'] for e in episodes}
    retained = []
    for suite, quota in TRAINING_QUOTAS.items():
        training = [e for e in episodes if e['safe_split'] == 'train' and e['suite'] == suite]
        if any(e['task_id'] >= 7 for e in training):
            raise ValueError('Held-out tasks cannot enter training')
        if suite == 'libero_10':
            if len(training) > quota:
                raise ValueError('Expected to supplement Long training')
            retained.extend(training)
            continue
        failures = [e for e in training if e['failure']]
        if len(failures) > quota or len(training) < quota:
            raise ValueError('Cannot meet short-suite quota while retaining failures')
        selected = list(failures)
        # Retain every task identity before filling the remaining success slots.
        successes = [e for e in training if not e['failure']]
        rng.shuffle(successes)
        for task in sorted({e['task_id'] for e in training}):
            if not any(e['task_id'] == task for e in selected):
                selected.append(next(e for e in successes if e['task_id'] == task))
        used = {e['source_run'] for e in selected}
        selected.extend(e for e in successes if e['source_run'] not in used)
        retained.extend(selected[:quota])
    current_long = [e for e in retained if e['suite'] == 'libero_10']
    available = {}
    for row in candidates:
        if (row.get('method') == 'candidate37' and row['suite'] == 'libero_10'
                and row['task_id'] < 7 and row['run'] not in existing):
            available[row['run']] = row
    added = []
    for task in range(7):
        # Uniform final task counts; assign the one remainder slot by task order.
        target = 120 // 7 + int(task < 120 % 7)
        need = target - sum(e['task_id'] == task for e in current_long)
        options = sorted((r for r in available.values() if r['task_id'] == task),
                         key=lambda r: r['run'])
        rng.shuffle(options)
        if need < 0 or len(options) < need:
            raise ValueError(f'Insufficient unused Long task{task} trajectories: need{need}, have{len(options)}')
        added.extend(options[:need])
    retained.sort(key=lambda e: e['episode'])
    if len(retained) + len(added) != 240:
        raise ValueError('Training quota did not total240')
    return retained, added
