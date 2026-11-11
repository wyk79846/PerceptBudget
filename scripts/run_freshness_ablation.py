"""Re-evaluate the triggered cohort with only critical freshness disabled."""
import argparse
import copy
import csv
import importlib.util
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_paired_seed_campaign import read, write, RECOVERY
from perceptbudget.m4_acceptance import comparison_energy

POINTER = Path('/tmp/perceptbudget-freshness-off-current.json')
LIVE = Path('/tmp/perceptbudget-freshness-off-progress.txt')


def prepare(source):
    original = read(source / 'campaign_plan.json')
    saved = read(source / 'results.json')
    if saved['status'] != 'completed':
        raise RuntimeError('Wait for the original campaign to complete')
    cohort = read(source / 'critical_freshness_full_cohort.json')['cases']
    selected = {(r['seed'], r['suite'], r['task_id']) for r in cohort}
    seeds = [s for s in original['seeds'] if any(k[0] == s for k in selected)]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = source.parent / f'{stamp}-freshness-off-triggered{len(selected)}'
    root.mkdir()
    signals = copy.deepcopy(read(source / f'seed{seeds[0]}/candidate37/action_signals.json'))
    removed = signals.pop('critical_freshness_max_age_steps')
    write(root / 'action_signals.json', signals)
    commands, tasks = {}, {}
    for seed in seeds:
        folder = root / f'seed{seed}/off'
        folder.mkdir(parents=True)
        indices = [i for i, (s, t) in enumerate(original['tasks']) if (seed, s, t) in selected]
        selected_tasks = [original['tasks'][i] for i in indices]
        old = source / f'seed{seed}/candidate37'
        wrapper = (old / 'candidate.py').read_text().replace('module.TOTAL=40', f'module.TOTAL={len(indices)}')
        wrapper = wrapper.replace(
            "if __name__=='__main__':raise SystemExit(module.main())",
            """# The original resume validates the full40 manifest; retain this selected cohort.
import sys
sys.path.insert(0, '/home/wyk/workspace/PerceptBudget')
from perceptbudget import m4_acceptance as acceptance
from unittest.mock import patch
selected={tuple(t) for t in __import__('json').loads((module.HERE/'plan.json').read_text())['tasks']}
original_resume=acceptance.resume_full
def scoped_resume(*args, **kwargs):
    result=original_resume(*args, **kwargs)
    result['remaining_tasks']=[t for t in result['remaining_tasks'] if tuple(t) in selected]
    if any((r['suite'],r['task_id']) not in selected for r in result['episodes']):
        raise ValueError('Resume contains an episode outside the selected cohort')
    return result
if __name__=='__main__':
    with patch.object(acceptance,'resume_full',scoped_resume):
        raise SystemExit(module.main())""")
        (folder / 'candidate.py').write_text(wrapper)
        command = list(original['commands'][str(seed)]['candidate37'])
        command[2] = str(folder / 'candidate.py')
        command[command.index('--action-signals') + 1] = str(root / 'action_signals.json')
        command.extend(['--task-indices', *map(str, indices)])
        metadata = copy.deepcopy(read(old / 'plan.json'))
        metadata.update(tasks=selected_tasks, episodes=len(indices), command=command, root=str(root),
                        evaluation_scope='Paired triggered-cohort ablation: remove critical freshness only; '
                        'keep gripper-lift feedback, images, RTC, Full/Fast, clocks, seed and init unchanged.')
        write(folder / 'plan.json', metadata)
        commands[str(seed)] = command
        tasks[str(seed)] = selected_tasks
    references = [r for r in saved['rows'] if (r['seed'], r['suite'], r['task_id']) in selected]
    write(root / 'references.json', references)
    write(root / 'plan.json', dict(root=str(root), source=str(source), seeds=seeds, tasks=tasks,
        total=len(selected), commands=commands, init_state_index=1, condition='freshness_off',
        removed_setting={'critical_freshness_max_age_steps': removed},
        unchanged_gripper_lift_feedback=signals['gripper_lift_feedback'],
        cohort_selection='All original candidate episodes with actual critical_freshness requests, including successes and failures'))
    write(POINTER, dict(run=str(root), progress=str(root / 'progress.md'), live_text=str(LIVE)))
    return root


class Campaign:
    def __init__(self, root):
        self.root, self.plan = root, read(root / 'plan.json')
        self.references = read(root / 'references.json')
        self.cache, self.active, self.last_count = {}, None, -1
        sys.path.insert(0, str(RECOVERY))
        spec = importlib.util.spec_from_file_location('freshness_resource_recovery', RECOVERY / 'run_campaign.py')
        self.recovery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.recovery)
        self.recovery.progress = self.progress
        self.recovery.condition_complete = self.complete

    def complete(self, folder):
        parent = self.recovery.parent_for(folder)
        if not parent or not (parent / 'summary.json').exists():
            return False
        expected = {tuple(t) for t in read(folder / 'plan.json')['tasks']}
        record = read(parent / 'recording.json')
        rows = record.get('episodes', [])
        return (read(parent / 'summary.json').get('status') == 'completed'
                and len(rows) == len(expected)
                and {(r['suite'], r['task_id']) for r in rows} == expected)

    def progress(self, status='running', error=None):
        rows, states = [], {}
        for seed in self.plan['seeds']:
            folder = self.root / f'seed{seed}/off'
            try:
                state = read(folder / 'progress.json')
            except (OSError, ValueError):
                state = dict(completed=0, successes=0, rows=[])
            states[str(seed)] = {k: v for k, v in state.items() if k != 'rows'}
            for item in state['rows']:
                run = item['run']
                if run not in self.cache:
                    try:
                        summary = read(Path(run) / 'summary.json')
                        episode = summary['episode']
                        self.cache[run] = dict(seed=seed, method='freshness_off', **item,
                            **comparison_energy(episode), audit_passed=summary.get('audit_passed'),
                            mean_refresh_steps=(sum(episode['actual_refresh_intervals_steps']) /
                                len(episode['actual_refresh_intervals_steps']))
                                if episode.get('actual_refresh_intervals_steps') else None)
                    except (OSError, KeyError, ValueError, TypeError):
                        continue
                rows.append(self.cache[run])
        keys = {(r['seed'], r['suite'], r['task_id']) for r in rows}
        groups = {}
        for method in ('freshness_off', 'candidate37', 'q10'):
            subset = rows if method == 'freshness_off' else [r for r in self.references
                if r['method'] == method and (r['seed'], r['suite'], r['task_id']) in keys]
            groups[method] = dict(completed=len(subset), successes=sum(r['outcome'] == 'success' for r in subset),
                                  total_energy_j=sum(r['total_energy_j'] for r in subset))
        count = sum(s.get('completed', 0) for s in states.values())
        result = dict(status=status, updated=datetime.now().astimezone().isoformat(timespec='seconds'),
            completed=count, total=self.plan['total'], active_seed=self.active, states=states,
            groups=groups, error=error)
        write(self.root / 'progress.json', result)
        lines = ['# 关键动作前补刷新：关闭开关的配对补测', '',
            f'状态{status}；完成{count}/{self.plan["total"]}；当前seed={self.active}。', '',
            '仅关闭critical_freshness；保留夹爪反馈和其他37版设置。历史开启组和q10直接复用。', '',
            '|相同已完成样本|成功|能量J|', '|---|---:|---:|']
        terminal = [f'{result["updated"]}  {status}  {count}/{self.plan["total"]}  seed={self.active}']
        for method, group in groups.items():
            lines.append(f'|{method}|{group["successes"]}/{group["completed"]}|{group["total_energy_j"]:.2f}|')
            terminal.append(f'{method}: 成功 {group["successes"]}/{group["completed"]}  {group["total_energy_j"]:.2f} J')
        lines.extend(['', '|seed|完成|成功|', '|---:|---:|---:|'])
        for seed in self.plan['seeds']:
            state = states[str(seed)]
            line = f'{seed}: {state.get("completed", 0)}/{len(self.plan["tasks"][str(seed)])} 成功{state.get("successes", 0)}'
            terminal.append(line)
            lines.append(f'|{seed}|{state.get("completed", 0)}/{len(self.plan["tasks"][str(seed)])}|{state.get("successes", 0)}|')
        for r in [r for r in rows if r['seed'] == self.active][-6:]:
            terminal.append(f'{r["suite"]} task{r["task_id"]}: {r["outcome"]}, {r["steps"]}步')
        if error:
            lines.extend(['', '错误：' + error]); terminal.append(error)
        (self.root / 'progress.md').write_text('\n'.join(lines) + '\n')
        LIVE.write_text('\n'.join(terminal) + '\n')
        if len(rows) != self.last_count or status in ('completed', 'invalid'):
            write(self.root / 'results.json', dict(**result, rows=rows, references=self.references))
            stream = io.StringIO()
            fields = ('seed', 'method', 'suite', 'task_id', 'outcome', 'steps', 'wall_seconds', 'total_energy_j', 'run')
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
            (self.root / 'episodes.csv').write_text(stream.getvalue())
            self.last_count = len(rows)

    def run(self):
        try:
            for seed in self.plan['seeds']:
                self.active = seed
                self.progress()
                self.recovery.run_condition(seed, 'off', folder=self.root / f'seed{seed}/off',
                    command_override=self.plan['commands'][str(seed)])
            self.active = None
            self.progress('completed')
            (self.root / 'results.md').write_text((self.root / 'progress.md').read_text())
            check = subprocess.run(['sudo', '-n', '/usr/local/sbin/perceptbudget-clocks', 'inspect'], capture_output=True, text=True)
            write(self.root / 'clock_after.json', dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr))
        except Exception as exc:
            self.progress('invalid', f'{type(exc).__name__}: {exc}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    root = args.run_dir or prepare(args.source)
    if args.prepare:
        print(root)
    else:
        Campaign(root).run()


if __name__ == '__main__':
    main()
