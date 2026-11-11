"""Run the frozen 37/40 candidate and q10, one complete pair per random seed.

Use --prepare to create the fixed seed list, then --run-dir PATH in a user
systemd service. Existing completed episodes survive resource-only retries.
"""
import argparse
import copy
import csv
import importlib.util
import io
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT.parent / '任务产出/perceptbudget/M4'
CANDIDATE_SOURCE = OUTPUT / '20261110T085132Z-v4-gripper-feedback-seed550949-all40'
Q10_SOURCE = OUTPUT / '20261110T073541Z-freshness-q10-seed550949-pair80/q10_run'
RECOVERY = OUTPUT / '20261106T180408.715677Z-critical-fast-q20-fresh5-seeds'
METHODS = ('candidate37', 'q10')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
POINTER = Path('/tmp/perceptbudget-25seeds-current.json')
LIVE_TEXT = Path('/tmp/perceptbudget-25seeds-progress.txt')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare(count):
    used = {7, 17, 27, 37, 47, 57, 67, 79, 89, 97, 109, 550949, 618022}
    for pattern in ('*/plan.json', '*/campaign_plan.json', '*/session_plan.json'):
        for path in OUTPUT.glob(pattern):
            try:
                value = read(path)
                if isinstance(value.get('seed'), int):
                    used.add(value['seed'])
                used.update(s for s in value.get('seeds', []) if isinstance(s, int))
            except (OSError, ValueError, TypeError):
                pass
    rng = random.SystemRandom()
    seeds = []
    while len(seeds) < count:
        seed = rng.randrange(1000, 1000000)
        if seed not in used and seed not in seeds:
            seeds.append(seed)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = OUTPUT / (stamp + '-candidate37-q10-25seeds')
    root.mkdir()
    source_plan = read(CANDIDATE_SOURCE / 'plan.json')
    q10_plan = read(Q10_SOURCE / 'plan.json')
    commands = {}
    scope = ('25 randomly drawn seeds, frozen before evaluation; init1; all40 '
             'LIBERO tasks; candidate37 then q10 for each seed; no tuning. '
             'Original live images and real asynchronous execution retained.')
    for seed in seeds:
        commands[str(seed)] = {}
        for method in METHODS:
            source_folder = (CANDIDATE_SOURCE / 'candidate' if method == 'candidate37'
                             else Q10_SOURCE / 'q10')
            folder = root / f'seed{seed}' / method
            folder.mkdir(parents=True)
            wrapper = (source_folder / 'candidate.py').read_text()
            wrapper = wrapper.replace('module.SEED=550949', f'module.SEED={seed}')
            (folder / 'candidate.py').write_text(wrapper)
            command = list(source_plan['commands']['candidate'] if method == 'candidate37'
                           else q10_plan['command'])
            command[2] = str(folder / 'candidate.py')
            if method == 'candidate37':
                write(folder / 'action_signals.json', read(source_folder / 'action_signals.json'))
                command[command.index('--action-signals') + 1] = str(folder / 'action_signals.json')
            metadata = copy.deepcopy(read(source_folder / 'plan.json'))
            metadata.update(seed=seed, init_state_index=1, episodes=40, command=command,
                            root=str(root), evaluation_scope=scope,
                            label=method, source_run=str(source_folder.parent))
            write(folder / 'plan.json', metadata)
            commands[str(seed)][method] = command
    plan = dict(root=str(root), created_utc=stamp, seeds=seeds, init_state_index=1,
                methods=list(METHODS), episodes_per_method_seed=40, total=count * 80,
                tasks=read(CANDIDATE_SOURCE / 'candidate/plan.json')['tasks'],
                order='for each seed: candidate37 all40, then q10 all40',
                candidate_source=str(CANDIDATE_SOURCE), q10_source=str(Q10_SOURCE),
                recovery_library=str(RECOVERY), commands=commands, evaluation_scope=scope)
    write(root / 'campaign_plan.json', plan)
    write(POINTER, dict(run=str(root), script=str(Path(__file__).resolve()),
                        progress=str(root / 'progress.md'), live_text=str(LIVE_TEXT)))
    return root


class Campaign:
    def __init__(self, root):
        self.root = root
        self.plan = read(root / 'campaign_plan.json')
        self.active_seed = None
        self.active_method = None
        self.row_cache = {}
        self.last_result_count = -1
        sys.path.insert(0, str(PROJECT))
        sys.path.insert(0, str(RECOVERY))
        spec = importlib.util.spec_from_file_location('paired_resource_recovery', RECOVERY / 'run_campaign.py')
        self.recovery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.recovery)
        self.recovery.progress = self.progress
        from perceptbudget.m4_acceptance import comparison_energy, request_path
        self.energy = comparison_energy
        self.request_path = request_path

    def collect(self):
        states = {}
        rows = []
        for seed in self.plan['seeds']:
            states[str(seed)] = {}
            for method in METHODS:
                folder = self.root / f'seed{seed}' / method
                try:
                    saved = read(folder / 'progress.json')
                except (OSError, ValueError):
                    saved = dict(status='not_started', completed=0, successes=0, total=40, rows=[])
                states[str(seed)][method] = {k: v for k, v in saved.items() if k != 'rows'}
                for item in saved.get('rows', []):
                    key = item.get('run')
                    if key not in self.row_cache:
                        try:
                            summary = read(Path(key) / 'summary.json')
                            episode = summary['episode']
                            energy = self.energy(episode)
                            requests = read(self.request_path(key))
                            row = dict(seed=seed, method=method, **item, **energy,
                                       requests=len(requests),
                                       refresh_intervals=episode.get('actual_refresh_intervals_steps', []),
                                       underflow_steps=episode.get('underflow_steps'),
                                       audit_passed=summary.get('audit_passed'))
                            self.row_cache[key] = row
                        except (OSError, ValueError, KeyError, TypeError):
                            continue
                    rows.append(self.row_cache[key])
        return states, rows

    @staticmethod
    def aggregate(rows):
        intervals = [i for r in rows for i in r.get('refresh_intervals', [])]
        return dict(completed=len(rows), successes=sum(r['outcome'] == 'success' for r in rows),
                    total_energy_j=sum(r['total_energy_j'] for r in rows),
                    total_task_seconds=sum(r.get('wall_seconds') or 0 for r in rows),
                    mean_refresh_steps=sum(intervals) / len(intervals) if intervals else None,
                    requests=sum(r['requests'] for r in rows),
                    per_suite={s: dict(completed=sum(r['suite'] == s for r in rows),
                                       successes=sum(r['suite'] == s and r['outcome'] == 'success' for r in rows))
                               for s in SUITES})

    def comparisons(self, rows):
        indexed = {(r['seed'], r['suite'], r['task_id'], r['method']): r for r in rows}
        paired = []
        for seed in self.plan['seeds']:
            for suite, task in self.plan['tasks']:
                a = indexed.get((seed, suite, task, 'candidate37'))
                b = indexed.get((seed, suite, task, 'q10'))
                if a is not None and b is not None:
                    paired.append((a, b))
        def metrics(pairs):
            a = sum(x['total_energy_j'] for x, _ in pairs)
            b = sum(y['total_energy_j'] for _, y in pairs)
            return dict(paired_tasks=len(pairs), candidate_energy_j=a, q10_energy_j=b,
                        energy_saving_percent=100 * (1 - a / b) if b else None,
                        candidate_successes=sum(x['outcome'] == 'success' for x, _ in pairs),
                        q10_successes=sum(y['outcome'] == 'success' for _, y in pairs))
        return dict(all_outcomes=metrics(paired),
                    common_success=metrics([(a, b) for a, b in paired
                                            if a['outcome'] == b['outcome'] == 'success']))

    def progress(self, status='running', error=None):
        states, rows = self.collect()
        groups = {method: self.aggregate([r for r in rows if r['method'] == method]) for method in METHODS}
        done = sum((x.get('completed') or 0) for pair in states.values() for x in pair.values())
        comparison = self.comparisons(rows)
        updated = datetime.now().astimezone().isoformat(timespec='seconds')
        result = dict(status=status, updated=updated, completed=done, total=self.plan['total'],
                      active_seed=self.active_seed, active_method=self.active_method,
                      states=states, groups=groups, comparison=comparison, error=error)
        write(self.root / 'progress.json', result)
        heading = ['# 37/40版与q10：25随机种子实时评测', '',
                   f'更新：{updated}；状态：{status}；完成：{done}/{self.plan["total"]}。',
                   f'当前 seed={self.active_seed}，方法={self.active_method}；init=1。', '',
                   '每个种子先候选40任务，再q10的40任务；参数固定，全部结果保留。', '']
        for method in METHODS:
            group = groups[method]
            n, success = group['completed'], group['successes']
            rate = f'{100 * success / n:.2f}%' if n else '--'
            heading.append(f'{method}：成功 {success}/{n}（{rate}），已完成能量 {group["total_energy_j"]:.1f} J。')
        paired = comparison['all_outcomes']
        saving = paired['energy_saving_percent']
        if saving is not None:
            heading.extend(['', f'已配对{paired["paired_tasks"]}任务，含失败能量节省 {saving:.2f}%。'])
        lines = heading + ['', '|顺序|seed|候选 完成/成功|q10 完成/成功|', '|---:|---:|---:|---:|']
        terminal = [f'更新时间 {updated}  状态 {status}  完成 {done}/{self.plan["total"]}',
                    f'当前 seed={self.active_seed} / {self.active_method}    init=1',
                    f'候选成功 {groups["candidate37"]["successes"]}/{groups["candidate37"]["completed"]}'
                    f'    q10成功 {groups["q10"]["successes"]}/{groups["q10"]["completed"]}',
                    '序号  seed     候选(完成/成功)  q10(完成/成功)']
        for i, seed in enumerate(self.plan['seeds'], 1):
            a, b = (states[str(seed)][m] for m in METHODS)
            ad, ass = a.get('completed', 0) or 0, a.get('successes', 0) or 0
            bd, bs = b.get('completed', 0) or 0, b.get('successes', 0) or 0
            lines.append(f'|{i}|{seed}|{ad}/40，成功{ass}|{bd}/40，成功{bs}|')
            terminal.append(f'{i:2d}  {seed:6d}   {ad:2d}/40 成功{ass:2d}   {bd:2d}/40 成功{bs:2d}')
        if self.active_seed is not None:
            state = states[str(self.active_seed)][self.active_method]
            count = state.get('completed') or 0
            if count < 40:
                suite, task = self.plan['tasks'][count]
                terminal.append(f'当前/下一任务：{suite} task{task}；已完成{count}/40')
            terminal.append('当前集合：' + json.dumps(state.get('per_suite', {}), ensure_ascii=False))
        if saving is not None:
            terminal.append(f'配对{paired["paired_tasks"]}任务（含失败）节能 {saving:.2f}%')
        latest = [r for r in rows if r['seed'] == self.active_seed and r['method'] == self.active_method][-8:]
        lines.extend(['', '|当前组最近任务|结果|步数|能量J|', '|---|---|---:|---:|'])
        for row in latest:
            label = f'{row["suite"]} task{row["task_id"]}'
            lines.append(f'|{label}|{row["outcome"]}|{row["steps"]}|{row["total_energy_j"]:.2f}|')
            terminal.append(f'{label}: {row["outcome"]}, {row["steps"]}步, {row["total_energy_j"]:.2f} J')
        if error:
            lines.extend(['', '错误：' + error])
            terminal.append('错误：' + error)
        lines.extend(['', f'[控制器日志]({self.root}/controller.log)',
                      f'[逐任务数据]({self.root}/episodes.csv)', ''])
        (self.root / 'progress.md').write_text('\n'.join(lines))
        LIVE_TEXT.write_text('\n'.join(terminal) + '\n')
        if len(rows) != self.last_result_count or status in ('completed', 'invalid'):
            write(self.root / 'results.json', dict(**result, seeds=self.plan['seeds'], rows=rows))
            fields = ('seed', 'method', 'suite', 'task_id', 'outcome', 'steps',
                      'wall_seconds', 'total_energy_j', 'requests', 'underflow_steps', 'audit_passed', 'run')
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
            (self.root / 'episodes.csv').write_text(stream.getvalue())
            self.last_result_count = len(rows)

    def run(self):
        self.progress()
        try:
            for seed in self.plan['seeds']:
                self.active_seed = seed
                for method in METHODS:
                    self.active_method = method
                    self.progress()
                    self.recovery.run_condition(seed, method,
                        folder=self.root / f'seed{seed}' / method,
                        command_override=self.plan['commands'][str(seed)][method])
                    self.progress()
                print('SEED_PAIR_COMPLETED=' + str(seed), flush=True)
            capture = subprocess.run(['sudo', '-n', '/usr/local/sbin/perceptbudget-clocks', 'inspect'],
                                     capture_output=True, text=True)
            write(self.root / 'clock_after.json', dict(returncode=capture.returncode,
                                                       stdout=capture.stdout, stderr=capture.stderr))
            self.active_seed = self.active_method = None
            self.progress('completed')
            (self.root / 'results.md').write_text((self.root / 'progress.md').read_text())
            print('CAMPAIGN_COMPLETED: 25 seeds, 2000 episodes', flush=True)
        except Exception as exc:
            self.progress('invalid', f'{type(exc).__name__}: {exc}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--seed-count', type=int, default=25)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir or prepare(args.seed_count)
    if args.prepare:
        print(root)
        return
    Campaign(root).run()


if __name__ == '__main__':
    main()
