#!/usr/bin/env python3
"""Bounded single-load async diagnostic; never resumes a formal run.

All raw records remain unchanged in shape; cleanup happens outside episode time.
"""

import argparse
import fcntl
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base
from scripts import run_async_q as runner
from scripts import probe_delay_buffer as probe


def memory_snapshot(torch):
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'SwapFree'):
            fields[key + '_kib'] = int(value.split()[0])
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            fields['rss_kib'] = int(line.split()[1])
    fields['cuda_allocated_bytes'] = torch.cuda.memory_allocated()
    fields['cuda_reserved_bytes'] = torch.cuda.memory_reserved()
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-ids', type=int, nargs='+', default=[0, 10, 20])
    parser.add_argument('--q', type=int, choices=(5, 8, 10, 12, 15), default=8)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--after-run', type=Path,
                        help='Wait for this existing run to complete before using GPU')
    args = parser.parse_args()
    if not 1 <= args.repeat <= 3 or not args.task_ids or any(not 0 <= t < 30 for t in args.task_ids):
        parser.error('Use task IDs 0..29, repeat 1..3')
    if args.after_run:
        deadline = time.monotonic() + 3600
        while True:
            state = json.loads((args.after_run / 'summary.json').read_text())['status']
            if state == 'completed':
                break
            if state != 'running' or time.monotonic() > deadline:
                raise RuntimeError(f'Prior run not completed: {state}; no diagnostic started')
            time.sleep(5)
    config = load_config(repo_root=ROOT)
    config.update(execution_mode='asynchronous_model_reuse_diagnostic', candidate_q=[args.q])
    runner.configure_handoff(config)
    with (Path(config['output_root']) / '.async_gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another GPU evaluation is active; do not run concurrently')
        run = create_run(config, 'M1', 'model-reuse-diagnostic', repo_root=ROOT)
        print('run_dir=' + str(run), flush=True)
        summary = {'status': 'preparing', 'config': config, 'episodes': [], 'memory': [],
                   'expected_episodes': len(args.task_ids) * args.repeat,
                   'note': 'cleanup diagnostic only; never merge into formal success statistics'}
        source = run / 'source'
        for name in ['scripts/probe_model_reuse.py',
                     'scripts/run_async_q.py', 'scripts/probe_delay_buffer.py', 'scripts/eval_q.py']:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        env = None
        try:
            torch, _ = base._check_cuda_and_native(config)
            # Deliberately do NOT apply base._patch_egl_cleanup(): native cleanup
            # is also needed on hard resets, not just at the final env.close().
            # This remains isolated until model+render coexistence is verified.
            summary['egl_cleanup'] = 'native destructors; no suppression patch'
            import numpy as np
            from perceptbudget.libero_input import load_libero_norm_stats
            stats = load_libero_norm_stats(config['norm_stats_path'])
            tasks = base._load_tasks(args.task_ids, config['screen']['suites'])
            loaded = time.monotonic()
            model = base._load_and_check_model(config, torch)
            summary['model_load_seconds'] = time.monotonic() - loaded
            summary['model_load_count'] = 1
            _, env_class = base._libero_api()
            summary['status'] = 'running'
            base._write_json_atomic(run / 'summary.json', summary)
            for repeat in range(args.repeat):
                for task in tasks:
                    before = memory_snapshot(torch)
                    if before['MemAvailable_kib'] < 1024 * 1024:
                        raise RuntimeError('Less than 1 GiB available; stop before environment allocation')
                    env = base._make_env(env_class, task)
                    obs, _ = base._prepare_env_episode(env, task, np)
                    base._infer_actions(model, base._make_snapshot(obs, stats, config['state_enabled']),
                                        task['instruction'], config['H'], {})
                    del obs
                    folder = run / f'trial{repeat+1}-task{task["task_id"]:02d}'
                    folder.mkdir()
                    with (folder / 'trace.jsonl').open('w') as stream:
                        episode = probe.run_episode(env, task, args.q, model, stats, config,
                            np, torch, folder, stream, **runner.handoff_options(config))
                    runner.validate_episode(episode, config['H'])
                    episode.update(artifact_dir=folder.name, suite=task['suite'], repeat=repeat+1)
                    summary['episodes'].append(episode)
                    env.close()
                    env = None
                    gc.collect()
                    torch.cuda.synchronize()
                    summary['memory'].append({'task_id': task['task_id'], 'repeat': repeat+1,
                                               'before': before, 'after_cleanup': memory_snapshot(torch)})
                    base._write_json_atomic(run / 'summary.json', summary)
                    print(f'completed={len(summary["episodes"])}/{summary["expected_episodes"]}', flush=True)
            summary['status'] = 'completed'
        except BaseException as exc:
            summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            # If explicit teardown failed, exit the diagnostic rather than trying
            # to reuse potentially invalid graphics state for another episode.
            base._write_json_atomic(run / 'summary.json', summary)
            base._save_run_status(run, summary['status'])


if __name__ == '__main__':
    main()
