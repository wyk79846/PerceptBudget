#!/usr/bin/env python3
"""Finish only q10/12/15, then summarize them with existing q5/q8 records."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/home/wyk/workspace/任务产出/perceptbudget/M1')
BASELINES = [RESULTS / '20261016T053343.959035Z-async-q5-baseline-a1f27927',
             RESULTS / '20261016T035800.217970Z-async-q8-screen-3650b7d9']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Continue the same remaining-q run without repeating completed tasks')
    args = parser.parse_args()
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_async_q.py')]
    if args.resume:
        command += ['--resume', str(args.resume.resolve())]
    else:
        command += ['--q', '10', '12', '15', '--model-lifetime', 'session',
                    '--refresh-anchor', 'request', '--purpose', 'async-remaining']
    remaining = None
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
        for line in child.stdout:
            print(line, end='', flush=True)
            if line.startswith('run_dir='):
                remaining = Path(line.strip().split('=', 1)[1])
        code = child.wait()
    if code:
        return code  # Preserve failures; never silently rerun completed episodes.
    if remaining is None:
        raise RuntimeError('Evaluator returned no run directory')
    return subprocess.call([sys.executable, '-u', str(ROOT / 'scripts/compare_async_runs.py'),
                            '--require-complete', *map(str, BASELINES), str(remaining)], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
