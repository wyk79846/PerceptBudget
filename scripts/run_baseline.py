#!/usr/bin/env python3
"""Run one task with the sole frozen RTC baseline; --dry-run does not use GPU."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.baseline import QS, SUITES, command
from perceptbudget.experiment import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, default='libero_spatial')
    parser.add_argument('--task-id', type=int, choices=range(10), default=0)
    parser.add_argument('--q', type=int, choices=QS, default=5)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(repo_root=ROOT)
    cmd = command(config, ROOT, suite=args.suite, task_id=args.task_id, q=args.q)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
