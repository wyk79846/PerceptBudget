#!/usr/bin/env python3
"""Run one child command, stopping its process group below 768 MiB free RAM.

Usage: python scripts/guarded_run.py -- python scripts/eval_q.py --prepare-only
"""

import os
import signal
import subprocess
import sys
import time


RAM_FLOOR_BYTES = 768 * 1024 * 1024
POLL_SECONDS = 0.1
TERMINATE_GRACE_SECONDS = 2.0


def available_ram_bytes():
    with open("/proc/meminfo", "r", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo 没有 MemAvailable")


def _stop_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "--" or len(args) == 1:
        print(
            "用法：python scripts/guarded_run.py -- <command> [args ...]",
            file=sys.stderr,
        )
        return 2
    command = args[1:]
    try:
        available = available_ram_bytes()
    except (OSError, ValueError, RuntimeError) as exc:
        print("无法读取可用 RAM，未启动子进程：{}".format(exc), file=sys.stderr)
        return 125
    if available < RAM_FLOOR_BYTES:
        print(
            "可用 RAM {:.1f} MiB 低于 768 MiB，未启动子进程".format(
                available / (1024 * 1024)
            ),
            file=sys.stderr,
        )
        return 125

    try:
        child = subprocess.Popen(command, start_new_session=True)
    except OSError as exc:
        print("无法启动子进程：{}".format(exc), file=sys.stderr)
        return 127

    try:
        while child.poll() is None:
            try:
                available = available_ram_bytes()
            except (OSError, ValueError, RuntimeError) as exc:
                print("RAM 监控失效，终止子进程：{}".format(exc), file=sys.stderr)
                _stop_child(child)
                return 125
            if available < RAM_FLOOR_BYTES:
                print(
                    "可用 RAM 降至 {:.1f} MiB；终止本 wrapper 启动的子进程组".format(
                        available / (1024 * 1024)
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                _stop_child(child)
                return 125
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        _stop_child(child)
        return 130
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
