#!/usr/bin/env python3
"""Root-only bounded clock lease; model inference remains an ordinary user."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perceptbudget.device_clocks import ClockControl, PROFILES, cycle_trace

STATE = Path('/run/perceptbudget-clocks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect')
    lease = sub.add_parser('lease')
    lease.add_argument('--profile', required=True, choices=PROFILES)
    lease.add_argument('--seconds', type=int, default=120, choices=range(1, 601))
    lease.add_argument('--trace', action='store_true', help='Print actual frequency samples every 0.2s')
    cycles = sub.add_parser('cycles', help='Five fixed static low/high cycles, three seconds per phase')
    cycles.add_argument('--profile', required=True, choices=('baseline', 'gpu_emc', 'gpu_emc_cpu'))
    sub.add_parser('restore')
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('Run this tool with sudo; never run the model as root')
    os.umask(0o077)
    STATE.mkdir(mode=0o700, exist_ok=True)
    if STATE.is_symlink() or STATE.stat().st_uid != 0 or STATE.stat().st_mode & 0o077:
        raise RuntimeError('Unsafe root state directory')
    with (STATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = ClockControl()
        snapshot = STATE/'original.json'
        if args.command == 'inspect':
            print(json.dumps(dict(settings=control.snapshot(), actual=control.actual(),
                                  candidates=PROFILES), indent=2), flush=True)
            return
        if args.command == 'restore':
            control.restore(json.loads(snapshot.read_text()))
            snapshot.unlink()  # remove only after successful readback
            print('RESTORED', flush=True)
            return
        if snapshot.exists():
            raise RuntimeError('Unrestored snapshot exists; run restore first')
        mode = subprocess.run(['/usr/sbin/nvpmodel', '-q'], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip().splitlines()
        if len(mode) < 2 or mode[-1].strip() != '4' or '40W' not in mode[0]:
            raise RuntimeError('Expected fixed 40W mode4; refusing clock changes')
        control.validate(args.profile)
        if args.command == 'cycles':
            control.validate('baseline')
        saved = control.snapshot()
        with snapshot.open('x') as out:
            json.dump(saved, out)
            out.flush()
            os.fsync(out.fileno())
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'signal {signum}')
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        try:
            if args.command == 'cycles':
                control.apply('baseline')
                print(json.dumps(dict(status='READY', monotonic=time.monotonic(),
                                      profile=args.profile, settings=control.snapshot(),
                                      actual=control.actual(), lease_seconds=40)), flush=True)
                cycle_trace(control, args.profile,
                            lambda event: print(json.dumps(event), flush=True))
                return
            control.apply(args.profile)
            print(json.dumps(dict(status='READY', profile=args.profile,
                                  monotonic=time.monotonic(),
                                  settings=control.snapshot(), actual=control.actual(),
                                  lease_seconds=args.seconds)), flush=True)
            end = time.monotonic() + args.seconds
            while time.monotonic() < end:
                if args.trace:
                    print(json.dumps(dict(status='SAMPLE', monotonic=time.monotonic(),
                                          actual=control.actual())), flush=True)
                time.sleep(min(.2, max(0, end-time.monotonic())))
        finally:
            # Prevent a second normal signal from interrupting restoration.
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            control.restore(saved)
            snapshot.unlink()
            print('RESTORED', flush=True)


if __name__ == '__main__':
    main()
