#!/usr/bin/env python3
"""Four candidate clock points; one Full model load; explicit sudo helper only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def monitored_worker(command, out):
    # Keep the user's controlling terminal/session for sudo's tty-bound cache.
    # Do not use guarded_run's setsid: that would lose this authorization.
    # Never send group signals: a finite privileged lease may be a descendant.
    from scripts.guarded_run import available_ram_bytes, RAM_FLOOR_BYTES
    if available_ram_bytes() < RAM_FLOOR_BYTES:
        raise RuntimeError('Available RAM below 768MiB; not starting worker')
    child = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
    try:
        while child.poll() is None:
            if available_ram_bytes() < RAM_FLOOR_BYTES:
                raise RuntimeError('RAM protection stopped worker; wait for finite root helper and verify restoration')
            time.sleep(.1)
        return child.returncode
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    from perceptbudget.baseline import validate
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from perceptbudget.device_clocks import PROFILES, SCREEN_PROFILES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--transitions', action='store_true', help='Resident static combination holds and transition cost')
    modes.add_argument('--recovery', action='store_true', help='Short/long idle recovery followed by prepared-input Full RTC')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error('Run model as ordinary user; only clock helper uses sudo')
    config = load_config(repo_root=ROOT)
    validate(config)
    source = Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
    if args.dry_run:
        if args.recovery:
            from perceptbudget.device_clock_recovery import PLAN
            print(json.dumps(dict(plan=PLAN, model_loads=1, H=50, N=5, state=True,
                                  requests_per_condition=5, total_requests=30,
                                  changes_power_mode=False, live_observation=False)))
            return 0
        print(json.dumps(dict(profiles={p: PROFILES[p] for p in (
                              ('baseline', 'gpu_emc', 'gpu_emc_cpu') if args.transitions else SCREEN_PROFILES)},
                              transitions=args.transitions, cycles_per_condition=5 if args.transitions else None,
                              model_loads=1, requests_per_profile=0 if args.transitions else 5,
                              static_hold_seconds=3, H=50, N=5, views=2, state=True,
                              changes_clocks=True, changes_power_mode=False,
                              source_run=str(source)), ensure_ascii=False))
        return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Worker requires M2 output directory')
        # Check the inherited tty credential before loading any weights.
        subprocess.run(['sudo', '-n', 'true'], check=True)
        from perceptbudget.device_profile import fixed_view_probe
        fixed_view_probe(config, args.run_dir, source, 'full', clock_screen=True,
                         clock_transitions=args.transitions, clock_recovery=args.recovery)
        return 0
    # Fail before loading weights if no interactive terminal/authorization.
    subprocess.run(['sudo', '-v'], check=True)
    purpose = 'clock-recovery' if args.recovery else 'clock-transitions' if args.transitions else 'clock-screen'
    run = create_run(config, 'M2', purpose, repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    summary = dict(status='running', source_run=str(source), scope='four candidate fixed-input screen')
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = [config['python_executable'], '-u', str(Path(__file__).resolve()),
                       '--worker', '--run-dir', str(run)]
            if args.transitions:
                command.append('--transitions')
            if args.recovery:
                command.append('--recovery')
            with (run/'worker.log').open('w') as out:
                code = monitored_worker(command, out)
            summary['worker_exit_code'] = code
            if code:
                raise RuntimeError('Worker failed; inspect worker.log and any clock logs for restoration')
            filename = 'clock_recovery.json' if args.recovery else 'clock_transitions.json' if args.transitions else 'clock_screen.json'
            record = json.loads((run/filename).read_text())
            if record['status'] != 'completed' or len(record['conditions']) != (6 if args.recovery else 4):
                raise RuntimeError('Incomplete clock screen')
            if args.transitions or args.recovery:
                summary.update(status='completed', scope=record['scope'], model_loads=record['model_loads'],
                               conditions=record['summary'])
                return 0
            summary.update(status='completed', model_loads=record['model_loads'], conditions=[
                dict(profile=r['profile'], median_ms=r['median_ms'],
                     mean_compute_energy_per_request_j=r['mean_compute_energy_per_request_j'],
                     compute_frequency_check=r['compute_frequency_check'],
                     max_abs_action_difference_vs_baseline=r['max_abs_action_difference_vs_baseline'],
                     static_hold_power_w=r['static_hold']['mean_power_w'], restored=r['restored'])
                for r in record['conditions']])
    except BaseException as exc:
        summary.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(run/'summary.json', summary)
        meta = json.loads((run/'run.json').read_text())
        meta['status'] = summary['status']
        _write_json_atomically(run/'run.json', meta)
        print('CLOCK_SCREEN_STATUS='+json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
