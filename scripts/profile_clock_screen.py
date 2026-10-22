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
                raise RuntimeError('RAM protection stopped worker; active root lease restores within25s')
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
    from perceptbudget.device_clocks import PROFILES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error('Run model as ordinary user; only clock helper uses sudo')
    config = load_config(repo_root=ROOT)
    validate(config)
    source = Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
    if args.dry_run:
        print(json.dumps(dict(profiles=PROFILES, model_loads=1, requests_per_profile=5,
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
        fixed_view_probe(config, args.run_dir, source, 'full', clock_screen=True)
        return 0
    # Fail before loading weights if no interactive terminal/authorization.
    subprocess.run(['sudo', '-v'], check=True)
    run = create_run(config, 'M2', 'clock-screen', repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    summary = dict(status='running', source_run=str(source), scope='four candidate fixed-input screen')
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = [config['python_executable'], '-u', str(Path(__file__).resolve()),
                       '--worker', '--run-dir', str(run)]
            with (run/'worker.log').open('w') as out:
                code = monitored_worker(command, out)
            summary['worker_exit_code'] = code
            if code:
                raise RuntimeError('Worker failed; inspect worker.log and any clock logs for restoration')
            record = json.loads((run/'clock_screen.json').read_text())
            if record['status'] != 'completed' or len(record['conditions']) != 4:
                raise RuntimeError('Incomplete clock screen')
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
