#!/usr/bin/env python3
"""High hold versus low hold/restore; one ordinary-user model load."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--continue-control', action='store_true', help='Bounded Spatial0 live action execution and one RTC handoff')
    parser.add_argument('--preflight', action='store_true', help='Continue-only one repeat per case, no clock writes or frequency/energy claims')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preflight and not args.continue_control:
        parser.error('--preflight requires --continue-control')
    from perceptbudget.device_idle_payback import plan, REPEATS
    if args.dry_run:
        print(json.dumps(dict(plan=[p for p in plan() if not args.continue_control or p[0] in (.2, 1.)],
            repeats=1 if args.preflight else REPEATS, model_loads=1,
            measured_requests=(4 if args.preflight else 20) if args.continue_control else 40,
            preparation_requests=None if args.continue_control else 40, root_model=False,
            continue_control=args.continue_control, no_clock_writes=args.preflight,
            refresh_observation_steps=[4,20] if args.continue_control else None,
            fixed_power_mode=True, automatic_threshold=False)))
        return 0
    if os.geteuid() == 0:
        parser.error('Run normally; only the bounded clock helper uses sudo')
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from perceptbudget.baseline import validate
    from scripts.profile_clock_screen import clock_preflight, monitored_worker
    config = load_config(repo_root=ROOT)
    validate(config)
    if args.preflight:
        config = dict(config, m2_continue_payback_preflight=True)
    source = Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Worker requires M2 output directory')
        if not args.preflight:
            subprocess.run(['sudo', '-n', 'true'], check=True)
        from perceptbudget.device_profile import fixed_view_probe
        fixed_view_probe(config, args.run_dir, source, 'full', clock_screen=True,
                         idle_payback=not args.continue_control, continue_payback=args.continue_control)
        return 0
    if not args.preflight:
        subprocess.run(['sudo', '-v'], check=True)
    run = create_run(config, 'M2', ('continue-payback-preflight' if args.preflight else 'continue-payback')
                     if args.continue_control else 'idle-payback', repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    summary = dict(status='running', model_loads=0, source_run=str(source))
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not args.preflight:
                clock_preflight(run, ('baseline', 'gpu_emc'), summary)
            summary['model_loads'] = None  # On failure inspect worker metadata, never assume zero loads.
            command = [config['python_executable'], '-u', str(Path(__file__).resolve()),
                       '--worker', '--run-dir', str(run)]
            if args.continue_control:
                command.append('--continue-control')
            if args.preflight:
                command.append('--preflight')
            with (run/'worker.log').open('w') as out:
                code = monitored_worker(command, out)
            summary['worker_exit_code'] = code
            if code:
                raise RuntimeError('Worker failed; inspect worker.log and clock restoration logs')
            filename = 'continue_payback.json' if args.continue_control else 'idle_payback.json'
            expected = (4 if args.preflight else 20) if args.continue_control else 40
            record = json.loads((run/filename).read_text())
            if record['status'] != 'completed' or len(record['rows']) != expected or (not args.preflight and not record['restored']):
                raise RuntimeError('Incomplete matched idle experiment')
            summary.update(status='completed', model_loads=1, conditions=record['summary'],
                           preflight_only=args.preflight, clock_settings_changed=not args.preflight)
    except BaseException as exc:
        summary.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(run/'summary.json', summary)
        meta = json.loads((run/'run.json').read_text())
        meta['status'] = summary['status']
        _write_json_atomically(run/'run.json', meta)
        print('IDLE_PAYBACK_STATUS='+json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
