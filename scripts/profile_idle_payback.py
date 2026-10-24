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
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    from perceptbudget.device_idle_payback import plan, REPEATS
    if args.dry_run:
        print(json.dumps(dict(plan=plan(), repeats=REPEATS, model_loads=1,
            measured_requests=40, preparation_requests=40, root_model=False,
            fixed_power_mode=True, automatic_threshold=False)))
        return 0
    if os.geteuid() == 0:
        parser.error('Run normally; only the bounded clock helper uses sudo')
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from perceptbudget.baseline import validate
    from scripts.profile_clock_screen import clock_preflight, monitored_worker
    config = load_config(repo_root=ROOT)
    validate(config)
    source = Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Worker requires M2 output directory')
        subprocess.run(['sudo', '-n', 'true'], check=True)
        from perceptbudget.device_profile import fixed_view_probe
        fixed_view_probe(config, args.run_dir, source, 'full', clock_screen=True, idle_payback=True)
        return 0
    subprocess.run(['sudo', '-v'], check=True)
    run = create_run(config, 'M2', 'idle-payback', repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    summary = dict(status='running', model_loads=0, source_run=str(source))
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            clock_preflight(run, ('baseline', 'gpu_emc'), summary)
            summary['model_loads'] = None  # On failure inspect worker metadata, never assume zero loads.
            with (run/'worker.log').open('w') as out:
                code = monitored_worker([config['python_executable'], '-u', str(Path(__file__).resolve()),
                    '--worker', '--run-dir', str(run)], out)
            summary['worker_exit_code'] = code
            if code:
                raise RuntimeError('Worker failed; inspect worker.log and clock restoration logs')
            record = json.loads((run/'idle_payback.json').read_text())
            if record['status'] != 'completed' or len(record['rows']) != 40 or not record['restored']:
                raise RuntimeError('Incomplete matched idle experiment')
            summary.update(status='completed', model_loads=1, conditions=record['summary'])
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
