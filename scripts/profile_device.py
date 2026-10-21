#!/usr/bin/env python3
"""M2 entry: read-only discovery, optionally synthetic GPU sensor steps."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.baseline import validate
from perceptbudget.device_profile import discovery, sensor_steps, fixed_view_probe
from perceptbudget.experiment import create_run, load_config, _write_json_atomically


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensor-check', action='store_true', help='GPU load steps, no model or clock changes')
    parser.add_argument('--fast-probe', action='store_true', help='Fixed-input Full and single-view RTC timing; no quality claim')
    parser.add_argument('--fast-quality', action='store_true', help='Resident Full/single-view paired screen, at most15 episodes')
    parser.add_argument('--source-run', type=Path, help='Successful H50 Spatial0 q5 source with original frames/actions')
    parser.add_argument('--repeats', type=int, default=5, choices=range(1, 21))
    parser.add_argument('--probe-worker', choices=('full', 'single'), help=argparse.SUPPRESS)
    parser.add_argument('--quality-worker', choices=('full', 'single'), help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(repo_root=ROOT)
    validate(config)
    if sum((args.fast_probe, args.sensor_check, args.fast_quality)) > 1:
        parser.error('Choose one M2 operation')
    if (args.fast_probe or args.probe_worker) and not args.source_run:
        parser.error('Fast probe requires --source-run')
    if args.probe_worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Worker requires an existing M2 run directory')
        fixed_view_probe(config, args.run_dir, args.source_run, args.probe_worker, args.repeats)
        return 0
    if args.quality_worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Quality worker requires an existing M2 run directory')
        from perceptbudget.device_quality import resident_quality
        resident_quality(config, args.run_dir, args.quality_worker)
        return 0
    run = create_run(config, 'M2', 'fast-quality' if args.fast_quality else 'device-discovery', repo_root=ROOT)
    print('run_dir=' + str(run), flush=True)
    summary = discovery(config)
    (run / 'inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    try:
        if args.fast_quality:
            summary.update(scope='resident paired task0 calibration and task3 holdout', quality=[])
            with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for kind in ('full', 'single'):
                    cmd = [config['python_executable'], str(ROOT/'scripts/guarded_run.py'), '--',
                           config['python_executable'], '-u', str(Path(__file__).resolve()),
                           '--quality-worker', kind, '--run-dir', str(run)]
                    with (run/f'{kind}_quality.log').open('w') as log:
                        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'{kind} resident quality exited {result.returncode}; completed episodes preserved')
                    record = json.loads((run/f'{kind}_quality.json').read_text())
                    if record['status'] != 'completed':
                        raise RuntimeError('Incomplete resident quality record')
                    summary['quality'].append(record)
                    _write_json_atomically(run/'summary.json', summary)
            full, single = summary['quality']
            selected = single['selection']
            summary['selected_candidate'] = selected
            full_holdout = [r for r in full['episodes'] if r['task_id']==3]
            fast_holdout = [r for r in single['episodes'] if r['task_id']==3]
            summary['fast_small_screen_passed'] = bool(selected and len(full_holdout)==len(fast_holdout)==3
                and all(r['outcome']=='success' and r['underflow_steps']==0 and r['audit_passed']
                        for r in full_holdout+fast_holdout))
        if args.fast_probe:
            summary['scope'] = 'fixed-input compute diagnostic; not the 27-cell service table'
            summary['fast_probe'] = []
            with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for kind in ('full', 'single'):
                    command = [config['python_executable'], str(ROOT/'scripts/guarded_run.py'), '--',
                        config['python_executable'], '-u', str(Path(__file__).resolve()),
                        '--probe-worker', kind, '--run-dir', str(run),
                        '--source-run', str(args.source_run.resolve()), '--repeats', str(args.repeats)]
                    with (run/f'{kind}.log').open('w') as log:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'{kind} probe failed with exit code {result.returncode}; see {kind}.log')
                    record = json.loads((run/f'{kind}_probe.json').read_text())
                    if record['status'] != 'completed':
                        raise RuntimeError('Incomplete probe record')
                    summary['fast_probe'].append(record)
                    _write_json_atomically(run/'summary.json', summary)
                    print(json.dumps(dict(kind=kind, status=record['status'],
                        measurements=[dict(view=m['view'], median_ms=m['median_ms'])
                                      for m in record['measurements']])), flush=True)
            if len({r['input_frame_sha256'] for r in summary['fast_probe']}) != 1 or \
               len({r['old_target_sha256'] for r in summary['fast_probe']}) != 1:
                raise RuntimeError('Full/Fast fixed-input provenance mismatch')
        if args.sensor_check:
            with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                summary['sensor'] = sensor_steps()
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(run / 'summary.json', summary)
        metadata = json.loads((run / 'run.json').read_text())
        metadata['status'] = summary['status']
        _write_json_atomically(run / 'run.json', metadata)
    print(json.dumps(dict(status=summary['status'], clock_settings_changed=False), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
