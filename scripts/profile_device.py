#!/usr/bin/env python3
"""M2 entry: read-only discovery, optionally synthetic GPU sensor steps."""
import argparse
import fcntl
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.baseline import validate
from perceptbudget.device_profile import discovery, sensor_steps
from perceptbudget.experiment import create_run, load_config, _write_json_atomically


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensor-check', action='store_true', help='GPU load steps, no model or clock changes')
    args = parser.parse_args()
    config = load_config(repo_root=ROOT)
    validate(config)
    run = create_run(config, 'M2', 'device-discovery', repo_root=ROOT)
    print('run_dir=' + str(run), flush=True)
    summary = discovery(config)
    (run / 'inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    try:
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
