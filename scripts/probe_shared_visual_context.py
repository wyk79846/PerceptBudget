#!/usr/bin/env python3
"""No-rollout feasibility probe for shared-weight Full/wrist context switching."""
import fcntl
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast-kind',choices=('wrist','pool2','pool2_agent'),default='wrist')
    args=parser.parse_args()
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from perceptbudget.baseline import validate
    from perceptbudget.device_profile import fixed_view_probe
    config = load_config(repo_root=ROOT)
    validate(config)
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = create_run(config, 'M2', 'shared-visual-context', repo_root=ROOT)
        print('run_dir='+str(run), flush=True)
        source = Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
        result = dict(status='running')
        try:
            result = fixed_view_probe(config, run, source, 'full', shared_views=True,shared_view_kind=args.fast_kind)
        except BaseException as exc:
            result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            _write_json_atomically(run/'summary.json', result)
            meta = json.loads((run/'run.json').read_text())
            meta['status'] = result['status']
            _write_json_atomically(run/'run.json', meta)
            print('SHARED_CONTEXT_STATUS='+result['status'], flush=True)


if __name__ == '__main__':
    main()
