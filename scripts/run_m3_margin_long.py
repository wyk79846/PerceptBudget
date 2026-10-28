#!/usr/bin/env python3
"""Two independent resident-model rounds of paired eta50/25 validation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def commands():
    return [[sys.executable,'-u',str(ROOT/'scripts/run_m3.py'),'--mode','margin-scan',
             '--margin-long','--margin-round',str(i)] for i in (1,2)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    jobs=commands()
    if args.dry_run:
        print(json.dumps(dict(rounds=2,episodes_per_round=18,total_episodes=36,
            model_loads=2,commands=jobs)))
        return 0
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    config=load_config(repo_root=ROOT)
    output=Path(config['output_root'])/'M3'
    directory=create_run(config,'M3','margin-long-session',repo_root=ROOT)
    print('index_dir='+str(directory),flush=True)
    record=dict(status='running',rounds=[])
    try:
        for index,command in enumerate(jobs,1):
            entry=dict(round=index,command=command,status='running')
            record['rounds'].append(entry)
            _write_json_atomically(directory/'session.json',record)
            print(f'START independent_round={index}/2 episodes=18',flush=True)
            before=set(output.glob('*-margin-scan-*'))
            code=subprocess.run(command,cwd=ROOT).returncode
            entry['run_directories']=[str(p) for p in sorted(set(output.glob('*-margin-scan-*'))-before)]
            entry.update(returncode=code,status='completed' if code==0 else 'invalid')
            if code:
                raise RuntimeError(f'Round{index} failed; stop without automatic retry')
        record['status']='completed'
        return 0
    except BaseException as exc:
        record.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(directory/'session.json',record)
        metadata=json.loads((directory/'run.json').read_text())
        metadata['status']=record['status']
        _write_json_atomically(directory/'run.json',metadata)
        print('MARGIN_LONG_STATUS='+record['status'],flush=True)


if __name__=='__main__':
    raise SystemExit(main())
