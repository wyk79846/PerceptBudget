#!/usr/bin/env python3
"""Three independent batches, resident model inside each geometry/session."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def jobs(batches):
    return [dict(batch=batch, mode=mode, args=(['--full-service'] if mode=='full_continue'
            else ['--selected-fast-clocks', '--live-service']))
            for batch in batches for mode in ('full_continue', 'fast')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--all',action='store_true',help='Run all three independent batches sequentially')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    batches=(1,2,3) if args.all else (args.batch,)
    if args.dry_run:
        print(json.dumps(dict(jobs=jobs(batches),model_loads=2*len(batches),
            service_requests=120*len(batches),continue_windows=60*len(batches),
            note='Full and Continue share one model; Fast geometry separate; no per-request reload')))
        return 0
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    config=load_config(repo_root=ROOT)
    run=create_run(config,'M2','validation-session',repo_root=ROOT)
    print('index_dir='+str(run),flush=True)
    result=dict(status='running',jobs=[],batches=list(batches),planned_model_loads=2*len(batches))
    try:
        for job in jobs(batches):
            subprocess.run(['sudo','-v'],check=True)
            label='batch'+str(job['batch'])+'_'+job['mode']
            log=run/(label+'.log')
            command=[config['python_executable'],'-u',str(ROOT/'scripts/profile_clock_screen.py'),
                     *job['args'],'--service-validation','--validation-batch',str(job['batch']),
                     '--service-profiles','baseline']
            print('START '+label+' log='+str(log),flush=True)
            with log.open('w') as out:
                code=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT).returncode
            child_runs=[line.split('=',1)[1] for line in log.read_text().splitlines() if line.startswith('run_dir=')]
            child=Path(child_runs[-1]) if child_runs else None
            row=dict(job,log=str(log),exit_code=code,run=None if child is None else str(child))
            result['jobs'].append(row)
            _write_json_atomically(run/'validation_index.json',result)
            if code or child is None or json.loads((child/'summary.json').read_text())['status']!='completed':
                raise RuntimeError('Validation failed at '+label+'; inspect '+str(log)+' and child clock restoration')
            if job['mode']=='fast':
                full_job=next(r for r in result['jobs'] if r['batch']==job['batch'] and r['mode']=='full_continue')
                full=json.loads((Path(full_job['run'])/'full_service.json').read_text())
                fast=json.loads((child/'selected_fast_service.json').read_text())
                if any(full[k]!=fast[k] for k in ('initial_physics_sha256','old_target_sha256')):
                    raise RuntimeError('Full/Fast calibration physics or old targets differ; do not compare costs')
                a=full['requests'][0]['measured'];b=fast['requests'][0]['measured']
                if a['frame_sha256']!=b['frame_sha256'] or a['raw_state']!=b['raw_state']:
                    raise RuntimeError('Full/Fast live observations differ; not a paired cost table')
                row['cross_mode_observation_pairing_passed']=True
                _write_json_atomically(run/'validation_index.json',result)
            print('DONE '+label+' run='+str(child),flush=True)
        result['status']='completed'
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(run/'validation_index.json',result)
        meta=json.loads((run/'run.json').read_text());meta['status']=result['status']
        _write_json_atomically(run/'run.json',meta)
        print('M2_VALIDATION_STATUS='+result['status'],flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
