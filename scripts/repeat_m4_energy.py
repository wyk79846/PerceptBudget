#!/usr/bin/env python3
"""Three extra paired Full/Fast batches using existing live-service measurement."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def jobs():
    out=[]
    for batch in (1,2,3):
        for mode in (('Full','Fast') if batch%2 else ('Fast','Full')):
            args=['--full-service','--skip-continue-validation'] if mode=='Full' else ['--selected-fast-clocks','--live-service']
            out.append(dict(batch=batch,mode=mode,command=[sys.executable,'-u',str(ROOT/'scripts/profile_clock_screen.py'),*args,
                '--service-validation','--service-hot-only','--validation-batch',str(batch),'--service-profiles','baseline']))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(jobs=jobs(),services=120,model_loads=6,continue_windows=0)));return
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    config=load_config(repo_root=ROOT);run=create_run(config,'M4','energy-repeat-session',repo_root=ROOT)
    print('index_dir='+str(run),flush=True);record=dict(status='running',jobs=[])
    try:
        for job in jobs():
            entry=dict(job,status='running');record['jobs'].append(entry);_write_json_atomically(run/'session.json',record)
            output=Path(config['output_root'])/'M2';before=set(output.iterdir())
            print('START batch='+str(job['batch'])+' mode='+job['mode'],flush=True)
            code=subprocess.run(job['command'],cwd=ROOT).returncode
            entry.update(status='completed' if code==0 else 'invalid',returncode=code,
                raw_runs=[str(p) for p in sorted(set(output.iterdir())-before) if p.is_dir()])
            if code:raise RuntimeError('Measurement failed; stop without automatic retry')
        record['status']='completed'
    except BaseException as exc:
        record.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        _write_json_atomically(run/'session.json',record)
        meta=json.loads((run/'run.json').read_text());meta['status']=record['status'];_write_json_atomically(run/'run.json',meta)


if __name__=='__main__':main()
