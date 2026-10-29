#!/usr/bin/env python3
"""Run fixed q30 once, then current M4; optionally reuse the saved q30 group."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run,_write_json_atomically
from perceptbudget.m4_acceptance import manifest,summarize,clock_audit


def validate_baseline(source,fixed):
    source=Path(source)
    meta=json.loads((source/'run.json').read_text())['config_snapshot']
    result=json.loads((source/'summary.json').read_text())
    if (meta.get('m4_acceptance_group')!='q30' or
            meta.get('m4_acceptance_manifest')!=fixed or result.get('status')!='completed'):
        raise ValueError('Baseline must be a completed q30 group with the identical fixed manifest')
    expected={tuple(t) for t in fixed['tasks']}
    if {(e['suite'],e['task_id']) for e in result['episodes']}!=expected or len(result['episodes'])!=len(expected):
        raise ValueError('Incomplete baseline task list')
    for episode in result['episodes']:
        run=Path(episode['run'])
        cfg=json.loads((run/'run.json').read_text())['config_snapshot']
        audit=json.loads((run/'system_audit.json').read_text())
        if (cfg.get('candidate_q')!=[30] or cfg.get('m4_enabled') or
                cfg.get('m3_scheduler_mode')!='record_only' or not cfg.get('m3_hold_high_clocks') or
                not audit['passed'] or not clock_audit(run)['passed']):
            raise ValueError('Baseline configuration/audit mismatch: '+str(run))
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run',type=Path,help='Reuse the completed q30 group path from an earlier session')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    cfg=load_config(repo_root=ROOT);fixed=manifest(cfg)
    if args.baseline_run:validate_baseline(args.baseline_run,fixed)
    groups=['joint'] if args.baseline_run else ['q30','joint']
    if args.dry_run:
        print(json.dumps(dict(manifest=fixed,groups_to_run=groups,episodes=12*len(groups),
            model_loads=len(groups),joint_version='current default; no new PACE rule; close-slowdown coverage off'),ensure_ascii=False))
        return 0
    subprocess.run(['sudo','-v'],check=True)
    session=create_run(dict(cfg,m4_acceptance_manifest=fixed),'M4','paired-acceptance',repo_root=ROOT)
    print('session_dir='+str(session),flush=True)
    _write_json_atomically(session/'fixed_manifest.json',fixed)
    sources={'q30':str(args.baseline_run.resolve())} if args.baseline_run else {}
    status=dict(status='running',sources=sources,groups_to_run=groups)
    _write_json_atomically(session/'summary.json',status)
    try:
        for group in groups:
            index=session/(group+'_index.json');log=session/(group+'.log')
            print('START '+group+' log='+str(log),flush=True)
            command=[cfg['python_executable'],'-u',str(ROOT/'scripts/run_m4.py'),
                '--acceptance-group',group,'--result-index',str(index)]
            with log.open('w') as stream:
                code=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
            if index.exists():sources[group]=json.loads(index.read_text())['run']
            _write_json_atomically(session/'summary.json',dict(status='running',sources=sources,groups_to_run=groups))
            if code:raise RuntimeError(group+' failed; inspect '+str(log)+'; completed episodes remain saved')
            if group=='q30':validate_baseline(sources[group],fixed)
            print('DONE '+group+' run='+sources[group],flush=True)
        result=summarize(session,sources)
        _write_json_atomically(session/'summary.json',result)
        print('M4_ACCEPTANCE_STATUS=completed',flush=True)
    except BaseException as exc:
        _write_json_atomically(session/'summary.json',dict(status='invalid',sources=sources,error=f'{type(exc).__name__}: {exc}'))
        print('M4_ACCEPTANCE_STATUS=invalid',flush=True)
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
