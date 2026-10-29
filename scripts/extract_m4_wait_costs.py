#!/usr/bin/env python3
"""Reuse measured model-resident control waits; no inference or clock writes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.m4_wait_costs import extract
from perceptbudget.experiment import load_config,create_run,_write_json_atomically

if __name__=='__main__':
    cfg=load_config(repo_root=ROOT);base=Path(cfg['output_root'])/'M3'
    sources=[base/'20261028T023452.333326Z-comparison-7e34c992/recording.json',
        base/'20261028T031146.686023Z-margin-scan-c52efee7/recording.json',
        base/'20261028T112013.062341Z-margin-scan-dc9d03b2/recording.json']
    runs=sorted({e['run'] for p in sources for e in json.loads(p.read_text())['episodes']})
    results=[]
    for guard in (.2,.3,.5):
        episodes=[extract(p,guard) for p in runs];groups=[]
        for profile in ('baseline','gpu_emc'):
            windows=[w for e in episodes for w in e['windows'] if w['profile']==profile]
            seconds=sum(w['seconds'] for w in windows);energy=sum(w['energy_j'] for w in windows)
            groups.append(dict(profile=profile,intervals=len(windows),seconds=seconds,energy_j=energy,
                mean_power_w=energy/seconds if seconds else None))
        results.append(dict(guard_seconds=guard,groups=groups,episodes=episodes))
    run=create_run(cfg,'M4','wait-cost-extraction',repo_root=ROOT)
    _write_json_atomically(run/'summary.json',dict(status='completed',source_runs=runs,results=results,
        scope='Active control with no neural request, frequency verified; guard sensitivity reported; not steady hardware idle proof',
        transition_energy_status='Separate transition energy unresolved; use paired cycle measurements before activation'))
    meta=json.loads((run/'run.json').read_text());meta['status']='completed';_write_json_atomically(run/'run.json',meta)
    print('run_dir='+str(run))
    for r in results:print(r['guard_seconds'],r['groups'])
