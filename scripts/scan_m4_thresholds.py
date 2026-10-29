#!/usr/bin/env python3
"""Scan existing measured-cost model; save curves, not online policy changes."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perceptbudget.m4_costs import load_service_costs,load_transitions
from perceptbudget.m4_candidates import wait_model
from perceptbudget.m4_thresholds import scan
from perceptbudget.experiment import load_config,create_run,_write_json_atomically

if __name__=='__main__':
    cfg=json.loads((ROOT/'m4.json').read_text());services=load_service_costs(cfg)
    waits=wait_model(json.loads(Path(cfg['wait_cost_source']).read_text()));down=load_transitions(cfg['directional_cost_source'])['baseline','gpu_emc']
    run=create_run(dict(load_config(repo_root=ROOT),m4_settings=cfg),'M4','threshold-scan',repo_root=ROOT)
    cases=[]
    for history in ('short','long'):
        low=next(c for c in services if c['mode']=='Full' and c['profile']=='gpu_emc' and c['history']==history)
        high=next(c for c in services if c['mode']=='Full' and c['profile']=='baseline' and c['history']==history)
        for period in (.05,.07):
            result=scan(low,high,waits,down,control_period=period)
            rows=result.pop('rows');name=history+'-'+str(round(period*1000))+'ms.csv'
            with (run/name).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            case=dict(history=history,control_period_seconds=period,curves=name,**result);cases.append(case)
            print(json.dumps(case))
    _write_json_atomically(run/'summary.json',dict(status='completed',cases=cases,range_seconds=[.001,10.],step_seconds=.001,
        scope='Offline model scan, not paired physical measurements or permanent runtime threshold; return-low cost uses250ms block',
        costs=cfg,runtime_changed=False))
    meta=json.loads((run/'run.json').read_text());meta['status']='completed';_write_json_atomically(run/'run.json',meta)
    print('run_dir='+str(run))
