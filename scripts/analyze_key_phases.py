#!/usr/bin/env python3
"""Read-only phase/guard replay. Does not predict counterfactual task success."""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perceptbudget.key_phases import absolute_intervals,coverage
from perceptbudget.action_events import EventConfig
from perceptbudget.experiment import create_run,load_config,_write_json_atomically


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    args=p.parse_args()
    run=create_run(load_config(repo_root=ROOT),'M4','key-phase-offline',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    result=dict(status='running',model_loads=0,policy_changed=False,cases=[],
                scope='Same recorded histories only; candidate guard decisions, not a new trajectory or success-rate estimate')
    for source in args.runs:
        cfg=json.loads((source/'run.json').read_text())['config_snapshot']
        episode=json.loads((source/'summary.json').read_text())['episode']
        requests=json.loads((source/'rtc/q5_requests.json').read_text())
        commits={c['prediction_id']:c['commit_step'] for c in episode['commits']}
        dynamic=bool(cfg.get('m3_fast_enabled'))
        history=[cfg.get('rtc_initial_delays_by_mode',{}).get('full',cfg.get('rtc_initial_delay_steps',10))]
        rows=[]
        for i,r in enumerate(requests):
            mode=r.get('m3_fast_choice',{}).get('mode') or r.get('context_refresh',{}).get('mode')
            if mode in ('Full','full'):
                if i and r['prediction_id'] in commits:history.append(commits[r['prediction_id']]-r['observation_step'])
                continue
            if i==0:continue
            previous=requests[i-1]
            phases=absolute_intervals(previous['actions'],previous['observation_step'],EventConfig(**cfg.get('m3_events',{})))
            if dynamic:
                span=cfg['m3_timing']['maximum_action_offset'];basis='dynamic maximum coverage before next Full; future Fast chunk unknown'
            else:
                span=min(cfg['H'],max(episode['q'],r['rtc_guidance']['delay_estimate_steps'])+max(history)+1)
                basis='q-counter: max(q,estimated Fast delay)+past Full delay+one tick; no future samples'
            gate=coverage(phases,step=r['observation_step'],now=0.,latest_start=0.,period=.05,horizon=span,prediction_id=previous['prediction_id'])
            rows.append(dict(prediction_id=r['prediction_id'],observation_step=r['observation_step'],
                phases=phases,gate=gate,forecast_basis=basis,
                actual_handoff_step=commits.get(r['prediction_id']),actual_next_handoff_step=commits.get(r['prediction_id']+1)))
        result['cases'].append(dict(source=str(source),outcome=episode['status'],dynamic=dynamic,
            fast_requests=len(rows),blocked=sum(not x['gate']['allowed'] for x in rows),requests=rows))
    result['status']='completed'
    _write_json_atomically(run/'summary.json',result)
    for c in result['cases']:print(json.dumps({k:v for k,v in c.items() if k!='requests'}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
