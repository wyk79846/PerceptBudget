#!/usr/bin/env python3
"""Build measured M4 candidate catalog without model loading or clock writes."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perceptbudget.m4_candidates import build,wait_model,templates
from perceptbudget.m4_costs import load_service_costs,load_transitions
from perceptbudget.experiment import load_config,create_run,_write_json_atomically

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--transition-run',type=Path)
    args=parser.parse_args()
    settings=json.loads((ROOT/'m4.json').read_text());services=load_service_costs(settings)
    transitions=load_transitions(args.transition_run or settings.get('directional_cost_source'))
    waits=wait_model(json.loads(Path(settings['wait_cost_source']).read_text()))
    cases=[]
    for state in settings['hold_profiles']:
        for scene,flags in [('ordinary',{}),('after_fast',{'consecutive_fast':1}),('bootstrap',{'bootstrap':True})]:
            result=build(settings,services,waits,current_profile=state,quiet_age_seconds=.5,
                window_seconds=2,deadline_seconds=2,transitions=transitions,**flags)
            cases.append(dict(current_profile=state,scene=scene,**result))
    run=create_run(dict(load_config(repo_root=ROOT),m4_settings=settings),'M4','candidate-check',repo_root=ROOT)
    _write_json_atomically(run/'summary.json',dict(status='completed',templates=templates(settings),
        wait_age_model_w=waits,cases=cases,runtime_activated=False,
        scope='Measured candidate construction; example timing only, not closed-loop success or energy results'))
    meta=json.loads((run/'run.json').read_text());meta['status']='completed';_write_json_atomically(run/'run.json',meta)
    print('run_dir='+str(run));print('templates',len(templates(settings)))
    for c in cases:print(c['current_profile'],c['scene'],'measured',c['measured_count'],'eligible',c['eligible_count'],'selected',c['selected'])
