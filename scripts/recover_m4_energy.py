#!/usr/bin/env python3
"""Reintegrate M2 board-power logs; no model load or frequency writes."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.m4_catalog import inventory
from perceptbudget.m4_energy import recover,recover_control_cycles
from perceptbudget.experiment import load_config,create_run,_write_json_atomically

if __name__=='__main__':
    settings=json.loads((ROOT/'m4.json').read_text())
    result=recover(inventory(settings))
    result['control_cycles']=recover_control_cycles(Path(settings['cost_summary']).parents[1]/
        '20261024T154537.029441Z-continue-payback-30059408/continue_payback.json')
    run=create_run(dict(load_config(repo_root=ROOT),m4_settings=settings),'M4','absolute-energy-recovery',repo_root=ROOT)
    _write_json_atomically(run/'summary.json',result)
    metadata=json.loads((run/'run.json').read_text());metadata['status']='completed'
    _write_json_atomically(run/'run.json',metadata)
    print('run_dir='+str(run));print('cells='+str(len(result['cells'])))
