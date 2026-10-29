#!/usr/bin/env python3
"""Inventory M4 candidates without loading models or changing clocks."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run,_write_json_atomically
from perceptbudget.m4_catalog import inventory


def main():
    settings=json.loads((ROOT/'m4.json').read_text())
    result=inventory(settings)
    run=create_run(dict(load_config(repo_root=ROOT),m4_settings=settings),'M4','candidate-inventory',repo_root=ROOT)
    _write_json_atomically(run/'summary.json',result)
    meta=json.loads((run/'run.json').read_text());meta['status']='completed'
    _write_json_atomically(run/'run.json',meta)
    print('run_dir='+str(run))
    print('M4_CATALOG_STATUS='+json.dumps(dict(status='completed',cells=len(result['cells']),model_loads=0,clocks_changed=False)))


if __name__=='__main__':main()
