"""Summarize matched closed-loop frequency policies, never mix terminal hold with Continue."""
import csv
import json
from pathlib import Path

from perceptbudget.experiment import _write_json_atomically


def summarize_pairs(episodes, directory):
    rows=[]
    for e in episodes:
        run=Path(e['run'])
        v=json.loads((run/'summary.json').read_text())['episode']
        tail=v['terminal_low']
        requests=json.loads((run/'rtc/q5_requests.json').read_text())
        rows.append(dict(suite=e['suite'],task_id=e['task_id'],group=e['view'],
            outcome=e['outcome'],steps=e['steps'],task_seconds=v['real_seconds'],
            task_j=v['power']['total_energy_mj']/1000,
            task_plus_inference_j=tail['task_plus_inference_power']['total_energy_mj']/1000,
            terminal_low_j=tail['power']['total_energy_mj']/1000,
            common_endpoint_j=v['including_drain_power']['total_energy_mj']/1000,
            requests=len(requests),low_requests=sum(r.get('m4_compute_profile')=='gpu_emc' for r in requests),
            unexecuted_requests=sum(r['prediction_id'] not in {c['prediction_id'] for c in v['commits'] if c['commit_step']<v['steps']} for r in requests),
            run=str(run)))
    for suite,tid in {(r['suite'],r['task_id']) for r in rows}:
        pair=[r for r in rows if (r['suite'],r['task_id'])==(suite,tid)]
        if len(pair)!=2 or {r['group'] for r in pair}!={'full_high','full_low'}:
            raise ValueError('Incomplete Full frequency pair')
    result=dict(rows=rows,scope='Matched whole tasks; task, inference drain, and common LOW endpoint reported separately; restoration excluded')
    directory=Path(directory)
    with (directory/'full_frequency_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    _write_json_atomically(directory/'full_frequency_pairs.json',result)
    return result
