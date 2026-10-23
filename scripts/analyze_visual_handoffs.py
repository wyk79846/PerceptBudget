#!/usr/bin/env python3
"""Offline action handoff diagnostics; never smooths or changes robot actions."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def latest_label(root,label):
    candidates=[]
    for run in root.iterdir():
        meta=run/'run.json'
        if not meta.is_file():
            continue
        d=json.loads(meta.read_text())
        if d.get('purpose') not in ('fresh-mixed-pooled-alternation','fresh-pooled-alternation'):
            continue
        if d.get('config_snapshot',{}).get('m2_test_label')==label:
            candidates.append((d['created_at_utc'],run))
    if not candidates:
        raise ValueError('No run for label '+label)
    return max(candidates)[1]  # Latest attempt, not latest successful attempt.


def handoff_metrics(controls,requests):
    by_id={r['prediction_id']:r for r in requests}
    handoffs=[]
    non_handoff=[]
    def mode(pid):
        return by_id[pid].get('context_refresh',{}).get('mode','unknown')
    for before,after in zip(controls,controls[1:]):
        if after['step']!=before['step']+1:
            raise ValueError('Control steps not consecutive')
        a=np.asarray(before['action'],dtype=float)
        b=np.asarray(after['action'],dtype=float)
        if a.shape!=(7,) or b.shape!=(7,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Require finite seven-dimensional control actions')
        delta=b-a
        entry=dict(step=after['step'],translation_delta_l2=float(np.linalg.norm(delta[:3])),
            rotation_delta_l2=float(np.linalg.norm(delta[3:6])),
            arm_delta_l2=float(np.linalg.norm(delta[:6])),gripper_delta_abs=float(abs(delta[6])))
        old,new=before.get('source'),after.get('source')
        if old is None or new is None:
            continue
        old_id,new_id=old['prediction_id'],new['prediction_id']
        if old_id==new_id:
            non_handoff.append(entry['arm_delta_l2'])
            continue
        entry['transition']=mode(old_id)+'->'+mode(new_id)
        entry['old_prediction_id'],entry['new_prediction_id']=old_id,new_id
        old_request,new_request=by_id[old_id],by_id[new_id]
        oi=after['step']-old_request['observation_step']
        ni=after['step']-new_request['observation_step']
        disagreement=None
        if 0<=oi<len(old_request['actions']) and 0<=ni<len(new_request['actions']):
            old_same_time=np.asarray(old_request['actions'][oi],dtype=float)
            new_same_time=np.asarray(new_request['actions'][ni],dtype=float)
            disagreement=float(np.linalg.norm((new_same_time-old_same_time)[:6]))
        entry['same_target_step_prediction_disagreement_arm_l2']=disagreement
        handoffs.append(entry)
    summary=[]
    for transition in sorted({r['transition'] for r in handoffs}):
        rows=[r for r in handoffs if r['transition']==transition]
        arm=[r['arm_delta_l2'] for r in rows]
        summary.append(dict(transition=transition,count=len(rows),
            arm_jump_median=float(np.median(arm)),arm_jump_p95_descriptive=float(np.percentile(arm,95)),
            gripper_jump_median=float(np.median([r['gripper_delta_abs'] for r in rows])),
            non_handoff_arm_delta_median=float(np.median(non_handoff)) if non_handoff else None))
    return handoffs,summary


def main():
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--labels',nargs='+',choices=('gentle-compression','less-fast'))
    source.add_argument('--runs',nargs='+',type=Path)
    args=parser.parse_args()
    cfg=load_config(repo_root=ROOT)
    parents=[latest_label(Path(cfg['output_root'])/'M2',label) for label in args.labels] if args.labels else args.runs
    records=[]
    for parent in parents:
        summary=json.loads((parent/'summary.json').read_text())
        if summary.get('status')!='completed':
            raise RuntimeError(f'Latest selected run is {summary.get("status")}, not completed: {parent}; no older run substituted')
        records.append((parent,json.loads((parent/'full_quality.json').read_text())))
    run=create_run(cfg,'M2','visual-handoff-analysis',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    report=dict(status='completed',source_runs=[str(p) for p in parents],episodes=[],
        scope='Offline control-space action differences, not physical smoothness certification or causal failure diagnosis; no actions changed')
    all_rows=[]
    for parent,record in records:
        for episode in record['episodes']:
            child=Path(episode['run'])
            requests=json.loads((child/'rtc/q5_requests.json').read_text())
            controls=[r for line in (child/'rtc/trace.jsonl').read_text().splitlines()
                      if (r:=json.loads(line))['event']=='control_step']
            rows,summaries=handoff_metrics(controls,requests)
            report['episodes'].append(dict(source_run=str(child),parent_run=str(parent),
                suite=episode['suite'],view=episode['view'],outcome=episode['outcome'],
                steps=episode['steps'],wall_seconds=episode['wall_seconds'],transitions=summaries))
            all_rows.extend(dict(source_run=str(child),**r) for r in rows)
    _write_json_atomically(run/'summary.json',report)
    if all_rows:
        with (run/'handoffs.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
            writer.writeheader();writer.writerows(all_rows)
    meta=json.loads((run/'run.json').read_text());meta['status']='completed'
    _write_json_atomically(run/'run.json',meta)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
