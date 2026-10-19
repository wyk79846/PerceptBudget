"""Offline old-tail/action-source audit. No claim about unsaved model inputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def audit(run):
    import numpy as np
    from perceptbudget.rtc_schedule import DelayQueue
    cfg=json.loads((run/'run.json').read_text())['config_snapshot']
    req=json.loads((run/'rtc/q5_requests.json').read_text())
    events=[json.loads(s) for s in (run/'rtc/trace.jsonl').read_text().splitlines()]
    controls={r['step']:r for r in events if r['event']=='control_step'}
    commits=[r for r in events if r['event']=='prediction_committed']
    by_id={r['prediction_id']:r for r in req}
    by_observation={r['observation_step']:r for r in req}
    history=DelayQueue(cfg['rtc_initial_delay_steps'])
    records=[]
    for r in req:
        g=r['rtc_guidance']
        if r['observation_step']==0:
            assert not r.get('delay_queue_observed')
            continue
        origin=r['observation_step']
        s=g['executed_steps']
        prior=by_observation[g['old_chunk_origin_step']]
        checks=dict(delay_history=g['delay_estimate_steps']==history.estimate,
            target_length=len(prior['actions_raw'][s:])==cfg['H']-s,
            old_origin=origin-s==prior['observation_step'],
            committed_actions_match=True)
        if 'rtc_solver_inputs' in r:
            solver=r['rtc_solver_inputs']
            checks['actual_solver_target']=np.array_equal(np.asarray(solver['previous_raw'],dtype=np.float32),
                np.asarray(prior['actions_raw'][s:],dtype=np.float32))
            checks['solver_delay']=solver['delay']==g['delay_estimate_steps']
            checks['solver_executed']=solver['executed']==s
        checked=0
        actual=g.get('actual_delay_steps')
        if actual is not None:
            for step in range(origin,origin+actual):
                if step not in controls:
                    continue
                row=controls[step]
                src=row['source']
                checks['committed_actions_match'] &= (src is not None
                    and src['prediction_id']==prior['prediction_id']
                    and src['offset']==step-prior['observation_step']
                    and np.array_equal(np.asarray(row['action'],dtype=np.float32),
                        np.asarray(prior['actions'][step-prior['observation_step']],dtype=np.float32)))
                checked+=1
            history.observe(actual,observation_step=origin)
        records.append(dict(prediction_id=r['prediction_id'],observation_step=origin,
                            old_executed_actions_checked=checked,checks=checks))
    source_valid=True
    for step,row in controls.items():
        src=row['source']
        source_valid &= (src is not None and row['applied_arm_gain']==1
            and src['origin_step']+src['offset']==step
            and np.array_equal(np.asarray(row['action'],dtype=np.float32),
                np.asarray(by_id[src['prediction_id']]['actions'][src['offset']],dtype=np.float32)))
    crops=all(r['expired_new_actions']==r['commit_step']-r['observation_step'] for r in commits)
    saved_inputs=all('rtc_solver_inputs' in r for r in req)
    return dict(run=str(run),scope='recorded old-tail alignment/history/crop/action provenance; historical unsaved solver targets cannot be checked',
        actual_solver_inputs_saved=saved_inputs,
        passed=bool(source_valid and crops and all(all(r['checks'].values()) for r in records)),
        source_valid=bool(source_valid),crops_match=crops,controls_checked=len(controls),requests=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit(args.run)
    if args.output:
        from scripts import eval_q as base
        base._write_json_atomic(args.output,result)
    print(json.dumps(result),flush=True)
