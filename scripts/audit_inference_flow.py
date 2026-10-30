#!/usr/bin/env python3
"""Independent read-only replay of recorded RTC input and queue invariants."""
import argparse,json
from pathlib import Path
import numpy as np


def audit(run):
    run=Path(run)
    cfg=json.loads((run/'run.json').read_text())['config_snapshot']
    q=cfg['candidate_q'][0]
    requests=json.loads((run/'rtc'/f'q{q}_requests.json').read_text())
    controls=[];commits=[]
    with (run/'rtc/trace.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['event']=='control_step':controls.append(row)
            elif row['event']=='prediction_committed':commits.append(row)
    errors=[];checks=0
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:errors.append(message)
    bypid={r['prediction_id']:r for r in requests}
    bystep={r['step']:r for r in controls}
    norm=json.loads(Path(cfg['norm_stats_path']).read_text())
    stats=norm['actions']
    low=np.array(stats['q01'],dtype=np.float32);high=np.array(stats['q99'],dtype=np.float32)
    span=np.where(high-low==0,np.float32(1e-8),high-low)
    for i,r in enumerate(requests):
        pid=r['prediction_id'];origin=r['observation_step'];raw=np.array(r['actions_raw'],dtype=np.float32)
        check(raw.shape==(cfg['H'],32) and np.isfinite(raw).all(),f'pid{pid}: raw shape/finite')
        decoded=(raw[:,:7]+1)*span/2+low
        check(np.array_equal(decoded,np.array(r['actions'],dtype=np.float32)),f'pid{pid}: decoding mismatch')
        guide=r['rtc_guidance'];solver=r.get('rtc_solver_inputs')
        if cfg['state_enabled']:
            sl=np.asarray(norm['state']['q01'],dtype=np.float32)
            sh=np.asarray(norm['state']['q99'],dtype=np.float32)
            ss=np.where(sh-sl==0,np.float32(1e-8),sh-sl)
            expected=np.pad(2*(np.asarray(r['observation_raw_state'],dtype=np.float32)-sl)/ss-1,(0,24))
            check(np.array_equal(expected,np.asarray(r['model_state'],dtype=np.float32)),f'pid{pid}: normalized state')
        if solver:
            z=np.asarray(solver['noise'])
            check(z.shape==(cfg['H'],32) and np.isfinite(z).all(),f'pid{pid}: noise shape/finite')
        if origin in bystep:
            check(np.array_equal(r['observation_raw_state'],bystep[origin]['before']['raw_state']),f'pid{pid}: state not from observation step')
        check(r['observation_monotonic']<=r['input_ready_monotonic']<=r['inference_started_monotonic']<=r['inference_ended_monotonic']<=r['ready_monotonic'],f'pid{pid}: request time order')
        if i:
            prior=requests[i-1];s=origin-prior['observation_step'];d=guide['delay_estimate_steps']
            check(guide['executed_steps']==s and 0<=d<=cfg['H']-s,f'pid{pid}: RTC s/d bounds')
            check(guide['guidance_enabled'] and guide['state_in_prompt']==r.get('context_refresh',{}).get('current_state_encoded',cfg['state_enabled']),f'pid{pid}: guidance/state flags')
            check(prior['ready_monotonic']<=r['observation_monotonic'],f'pid{pid}: overlapping inference')
            if solver:
                check(solver['delay']==d and solver['executed']==s,f'pid{pid}: solver delay/executed mismatch')
                target=np.array(prior['actions_raw'],dtype=np.float32)[s:]
                check(np.array_equal(np.array(solver['previous_raw'],dtype=np.float32),target),f'pid{pid}: wrong old raw tail')
        elif solver:check(solver['previous_raw'] is None,f'pid{pid}: bootstrap guided')
    for c in commits:
        r=bypid[c['prediction_id']];elapsed=c['commit_step']-r['observation_step']
        check(elapsed>=0 and c['expired_new_actions']==min(elapsed,cfg['H']),f"pid{c['prediction_id']}: expired offset")
        check(c['new_queue_actions']==max(0,cfg['H']-elapsed),f"pid{c['prediction_id']}: queue size")
        check(r['ready_monotonic']<=c['commit_monotonic'],f"pid{c['prediction_id']}: committed before ready")
    for row in controls:
        src=row.get('source');step=row['step']
        if src is None:continue
        r=bypid[src['prediction_id']];offset=step-r['observation_step']
        check(src['origin_step']==r['observation_step'] and src['offset']==offset and 0<=offset<cfg['H'],f'step{step}: offset/origin')
        action=np.array(r['actions'][offset],dtype=np.float32);action[:6]*=row['applied_arm_gain']
        check(np.array_equal(action,np.array(row['action'],dtype=np.float32)),f'step{step}: wrong executed action')
        applicable=[c for c in commits if c['commit_step']<=step]
        check(bool(applicable) and src['prediction_id']==applicable[-1]['prediction_id'],f'step{step}: stale/new source')
        check(row['applied_arm_gain']==1.,f'step{step}: unexpected gain')
    return dict(run=str(run),passed=not errors,checks=checks,requests=len(requests),controls=len(controls),errors=errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cases=[audit(p) for p in args.runs]
    result=dict(passed=all(c['passed'] for c in cases),runs=len(cases),checks=sum(c['checks'] for c in cases),cases=cases)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    for c in cases:
        if not c['passed']:print(c['run'],c['errors'][:8])
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
