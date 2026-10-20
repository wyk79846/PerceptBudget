import json
import numpy as np
import pytest
from scripts.audit_rtc_system import audit,decode_raw


def fixture(tmp_path):
    (tmp_path/'rtc').mkdir()
    stats={'state':{'q01':[-1.]*8,'q99':[1.]*8},'actions':{'q01':[-1.]*7,'q99':[1.]*7}}
    norm=tmp_path/'norm.json';norm.write_text(json.dumps(stats))
    cfg=dict(H=25,rtc_initial_delay_steps=5,controlled_delay_steps=5,rtc_paper_schedule=False,norm_stats_path=str(norm))
    (tmp_path/'run.json').write_text(json.dumps({'config_snapshot':cfg}))
    normalized=((np.zeros(8,dtype=np.float32)+1)/(2+1e-6)*2-1).tolist()
    requests=[]
    for pid,origin in ((0,0),(1,5)):
        raw=np.full((25,32),pid*.1,dtype=np.float32)
        requests.append(dict(prediction_id=pid,observation_step=origin,actions_raw=raw.tolist(),
            actions=decode_raw(raw,stats).tolist(),observation_raw_state=[0]*8,model_state=normalized,
            observation_monotonic=origin*10.,input_ready_monotonic=origin*10.+.1,
            inference_started_monotonic=origin*10.+.2,inference_ended_monotonic=origin*10.+.3,
            ready_monotonic=origin*10.+.4,rtc_solver_inputs=dict(noise=np.zeros((25,32)).tolist(),
                previous_raw=np.zeros((20,32)).tolist() if pid else None,executed=5 if pid else 0,delay=5 if pid else 0),
            rtc_guidance=dict(delay_estimate_steps=5 if pid else 0,executed_steps=5 if pid else 0,actual_delay_steps=5 if pid else 0)))
    events=[]
    for pid,origin,at in ((0,0,0),(1,5,10)):
        events.append(dict(event='prediction_committed',prediction_id=pid,observation_step=origin,
            commit_step=at,commit_monotonic=at*10.+.5,expired_new_actions=at-origin,
            new_prediction_remaining_actions=25-(at-origin)))
    for step in range(11):
        pid=int(step>=10);offset=step-(5 if pid else 0)
        events.append(dict(event='control_step',step=step,before={'raw_state':[0]*8},
            action_end_monotonic=step*10.+1,applied_arm_gain=1,action=requests[pid]['actions'][offset],
            source=dict(prediction_id=pid,offset=offset,origin_step=5 if pid else 0)))
    def save():
        (tmp_path/'rtc/q5_requests.json').write_text(json.dumps(requests))
        (tmp_path/'rtc/trace.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    save()
    return requests,events,save


def test_synthetic_valid_system_trace(tmp_path):
    fixture(tmp_path)
    assert audit(tmp_path)['passed']


@pytest.mark.parametrize('change,expected',[
    ('state','state_normalization'),('target','old_target_alignment'),
    ('decode','action_postprocessing'),('source','action_is_latest_committed_chunk'),
    ('crop','exact_absolute_crop')])
def test_system_audit_rejects_injected_mistakes(tmp_path,change,expected):
    requests,events,save=fixture(tmp_path)
    if change=='state':requests[1]['model_state'][0]=10
    if change=='target':requests[1]['rtc_solver_inputs']['previous_raw'][0][0]=10
    if change=='decode':requests[1]['actions'][0][0]=10
    if change=='source':events[-1]['source']['offset']=0
    if change=='crop':events[1]['expired_new_actions']=4
    save()
    report=audit(tmp_path)
    assert not report['passed']
    assert expected in [x['check'] for x in report['failures']]
