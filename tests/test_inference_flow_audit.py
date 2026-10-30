import json
from scripts.audit_inference_flow import audit


def test_record_audit_rejects_changed_action_and_source_offset(tmp_path):
    """Independent audit must detect corrupt execution, even if shapes are valid."""
    rtc=tmp_path/'rtc';rtc.mkdir()
    norm=tmp_path/'norm.json'
    norm.write_text(json.dumps(dict(actions=dict(q01=[-1.]*7,q99=[1.]*7))))
    cfg=dict(candidate_q=[5],H=2,state_enabled=False,norm_stats_path=str(norm))
    (tmp_path/'run.json').write_text(json.dumps(dict(config_snapshot=cfg)))
    request=dict(prediction_id=0,observation_step=0,actions_raw=[[0.]*32]*2,
        actions=[[0.]*7]*2,observation_raw_state=[0.]*8,rtc_guidance={},
        rtc_solver_inputs=dict(previous_raw=None,noise=[[0.]*32]*2),
        observation_monotonic=1.,input_ready_monotonic=2.,inference_started_monotonic=3.,
        inference_ended_monotonic=4.,ready_monotonic=5.)
    (rtc/'q5_requests.json').write_text(json.dumps([request]))
    commit=dict(event='prediction_committed',prediction_id=0,commit_step=0,
        commit_monotonic=6.,expired_new_actions=0,new_queue_actions=2)
    step=dict(event='control_step',step=0,source=dict(prediction_id=0,origin_step=0,offset=0),
        action=[0.]*7,applied_arm_gain=1.,before=dict(raw_state=[0.]*8))
    def save():
        (rtc/'trace.jsonl').write_text(json.dumps(commit)+'\n'+json.dumps(step)+'\n')
    save();assert audit(tmp_path)['passed']
    step['action'][0]=.5;save()
    assert any('wrong executed action' in e for e in audit(tmp_path)['errors'])
    step['action'][0]=0.;step['source']['offset']=1;save()
    assert any('offset/origin' in e for e in audit(tmp_path)['errors'])
