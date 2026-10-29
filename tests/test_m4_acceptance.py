import json

from perceptbudget.baseline import FROZEN
from perceptbudget.m4_acceptance import manifest, configure_baseline, request_path, summarize


def test_manifest_fixes_twelve_cases_and_detects_model_changes():
    cfg=dict(FROZEN,checkpoint_path='/checkpoint',norm_stats_path='/stats')
    fixed=manifest(cfg)
    assert len(fixed['tasks'])==len({tuple(t) for t in fixed['tasks']})==12
    assert {s for s,t in fixed['tasks']}=={'libero_spatial','libero_object','libero_goal','libero_10'}
    assert fixed['seed']==7 and fixed['init']==0
    cfg['N']=10
    assert manifest(cfg)['fingerprint']!=fixed['fingerprint']


def test_q30_has_no_joint_scheduler_or_fast():
    cfg=dict(m4_enabled=True,m3_scheduler_mode='deadline_dvfs',m3_action_events_enabled=True,
             m3_dvac_enabled=True,m2_visual_fast_enabled=True,m2_split_delay=True)
    assert configure_baseline(cfg)==30
    assert cfg['candidate_q']==[30] and cfg['m3_scheduler_mode']=='record_only'
    assert cfg['m3_hold_high_clocks']
    assert not any(cfg[k] for k in ('m4_enabled','m3_action_events_enabled','m3_dvac_enabled',
                                  'm2_visual_fast_enabled','m2_split_delay'))


def test_report_reads_actual_q_and_does_not_call_failure_energy_savings(tmp_path):
    sources={}
    for group,q in [('q30',30),('joint',5)]:
        parent=tmp_path/group;run=parent/'episode';(run/'rtc').mkdir(parents=True)
        (run/'run.json').write_text(json.dumps({'config_snapshot':{'candidate_q':[q]}}))
        (run/'summary.json').write_text(json.dumps({'episode':{
            'real_seconds':1,'including_drain_seconds':1.1,
            'including_drain_power':{'total_energy_mj':1000 if group=='q30' else 500},'underflow_steps':[]}}))
        request=dict(observation_raw_state=[1],model_state=[2],actions_raw=[[3]],rtc_solver_inputs={'noise':[[4]]})
        if group=='joint':request['actions_raw']=[[3.01]]
        request_path(run).write_text(json.dumps([request]))
        episode=dict(suite='libero_spatial',task_id=0,run=str(run),steps=10,
            outcome='success' if group=='q30' else 'timeout',initial={'initial_hashes':{'image':'same'}})
        (parent/'summary.json').write_text(json.dumps({'status':'completed','episodes':[episode]}))
        sources[group]=parent
    out=tmp_path/'report';out.mkdir()
    result=summarize(out,sources)
    pair=result['pairs'][0]
    assert pair['initial_inputs_equal'] and not pair['initial_actions_exact']
    assert pair['initial_raw_max_abs']>0
    assert not pair['both_success'] and pair['energy_saving_percent'] is None
    assert len(result['rows'])==2 and (out/'comparison.csv').exists()
