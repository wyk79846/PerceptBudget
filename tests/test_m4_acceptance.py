import json
import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from perceptbudget.baseline import FROZEN
from perceptbudget.m4_acceptance import manifest, configure_baseline, request_path, summarize


def test_common_endpoint_tail_is_separate_from_comparison_energy():
    from perceptbudget.m4_acceptance import comparison_energy
    episode=dict(including_drain_power={'total_energy_mj':16000},terminal_low={
        'task_plus_inference_power':{'total_energy_mj':10000},'power':{'total_energy_mj':6000}})
    result=comparison_energy(episode)
    assert result['total_energy_j']==10.
    assert result['measurement_tail_energy_j']==6.
    assert result['common_endpoint_energy_j']==16.
    assert result['energy_scope']=='task_plus_inference'
    del episode['terminal_low']
    assert comparison_energy(episode)['total_energy_j']==16.


def test_new_twelve_task_command_forwards_current_clock_policy_without_loading(monkeypatch,capsys):
    from scripts import run_m4_acceptance, run_m4
    args=['run_m4_acceptance.py','--candidate','m3','--low-full','--three-frequency',
        '--start-floor-ms','25','--critical-wait','--critical-wait-phases',
        '--first-refresh-step','30','--phase-refresh','--dry-run']
    monkeypatch.setattr(sys,'argv',args)
    assert run_m4_acceptance.main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['groups_to_run']==['q30','m3'] and plan['episodes']==24 and plan['model_loads']==2
    assert plan['lean_capture'] and plan['common_low_tail_seconds']==3.
    assert plan['candidate_clock_arguments']==['--low-full','--three-frequency','--start-floor-ms','25']
    for group in ('q30','m3'):
        flags=plan['candidate_clock_arguments'] if group=='m3' else []
        monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group',group,
            '--common-low-tail','--lean-continue','--dry-run',*flags])
        assert run_m4.main()==0
        child=json.loads(capsys.readouterr().out)
        assert child['episodes']==12 and child['lean_continue'] and child['common_low_tail_seconds']==3.
        assert child['three_frequency']==(group=='m3')


def test_render_noise_does_not_mask_physics_or_visible_image_change():
    from perceptbudget.m4_acceptance import render_parity
    case=dict(physical_max_abs=0.,state_max_abs=0.,success_equal=True,pixel_max_abs=1,max_changed_pixels=8)
    assert render_parity(case,2*256*256)['passed']
    assert render_parity(case,2*256*256)['changed_pixel_limit']==13
    for field,value in [('physical_max_abs',1e-12),('state_max_abs',1e-12),('success_equal',False),
                        ('pixel_max_abs',2),('max_changed_pixels',14)]:
        assert not render_parity(dict(case,**{field:value}),2*256*256)['passed']


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


def test_task_local_prefixes_do_not_depend_on_task0_history():
    from perceptbudget.m4_acceptance import visual_prefixes
    full=list(range(653,670))
    for valid in (335,340,344,368):
        lengths,coverage=visual_prefixes(full,valid,48)
        assert lengths==full+[valid]
        assert coverage['fast_state_enabled'] is False
    for valid in (320,369):
        with pytest.raises(ValueError):visual_prefixes(full,valid,48)


@pytest.mark.parametrize('candidate',['joint','m3'])
def test_report_reads_actual_q_and_does_not_call_failure_energy_savings(tmp_path,candidate):
    sources={}
    for group,q in [('q30',30),(candidate,5)]:
        parent=tmp_path/group;run=parent/'episode';(run/'rtc').mkdir(parents=True)
        (run/'run.json').write_text(json.dumps({'config_snapshot':{'candidate_q':[q]}}))
        (run/'summary.json').write_text(json.dumps({'episode':{
            'real_seconds':1,'including_drain_seconds':1.1,
            'including_drain_power':{'total_energy_mj':1000 if group=='q30' else 500},'underflow_steps':[]}}))
        request=dict(observation_raw_state=[1],model_state=[2],actions_raw=[[3]],rtc_solver_inputs={'noise':[[4]]})
        if group!='q30':request['actions_raw']=[[3.01]]
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
    baseline_file=sources['q30']/'summary.json'
    baseline=json.loads(baseline_file.read_text())
    baseline['episodes'].append(dict(baseline['episodes'][0],suite='libero_object',run='/excluded-baseline-task'))
    baseline_file.write_text(json.dumps(baseline))
    subset=tmp_path/'subset';subset.mkdir()
    screened=summarize(subset,sources,task_plan=[['libero_spatial',0]])
    assert len(screened['rows'])==2 and len(screened['pairs'])==1
    with pytest.raises(ValueError,match='task list'):
        summarize(subset,sources,task_plan=[['libero_spatial',3]])


def test_m3_comparator_uses_original_d_joint_with_no_m4_or_fast():
    from perceptbudget.m4_acceptance import configure_m3
    cfg=dict(m4_enabled=True,m4_settings={'not_used':True},m4_fast_prompt_capacity=20)
    settings={'timing':{}}
    configure_m3(cfg,settings)
    assert cfg['m4_enabled'] is False and 'm4_settings' not in cfg
    assert not cfg['m2_visual_fast_enabled'] and not cfg['m3_dvac_enabled']
    assert cfg['m3_action_events_enabled'] and cfg['m3_dvfs_policy']=='joint'
    assert cfg['m3_formal_group']=='D_joint' and cfg['m3_scheduler_mode']=='deadline_dvfs'
    assert not cfg['m3_hold_high_clocks']
    assert settings['timing']['safety_margin_seconds']==.025
    assert settings['timing']['maximum_action_offset']==50
    assert settings['timing']['maximum_observation_age_seconds'] is None


def test_resident_environment_only_loads_validated_binary(tmp_path,monkeypatch):
    from perceptbudget.m4_acceptance import precompiled_environments
    loaded=[]
    class Sim:
        def __init__(self,model):self.model=model
        @classmethod
        def from_xml_string(cls,xml):
            raise AssertionError('Resident worker must not compile XML')
    mujoco=ModuleType('mujoco');mujoco.__version__='test'
    mujoco.MjModel=SimpleNamespace(from_binary_path=lambda p: loaded.append(p) or 'binary-model')
    binding=ModuleType('robosuite.utils.binding_utils');binding.MjSim=Sim
    package=ModuleType('robosuite');package.__path__=[]
    utils=ModuleType('robosuite.utils');utils.__path__=[]
    monkeypatch.setitem(sys.modules,'mujoco',mujoco)
    monkeypatch.setitem(sys.modules,'robosuite',package)
    monkeypatch.setitem(sys.modules,'robosuite.utils',utils)
    monkeypatch.setitem(sys.modules,'robosuite.utils.binding_utils',binding)
    report=dict(status='passed',frozen=True,version='test',cases=[dict(suite='libero_10',task_id=0,passed=True)])
    (tmp_path/'environment_cache.json').write_text(json.dumps(report))
    folder=tmp_path/'environment_models';folder.mkdir()
    key=hashlib.sha256(b'test\0xml').hexdigest();(folder/(key+'.mjb')).write_bytes(b'model')
    with precompiled_environments(tmp_path,[['libero_10',0]],{}) as cache:
        assert Sim.from_xml_string('xml').model=='binary-model'
        with pytest.raises(RuntimeError,match='Unprepared environment'):
            Sim.from_xml_string('unexpected xml')
        assert cache['preparation_process_exited']
    assert len(loaded)==1
    assert json.loads((tmp_path/'environment_cache.json').read_text())['resident_binary_loads']==1
    with pytest.raises(ValueError,match='matching environment'):
        with precompiled_environments(tmp_path,[['libero_10',3]],{}):pass


def test_reuse_environment_cache_requires_exact_passed_manifest_and_links_models(tmp_path):
    from perceptbudget.m4_acceptance import copy_precompiled_environment_cache
    source=tmp_path/'source';source.mkdir();destination=tmp_path/'destination';destination.mkdir()
    manifest={'fingerprint':'same'}
    (source/'run.json').write_text(json.dumps({'config_snapshot':{
        'm4_acceptance_group':'m3_fast','m4_acceptance_manifest':manifest}}))
    cases=[dict(suite='libero_spatial',task_id=0,passed=True),
           dict(suite='libero_object',task_id=3,passed=True)]
    (source/'environment_cache.json').write_text(json.dumps(dict(
        status='passed',frozen=True,cases=cases)))
    models=source/'environment_models';models.mkdir()
    for i in range(2):(models/(str(i)+'.mjb')).write_bytes(b'model'+bytes([i]))
    result=copy_precompiled_environment_cache(source,destination,
        [('libero_spatial',0),('libero_object',3)],
        {'m4_acceptance_group':'m3_fast','m4_acceptance_manifest':manifest})
    assert result['models']==2 and result['hardlinks']==2 and result['copies']==0
    assert (destination/'environment_cache.json').is_file()
    assert (destination/'environment_models'/'0.mjb').stat().st_ino==(models/'0.mjb').stat().st_ino
    with pytest.raises(ValueError,match='exactly these tasks'):
        copy_precompiled_environment_cache(source,tmp_path/'wrong-tasks',
            [('libero_spatial',0)],{'m4_acceptance_group':'m3_fast','m4_acceptance_manifest':manifest})
    with pytest.raises(ValueError,match='manifest'):
        copy_precompiled_environment_cache(source,tmp_path/'wrong-manifest',
            [('libero_spatial',0),('libero_object',3)],
            {'m4_acceptance_group':'m3_fast','m4_acceptance_manifest':{'fingerprint':'other'}})


def test_calibrated_start_requires_unchanged_state_and_task():
    import numpy as np
    from perceptbudget.m4_acceptance import calibrated_start
    physics=np.asarray([1.,2.])
    env=SimpleNamespace(env=SimpleNamespace(control_freq=20,sim=SimpleNamespace(
        get_state=lambda: SimpleNamespace(flatten=lambda: physics.copy()))))
    task=dict(suite='libero_10',task_id=0)
    prepared=dict(task,observation={'image':'unchanged'},physics=physics.copy())
    observation,hz=calibrated_start(env,task,prepared,np)
    assert observation is prepared['observation'] and hz==20
    with pytest.raises(ValueError):calibrated_start(env,dict(task,task_id=3),prepared,np)
    physics[0]+=1
    with pytest.raises(ValueError):calibrated_start(env,task,prepared,np)


def test_resume_keeps_failures_and_only_returns_missing_tasks(tmp_path,monkeypatch):
    import scripts.audit_rtc_system as auditing
    import perceptbudget.m4_acceptance as acceptance
    from perceptbudget.m4_planner import POLICY_REVISION
    cfg=dict(FROZEN,checkpoint_path='/checkpoint',norm_stats_path='/stats',m4_settings={'fixed':True},m4_policy_revision=POLICY_REVISION)
    fixed=manifest(cfg)
    (tmp_path/'run.json').write_text(json.dumps({'config_snapshot':dict(cfg,m4_acceptance_group='joint',
        m4_acceptance_manifest=fixed,m4_fast_prompt_capacity=20,m4_cover_close_slowdown=False)}))
    episodes=[]
    for i,(suite,tid) in enumerate(fixed['tasks'][:9]):
        run=tmp_path/str(i);run.mkdir();(run/'summary.json').write_text(json.dumps({'status':'completed'}))
        episodes.append(dict(suite=suite,task_id=tid,view='full',run=str(run),outcome='timeout' if i==3 else 'success'))
    (tmp_path/'recording.json').write_text(json.dumps({'episodes':episodes,'model_loads':1}))
    monkeypatch.setattr(auditing,'audit',lambda run: {'passed':True})
    monkeypatch.setattr(acceptance,'clock_audit',lambda run:{'passed':True})
    resumed=acceptance.resume_joint(tmp_path,cfg)
    assert len(resumed['episodes'])==9 and resumed['episodes'][3]['outcome']=='timeout'
    assert resumed['remaining_tasks']==[['libero_10',0],['libero_10',3],['libero_10',6]]
    monkeypatch.setattr(acceptance,'clock_audit',lambda run:{'passed':False})
    with pytest.raises(ValueError,match='audit'):acceptance.resume_joint(tmp_path,cfg)
    meta=json.loads((tmp_path/'run.json').read_text());meta['config_snapshot'].pop('m4_policy_revision')
    (tmp_path/'run.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='version changed'):acceptance.resume_joint(tmp_path,cfg)
