import json
import sys

import pytest

from perceptbudget.ordinary_fast import choose,FEEDBACK_POLICY
from tests.test_three_frequency import ordinary_costs


def decision():
    return dict(decision_monotonic=1.,deadline=2.,margin_seconds=.025,latest_start=1.4,
        compute_profile='gpu_emc',service_seconds=.5,rtc_delay_steps=10,
        estimated_control_period_seconds=.05,reason='buffer',deadline_already_unreachable=False,
        start_constraint='service_deadline')


def test_energy_fast_does_not_resurrect_predictable_motion_phases():
    d=decision();args=dict(step=30,previous_mode='Full',phases=[dict(start_step=28,stop_step=35)],
        events=[dict(kind='turn',target_step=32)],costs=ordinary_costs(),fast_delay=5)
    assert choose(d,**args)['mode']=='Full'
    r=choose(d,**args,policy=FEEDBACK_POLICY)
    assert r['mode']=='Fast' and r['ordinary']
    assert all(d[k]==v for k,v in r['full_schedule'].items())


def test_energy_fast_keeps_gripper_margin_required_feedback_and_full_after_fast():
    d=decision();args=dict(step=30,previous_mode='Full',phases=[],events=[],
        costs=ordinary_costs(),fast_delay=5,policy=FEEDBACK_POLICY)
    assert choose(d,**args)['mode']=='Fast'
    assert choose(dict(d,reason='action_event'),**args)['mode']=='Full'
    assert choose(d,**dict(args,events=[dict(kind='gripper_open',target_step=34)]))['mode']=='Full'
    assert choose(d,**dict(args,previous_mode='Fast'))['mode']=='Full'
    assert choose(d,**dict(args,fast_delay=11))['mode']=='Full'


def test_ordinary_fast_keeps_finite_gripper_operations_full_without_blocking_motion():
    from perceptbudget.ordinary_fast import FEEDBACK_POLICY_V2
    d=decision();phase=dict(start_step=20,stop_step=31,anchors=[dict(kind='gripper_open',offset=4)])
    args=dict(step=30,previous_mode='Full',phases=[phase],events=[],costs=ordinary_costs(),fast_delay=5)
    assert choose(d,**args,policy=FEEDBACK_POLICY_V2)['mode']=='Fast'
    result=choose(d,**args,policy=FEEDBACK_POLICY)
    assert result['mode']=='Full' and result['protected_gripper_phases']==[phase]
    assert result['full_schedule']==choose(d,**args,policy=FEEDBACK_POLICY_V2)['full_schedule']
    assert choose(d,**dict(args,step=31),policy=FEEDBACK_POLICY)['mode']=='Fast'
    motion=dict(phase,anchors=[dict(kind='turn',offset=4)])
    assert choose(d,**dict(args,phases=[motion]),policy=FEEDBACK_POLICY)['mode']=='Fast'


def test_idle_threshold_does_not_override_fast_pair_labels():
    from perceptbudget.device_quality import frequency_pair_views
    c=dict(m3_ordinary_fast_pair=True,m3_idle_min_ms=50,m3_three_frequency=True)
    assert frequency_pair_views(c,0)==('ordinary_full','ordinary_fast')
    assert frequency_pair_views(c,1)==('ordinary_fast','ordinary_full')


def test_simple_alternation_keeps_order_even_at_key_phases():
    from perceptbudget.ordinary_fast import ALTERNATING_POLICY
    phase=dict(start_step=20,stop_step=40,anchors=[dict(kind='gripper_close',offset=4)])
    d=decision();d['reason']='action_event'
    args=dict(step=30,phases=[phase],events=[dict(kind='gripper_close',target_step=31)],
        costs=ordinary_costs(),fast_delay=12,policy=ALTERNATING_POLICY)
    before=dict(d)
    assert choose(d,previous_mode='Full',**args)['mode']=='Fast'
    assert choose(d,previous_mode='Fast',**args)['mode']=='Full'
    assert d==before


def test_gripper_full_window_covers_reachable_grasp_and_preserves_schedule():
    from perceptbudget.ordinary_fast import ALTERNATING_NO_STATE_POLICY
    d=decision()
    d['critical_forecast']=dict(result=dict(events=[dict(kind='gripper_close',target_step=45)]))
    args=dict(step=30,previous_mode='Full',phases=[],events=[],costs=ordinary_costs(),
        fast_delay=5,policy=ALTERNATING_NO_STATE_POLICY)
    original=choose(d,**args)
    guarded=choose(d,**args,gripper_full_margin_steps=5)
    assert original['mode']=='Fast' and 'gripper_full_guard' not in original
    assert guarded['mode']=='Full' and guarded['reason']=='full_gripper_service_window'
    assert guarded['full_schedule']==original['full_schedule']
    assert guarded['gripper_full_guard']['arrival_step']==41
    assert guarded['protected_gripper_phases']==[]
    assert choose(d,**dict(args,previous_mode='Fast'),gripper_full_margin_steps=5)['mode']=='Full'


@pytest.mark.parametrize('kind,target',[
    ('gripper_close',29),('gripper_close',40),('gripper_close',47),('turn',43),('slowdown',43)])
def test_gripper_full_window_does_not_expand_to_committed_prefix_or_other_motion(kind,target):
    from perceptbudget.ordinary_fast import ALTERNATING_NO_STATE_POLICY
    d=decision();d['critical_forecast']=dict(result=dict(events=[dict(kind=kind,target_step=target)]))
    r=choose(d,step=30,previous_mode='Full',phases=[],events=[],costs=ordinary_costs(),
        fast_delay=5,policy=ALTERNATING_NO_STATE_POLICY,gripper_full_margin_steps=5)
    assert r['mode']=='Fast' and not r['gripper_full_guard']['overrode_fast']


def test_gripper_full_margin_cli_is_explicit_and_keeps_the_controller(monkeypatch,capsys):
    from scripts import run_m4
    from perceptbudget import ordinary_fast
    monkeypatch.setattr(ordinary_fast,'load_costs',lambda *args:ordinary_costs())
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','m3','--task-indices','0',
        '--ordinary-fast','--fast-alternate','--fast-no-state','--ordinary-fast-view','pool2_agent',
        '--ordinary-fast-costs','/tmp/measured.json','--energy-refresh','--ordinary-fast-positions',
        '--low-full','--three-frequency','--idle-min-ms','50','--start-floor-ms','25',
        '--full-gripper-margin-steps','5','--dry-run'])
    assert run_m4.main()==0
    r=json.loads(capsys.readouterr().out)
    assert r['full_gripper_margin_steps']==5 and r['full_per_fast']==1
    assert r['feedback_minimum_steps']==20 and r['H']==50 and r['N']==5
    monkeypatch.setattr(sys,'argv',['run_m4.py','--full-gripper-margin-steps','5','--dry-run'])
    with pytest.raises(SystemExit):run_m4.main()


@pytest.mark.parametrize('ratio,expected',[(1,['Full','Fast']*4),(2,['Full','Full','Fast']*3)])
def test_strict_cadence_includes_bootstrap_and_only_advances_on_commit(monkeypatch,ratio,expected):
    from perceptbudget.three_frequency import ThreeFullRuntime
    from perceptbudget.m3_low_full import LowFullRuntime
    from perceptbudget.ordinary_fast import ALTERNATING_NO_STATE_POLICY
    from tests.test_three_frequency import config
    monkeypatch.setattr(LowFullRuntime,'committed',lambda *args:None)
    runtime=ThreeFullRuntime(dict(config(),m3_ordinary_fast=True,ordinary_fast_costs=ordinary_costs(),
        m3_ordinary_fast_policy=ALTERNATING_NO_STATE_POLICY,m3_fast_full_per_fast=ratio))
    observed=[]
    for pid in range(len(expected)):
        if pid==0:
            mode='Full'
        else:
            kwargs=dict(step=pid,previous_mode=runtime.ordinary_previous,phases=[],events=[],
                costs=ordinary_costs(),fast_delay=5,policy=ALTERNATING_NO_STATE_POLICY,
                full_per_fast=runtime.ordinary_full_per_fast,full_streak=runtime.ordinary_full_streak)
            first=choose(decision(),**kwargs);second=choose(decision(),**kwargs)
            assert first['mode']==second['mode']
            mode=first['mode']
        observed.append(mode)
        request=dict(prediction_id=pid,m3_fast_choice=dict(mode=mode))
        runtime.committed(request,dict(commit_monotonic=float(pid)))
        if pid==0:
            runtime.committed(request,dict(commit_monotonic=0.))
            assert runtime.ordinary_full_streak==1
    assert observed==expected


def test_cadence_pair_rotates_order_and_keeps_both_fast_pipelines(monkeypatch,capsys):
    from perceptbudget.device_quality import frequency_pair_views,configure_frequency_variant,camera_view_for_condition
    from perceptbudget import ordinary_fast
    from scripts import run_m4
    cfg=dict(m3_fast_cadence_pair=True,m3_ordinary_fast_pair=True)
    assert frequency_pair_views(cfg,0)==('cadence_1full','cadence_2full')
    assert frequency_pair_views(cfg,1)==('cadence_2full','cadence_1full')
    for label,ratio in [('cadence_1full',1),('cadence_2full',2)]:
        child=dict(cfg);configure_frequency_variant(child,label)
        assert child['m3_ordinary_fast'] and child['m3_fast_full_per_fast']==ratio
        assert camera_view_for_condition(label)=='full'
    monkeypatch.setattr(ordinary_fast,'load_costs',lambda *args:ordinary_costs())
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','m3','--all-tasks',
        '--task-indices','0','10','30','--ordinary-fast','--fast-alternate','--fast-no-state',
        '--ordinary-fast-view','pool2_agent','--ordinary-fast-costs','/tmp/measured.json',
        '--energy-refresh','--ordinary-fast-positions','--low-full','--three-frequency',
        '--fast-cadence-pair','--dry-run'])
    assert run_m4.main()==0
    result=json.loads(capsys.readouterr().out)
    assert result['episodes']==6 and result['fast_cadence_pair'] and result['Fast']=='pool2_agent_no_state'


def test_alternating_pool_retains_current_state_in_real_context_inputs():
    from perceptbudget.rtc_visual_alternation import configure_explicit_inputs,FreshViewRTCAdapter
    cfg=dict(m3_ordinary_fast_context=True,m3_fast_alternation=True,
        m3_ordinary_fast_visual_mode='selective464',m2_fast_no_state=True)
    configure_explicit_inputs(cfg)
    adapter=object.__new__(FreshViewRTCAdapter);adapter.config=cfg
    snapshot=dict(images=[object(),object()],state=object())
    result=adapter.context_inputs(snapshot,'selective464')
    assert cfg['m2_fast_no_state'] is False
    assert result['state'] is snapshot['state'] and result['images']==snapshot['images']


@pytest.mark.parametrize('tokens,positions',[(320,False),(464,False),(464,True)])
def test_energy_fast_cli_forces_request_only_images_and_keeps_frozen_controller(monkeypatch,capsys,tokens,positions):
    from scripts import run_m4
    args=['run_m4.py','--acceptance-group','m3','--task-indices','0','--energy-refresh',
        '--ordinary-fast','--ordinary-fast-tokens',str(tokens),'--ordinary-fast-pair','--low-full','--three-frequency',
        '--idle-min-ms','50','--start-floor-ms','25','--dry-run']
    if positions:args.append('--ordinary-fast-positions')
    monkeypatch.setattr(sys,'argv',args)
    assert run_m4.main()==0
    r=json.loads(capsys.readouterr().out)
    assert r['request_only_images'] and r['lean_continue'] and r['ordinary_fast']
    assert r['feedback_policy']=='overlap_pacing_v2' and r['feedback_minimum_steps']==20
    assert r['H']==50 and r['N']==5 and r['max_refresh_interval']==50
    assert r['episodes']==2 and r['Fast']=='selective'+str(tokens)+'_no_state'
    assert r['idle_min_ms']==50 and r['three_frequency']
    assert r['ordinary_fast_positions'] is positions


def test_fast_capture_cannot_silently_revert_to_unused_inference_images(monkeypatch):
    from scripts import run_m4
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','m3','--ordinary-fast',
        '--low-full','--three-frequency','--full-capture','--dry-run'])
    with pytest.raises(SystemExit):run_m4.main()


def test_mild_fast_uses_shorter_real_encoder_and_exact_geometry_bank(monkeypatch):
    from types import SimpleNamespace
    from perceptbudget.rtc_visual_alternation import visual_geometry,FreshViewRTCAdapter
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.selective_pipeline import pipeline_type
    from perceptbudget.m4_acceptance import visual_prefixes
    assert visual_geometry('selective464')['visual_tokens']==464
    class Base:
        def _allocate_buffers(self):return self.encoder_seq_len,self.total_kv
    pipeline=pipeline_type(Base,visual_tokens=464)()
    pipeline.num_views=2;pipeline.vision_pool_factor=2;pipeline.max_prompt_len=32;pipeline.chunk_size=50
    assert pipeline._allocate_buffers()==(496,546)
    lengths,coverage=visual_prefixes(list(range(646,663)),477,32,
        fast_visual_tokens=464,fast_mode='selective464')
    prepared=[]
    def prepare(self,family,progress=None):
        prepared.append(family)
        self.step_bank=SimpleNamespace(entries={n:(n,n) for n in family},fallback='frozen')
        return dict(prepare_seconds=0.)
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'prepare_step_bank',prepare)
    adapter=object.__new__(FreshViewRTCAdapter)
    adapter.position_policy=None
    adapter.config=dict(m2_visual_fast_mode='selective464',m2_visual_prefix_coverage=coverage,
        source_prefix_coverage_rule='h50_state8_tokenizer_bounds_v1')
    adapter.torch=SimpleNamespace(cuda=SimpleNamespace(graph_pool_handle=object))
    adapter.prepare_step_bank(lengths)
    assert prepared==[list(range(646,663)),[477]]


def test_mild_fast_rejects_320_costs_before_any_model_load(tmp_path):
    from perceptbudget.ordinary_fast import load_costs
    config=dict(H=50,N=5,precision='bfloat16',checkpoint_provenance_revision='frozen',rtc_beta=5.,
        m3_ordinary_fast_visual_mode='selective464')
    (tmp_path/'run.json').write_text(json.dumps(dict(config_snapshot=config)))
    path=tmp_path/'costs.json';path.write_text(json.dumps(dict(status='completed',restored=True)))
    with pytest.raises(ValueError,match='geometry/state mismatch'):load_costs(path,config)


def test_stateful_wrist_binds_latest_state_and_only_the_current_wrist_image():
    from perceptbudget.rtc_visual_alternation import configure_explicit_inputs,FreshViewRTCAdapter
    cfg=dict(m3_ordinary_fast_context=True,m3_ordinary_fast_visual_mode='wrist',m2_fast_no_state=True)
    configure_explicit_inputs(cfg)
    adapter=object.__new__(FreshViewRTCAdapter);adapter.config=cfg
    snapshot=dict(images=['external-now','wrist-now'],state=[.2]*32,raw_state=[.1]*8)
    fast=adapter.context_inputs(snapshot,'wrist')
    assert fast['images']==['wrist-now'] and fast['state'] is snapshot['state']
    assert adapter.context_inputs(snapshot,'full')['images']==snapshot['images']
    assert not cfg['m2_fast_no_state']


def test_stateful_wrist_costs_reject_silently_omitted_state(tmp_path):
    from perceptbudget.ordinary_fast import load_costs
    cfg=dict(H=50,N=5,precision='bfloat16',checkpoint_provenance_revision='frozen',rtc_beta=5.,
        m3_ordinary_fast_visual_mode='wrist')
    (tmp_path/'run.json').write_text(json.dumps(dict(config_snapshot=cfg)))
    path=tmp_path/'costs.json'
    path.write_text(json.dumps(dict(status='completed',restored=True,
        fast_input_contract=dict(visual_mode='wrist',visual_tokens=256,state_enabled=False))))
    with pytest.raises(ValueError,match='geometry/state mismatch'):load_costs(path,cfg)


def test_original_coordinates_shift_fast_actions_and_leave_full_inputs_exact():
    import torch
    from types import SimpleNamespace
    from perceptbudget.selective_pipeline import align_expert_positions
    bridge=SimpleNamespace(forward=lambda **kw:kw['position_ids'])
    expert=SimpleNamespace(suffix=SimpleNamespace(paligemma_with_expert=bridge))
    policy=align_expert_positions(expert,464,32)
    def arguments(prefix):
        return dict(attention_mask=torch.zeros(1,1,50,prefix+50),
            inputs_embeds=[None,torch.zeros(1,50,1)],
            position_ids=torch.arange(prefix,prefix+50)[None])
    full=arguments(656);fast=arguments(477)
    assert bridge.forward(**full) is full['position_ids']
    assert torch.equal(bridge.forward(**fast),fast['position_ids']+48)
    assert bridge.forward(**full) is full['position_ids']
    assert policy['full_position_offset']==0 and policy['rtc_sampler_changed'] is False
    with pytest.raises(ValueError):bridge.forward(**arguments(400))


def test_original_position_table_does_not_restore_full_encoder_compute():
    from perceptbudget.selective_pipeline import pipeline_type
    class Base:
        def _allocate_buffers(self):pass
        def _build_rope_table(self):self.table_sequence=self.encoder_seq_len
    p=pipeline_type(Base,visual_tokens=464,preserve_positions=True)()
    p.num_views=2;p.vision_pool_factor=2;p.max_prompt_len=32;p.chunk_size=50
    p._allocate_buffers();p._build_rope_table()
    assert p.table_sequence==544 and p.encoder_seq_len==496 and p.total_kv==546
    assert p._pb_original_visual_positions is True


def test_stateful_original_positions_use_explicit_geometry_for_overlapping_lengths():
    import torch
    from types import SimpleNamespace
    from perceptbudget.selective_pipeline import align_expert_positions
    bridge=SimpleNamespace(forward=lambda **kw:kw['position_ids'])
    expert=SimpleNamespace(suffix=SimpleNamespace(paligemma_with_expert=bridge))
    active=dict(fast=False)
    align_expert_positions(expert,464,200,active_fast=lambda:active['fast'])
    args=dict(attention_mask=torch.zeros(1,1,50,656+50),inputs_embeds=[None,torch.zeros(1,50,1)],
              position_ids=torch.arange(656,706)[None])
    assert bridge.forward(**args) is args['position_ids']
    active['fast']=True
    assert torch.equal(bridge.forward(**args),args['position_ids']+48)
    active['fast']=False
    assert bridge.forward(**args) is args['position_ids']


def test_positioned_fast_rejects_unpositioned_cost_table(tmp_path):
    from perceptbudget.ordinary_fast import load_costs
    cfg=dict(H=50,N=5,precision='bfloat16',checkpoint_provenance_revision='frozen',rtc_beta=5.,
        m3_ordinary_fast_visual_mode='selective320',m3_ordinary_fast_positions=True)
    (tmp_path/'run.json').write_text(json.dumps(dict(config_snapshot=cfg)))
    path=tmp_path/'costs.json';path.write_text(json.dumps(dict(status='completed',restored=True)))
    with pytest.raises(ValueError,match='position policy mismatch'):load_costs(path,cfg)
