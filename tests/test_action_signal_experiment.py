"""Pure CPU checks: alignment, budget and optional runtime integration."""
import json
import sys

import pytest

from perceptbudget.action_events import EventConfig
from perceptbudget.action_signal_experiment import (
    VERSION, ALIGNED_VERSION, PAIRED_VERSION, extract, prepare, apply_candidate, observe_execution,
)
from tests.test_ordinary_turns import runtime


def action(rotation=0., grip=-1.):
    return [0.4, 0., 0., rotation, 0., 0., grip]


def settings():
    return dict(version=VERSION, minimum_request_steps=20, minimum_mean_steps=22,
        thresholds=dict(translation_change=.3, rotation_change=.1,
                        translation_disagreement=.3, rotation_disagreement=.1))


def packet(origin=0, count=1):
    # The old prediction ends at25. Current overlap disagrees, then a rotation
    # change at36 needs feedback. Do not pretend the old chunk reaches36.
    actions=[action(.3)]*36+[action(.7)]*14
    return prepare(actions=actions,origin=origin,source_step=origin+8,
        history={},reference=dict(origin=origin-25,actions=[action()]*50),
        config=EventConfig(),settings=settings(),period=.05,response_seconds=.5,
        margin_seconds=.025,prediction_count=count)


def test_reference_uses_absolute_execution_steps_not_chunk_array_indices():
    old=[action(i*.01) for i in range(50)]
    current=old[20:]+[action(9.)]*20
    rows=extract(current,origin=20,source_step=28,history={},
        reference=dict(origin=0,actions=old),config=EventConfig())
    overlap=[r for r in rows if r['reference_overlap']]
    assert overlap and all(r['rotation_disagreement']==0. for r in overlap)
    assert all(r['rotation_disagreement'] is None for r in rows if r['target_step']>=49)
    assert rows[0]['target_step']==28


def test_executed_history_replaces_the_new_chunks_expired_prefix():
    current=[action(8.)]*8+[action(.3)]*42
    rows=extract(current,origin=0,source_step=8,history={6:action(.3),7:action(.3)},
                 reference=None,config=EventConfig())
    assert rows[0]['rotation_change']==0.
    assert not rows[0]['reference_overlap']


def test_two_families_can_use_real_overlap_without_extending_old_horizon():
    p=packet();r=p['review']
    assert r['candidate']['kind']=='action_revision'
    assert r['candidate']['target_step']>=r['earliest_new_target_step']
    assert r['overlap_evidence']['last_step']==23
    assert r['candidate']['target_step']>23
    assert len(r['candidate']['evidence']['geometry'])>0
    assert len(r['candidate']['evidence']['disagreement'])>0


def test_extra_cues_do_not_spend_an_already_exhausted_refresh_budget():
    assert packet(origin=100,count=10)['review']['candidate'] is None
    p=packet();r=p['review']
    assert r['earliest_new_request_step']==22
    assert r['earliest_new_target_step']==22+13  # Full+eta, ACK and rounding


def test_budget_opt_out_keeps_signal_thresholds_but_admits_imminent_changes():
    cfg=dict(settings(),refresh_budget_enabled=False)
    actions=[action(.3)]*10+[action(.7)]*40
    p=prepare(actions=actions,origin=100,source_step=108,history={},
        reference=dict(origin=75,actions=[action()]*50),config=EventConfig(),settings=cfg,
        period=.05,response_seconds=.5,margin_seconds=.025,prediction_count=10)
    r=p['review']
    assert r['refresh_budget_enabled'] is False
    assert r['budget_rejected']==0 and r['candidate']['target_step']<120
    assert len(apply_candidate([],packet=p,step=108,end=150,prediction_id=9,
        covered=[],tolerance=5))==1
    assert cfg['thresholds']==settings()['thresholds']


def test_constant_rotation_or_motion_alone_does_not_trigger():
    p=prepare(actions=[action(.3)]*50,origin=0,source_step=8,history={},
        reference=dict(origin=-25,actions=[action()]*50),config=EventConfig(),
        settings=settings(),period=.05,response_seconds=.5,margin_seconds=.025,prediction_count=1)
    assert p['review']['candidate'] is None


def test_past_and_covered_cues_do_not_slide_or_repeat():
    p=packet();event=p['review']['candidate'];at=event['target_step']
    kwargs=dict(packet=p,end=50,prediction_id=1,tolerance=5)
    assert apply_candidate([],step=8,covered=[],**kwargs)==[]
    assert apply_candidate([],step=22,covered=[],period=.02,**kwargs)==[]
    assert len(apply_candidate([],step=22,covered=[],**kwargs))==1
    assert len(apply_candidate([],step=at-2,covered=[],**kwargs))==1
    assert apply_candidate([],step=at+1,covered=[],**kwargs)==[]
    assert apply_candidate([],step=22,covered=[event],**kwargs)==[]
    earlier=[dict(kind='gripper_close',target_step=at-2)]
    assert apply_candidate(earlier,step=22,covered=[],**kwargs)==earlier


def test_runtime_opt_in_keeps_baseline_forecast_and_logs_one_packet(tmp_path):
    r=runtime(50);r.feedback_policy='overlap_pacing_v2';r.feedback_minimum=20
    r.action_signal_settings=settings();r.action_signal_directory=tmp_path
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=[action()]*50),
                dict(commit_step=0,commit_monotonic=0.))
    actions=[action(.3)]*38+[action(.7)]*12
    r.committed(dict(prediction_id=1,observation_step=20,observation_monotonic=1.,actions=actions),
                dict(commit_step=28,commit_monotonic=1.4))
    queue={20+i:(a,1,20,i) for i,a in enumerate(actions)}
    a=r.decide(now=1.4,step=28,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    b=r.decide(now=1.45,step=29,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert a['action_signal_review']==b['action_signal_review']
    assert len((tmp_path/'action_signals.jsonl').read_text().splitlines())==1
    assert a['critical_forecast']==b['critical_forecast']
    assert 'action_revision' not in {e['kind'] for e in a['critical_forecast']['result']['events']}
    c=r.decide(now=2.2,step=44,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    assert c['candidate_event_kinds']==['action_revision'] and c['action_signal_added']
    assert c['suppressed_event_count']==0
    record=json.loads((tmp_path/'action_signals.jsonl').read_text())
    assert 'features' in record and 'features' not in a['action_signal_review']


def test_baseline_has_no_added_signal_or_log(tmp_path):
    r=runtime(50);r.action_signal_directory=tmp_path
    actions=[action()]*50
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions),
                dict(commit_step=0,commit_monotonic=0.))
    d=r.decide(now=1.,step=20,queue={i:(a,0,0,i) for i,a in enumerate(actions)},
               rows=[],inflight=False,rtc_delay_steps=10)
    assert 'action_signal_review' not in d
    assert not list(tmp_path.iterdir())


def test_execution_feedback_records_only_robot_state():
    rows=[dict(step=3,before=dict(raw_state=[0.]*8,contacts=['never used']),
               after=dict(raw_state=[.1,0,0,0,0,0,.01,-.01]))]
    r=observe_execution(rows)
    assert r['translation_distance']==.1 and r['gripper_after']==[.01,-.01]
    assert 'contacts' not in r and 'success' not in r


def test_cli_explicit_settings_and_q10_separation(tmp_path, monkeypatch, capsys):
    from scripts import run_m4
    path=tmp_path/'signals.json';path.write_text(json.dumps(settings()))
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','m3',
        '--energy-refresh','--action-signals',str(path),'--dry-run'])
    assert run_m4.main()==0
    result=json.loads(capsys.readouterr().out)
    assert result['action_signal_settings']==settings()
    assert result['H']==50 and result['N']==5
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','q30',
        '--fixed-q','10','--action-signals',str(path),'--dry-run'])
    with pytest.raises(SystemExit):run_m4.main()
    capsys.readouterr()
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','q30',
        '--fixed-q','10','--dry-run'])
    assert run_m4.main()==0
    assert json.loads(capsys.readouterr().out)['action_signal_settings'] is None


def aligned_packet(actions, reference, history=None, version=ALIGNED_VERSION):
    return prepare(actions=actions, origin=0, source_step=8, history=history or {},
        reference=reference, config=EventConfig(),
        settings=dict(settings(), version=version, refresh_budget_enabled=False),
        period=.05, response_seconds=.5, margin_seconds=.025, prediction_count=1)


@pytest.mark.parametrize('new', [action(.7), action(grip=1.)])
def test_v2_handoff_change_is_not_a_new_plan_change(new):
    old=[action()]*50; current=[new]*50
    history={6:action(), 7:action()}
    v1=aligned_packet(current, dict(origin=0, actions=old), history, version=VERSION)
    v2=aligned_packet(current, dict(origin=0, actions=old), history)
    assert v1['review']['candidate']['target_step']==8
    assert v2['review']['candidate'] is None
    assert all(f['translation_change'] is None and f['rotation_change'] is None
               for f in v2['features'][:2])


def test_v2_online_and_calibration_features_ignore_the_same_handoff_prefix():
    current=[action(.3)]*18+[action(.7)]*32
    reference=dict(origin=0, actions=[action()]*50)
    offline=aligned_packet(current, reference)
    online=aligned_packet(current, reference, {6:action(9.),7:action(9.)})
    assert offline==online
    assert online['review']['candidate'] is not None


def test_v2_distant_disagreement_does_not_justify_an_earlier_change():
    current=[action()]*18+[action(.7)]*32
    old=current[:40]+[action()]*10
    v1=aligned_packet(current, dict(origin=0, actions=old), version=VERSION)
    v2=aligned_packet(current, dict(origin=0, actions=old))
    assert v1['review']['candidate']['target_step']<20
    assert v2['review']['overlap_evidence']['rotation_disagreement']>.1
    assert v2['review']['candidate'] is None


def test_v2_same_window_change_and_disagreement_remain_actionable():
    current=[action()]*18+[action(.7)]*32
    p=aligned_packet(current, dict(origin=0, actions=[action()]*50))
    cue=p['review']['candidate']
    assert cue['target_step']==18 and cue['evidence']['steps']==[18,19]
    assert cue['evidence']['geometry']==['rotation_change']
    assert cue['evidence']['disagreement']==['rotation_disagreement']
    assert apply_candidate([],packet=p,step=8,end=50,prediction_id=1,
                           covered=[],tolerance=5)[0]['target_step']==18


def test_v2_does_not_extend_the_old_prediction_to_support_a_later_cue():
    current=[action(.3)]*36+[action(.7)]*14
    p=aligned_packet(current, dict(origin=-25, actions=[action()]*50))
    assert p['review']['overlap_evidence']['last_step']==23
    assert p['review']['candidate'] is None


def test_runtime_v2_is_opt_in_and_records_aligned_evidence(tmp_path):
    r=runtime(50);r.feedback_policy='overlap_pacing_v2';r.feedback_minimum=20
    r.action_signal_settings=dict(settings(),version=ALIGNED_VERSION,refresh_budget_enabled=False)
    r.action_signal_directory=tmp_path
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=[action()]*50),
                dict(commit_step=0,commit_monotonic=0.))
    actions=[action(.3)]*18+[action(.7)]*32
    r.committed(dict(prediction_id=1,observation_step=20,observation_monotonic=1.,actions=actions),
                dict(commit_step=28,commit_monotonic=1.4))
    d=r.decide(now=1.4,step=28,queue={20+i:(a,1,20,i) for i,a in enumerate(actions)},
               rows=[],inflight=False,rtc_delay_steps=10)
    assert d['action_signal_review']['version']==ALIGNED_VERSION
    assert d['action_signal_added'] and d['candidate_event_kinds']==['action_revision']
    cue=d['action_signal_review']['candidate']
    assert cue['evidence']['steps']==[37,38]
    assert d['action_signal_review']['disagreement_scope']=='same_target_window'
    assert len((tmp_path/'action_signals.jsonl').read_text().splitlines())==1


def test_v3_requires_change_and_disagreement_in_the_same_component():
    current=[action()]*18+[action(.7)]*32
    old=[[-1.,*a[1:]] for a in current]
    v2=aligned_packet(current,dict(origin=0,actions=old))
    v3=aligned_packet(current,dict(origin=0,actions=old),version=PAIRED_VERSION)
    assert v2['review']['candidate']['evidence']['geometry']==['rotation_change']
    assert v2['review']['candidate']['evidence']['disagreement']==['translation_disagreement']
    assert v3['review']['candidate'] is None
    useful=aligned_packet(current,dict(origin=0,actions=[action()]*50),version=PAIRED_VERSION)
    assert useful['review']['candidate']['target_step']==18


def test_v3_skips_covered_phase_and_preserves_a_later_independent_change():
    actions=[action()]*18+[action(.7)]*18+[action(1.2)]*14
    kw=dict(actions=actions,origin=0,source_step=8,history={},
        reference=dict(origin=0,actions=[action()]*50),config=EventConfig(),period=.05,
        response_seconds=.5,margin_seconds=.025,prediction_count=1)
    phase=dict(start_step=15,stop_step=30,kinds=['action_revision'],event_source='remaining_v2')
    cfg=dict(settings(),version=PAIRED_VERSION,refresh_budget_enabled=False)
    before=prepare(**kw,settings=cfg)
    after=prepare(**kw,settings=cfg,covered_phases=[phase])
    assert before['review']['candidate']['target_step']==18
    assert after['review']['candidate']['target_step']==35
    assert after['review']['phase_rejected']>0
    assert after['review']['budget_rejected']==0
    assert after['review']['covered_phases']==[phase]
    assert all(not p['admitted'] for p in after['opportunities'] if 15<=p['target_step']<30)
    legacy=dict(cfg,version=ALIGNED_VERSION)
    assert prepare(**kw,settings=legacy)==prepare(**kw,settings=legacy,covered_phases=[phase])


def test_optional_point_filter_selects_later_candidate_and_preserves_v4_default():
    actions=[action()]*18+[action(.7)]*18+[action(1.2)]*14
    kw=dict(actions=actions,origin=0,source_step=8,history={},
        reference=dict(origin=0,actions=[action()]*50),config=EventConfig(),period=.05,
        response_seconds=.5,margin_seconds=.025,prediction_count=1)
    cfg=dict(settings(),version=PAIRED_VERSION,refresh_budget_enabled=False)
    covered=[dict(kind='action_revision',target_step=18)]
    old=prepare(**kw,settings=cfg,covered_events=covered)
    assert old==prepare(**kw,settings=cfg)
    assert old['review']['candidate']['target_step']==18
    assert apply_candidate([],packet=old,step=8,end=50,prediction_id=1,
        covered=covered,tolerance=5)==[]
    fixed=prepare(**kw,settings=dict(cfg,skip_covered_candidates=True),covered_events=covered)
    assert fixed['review']['candidate']['target_step']==35
    assert fixed['review']['point_rejected']>0
    actual=apply_candidate([],packet=fixed,step=8,end=50,prediction_id=1,
        covered=covered,tolerance=5)
    assert [e['target_step'] for e in actual]==[35]
    earlier=[dict(kind='gripper_close',target_step=30)]
    assert apply_candidate(earlier,packet=fixed,step=8,end=50,prediction_id=1,
        covered=covered,tolerance=5)==earlier
    phase=dict(start_step=30,stop_step=45,kinds=['action_revision'],event_source='remaining_v2')
    no_later=prepare(**kw,settings=dict(cfg,skip_covered_candidates=True),
        covered_events=covered,covered_phases=[phase])
    assert no_later['review']['candidate'] is None


def test_point_filter_is_typed_and_applied_before_the_saved_runtime_packet(tmp_path):
    r=runtime(50);r.feedback_policy='overlap_pacing_v2';r.feedback_minimum=20
    r.action_signal_settings=dict(settings(),version=PAIRED_VERSION,
        refresh_budget_enabled=False,skip_covered_candidates=True)
    r.action_signal_directory=tmp_path
    r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=[action()]*50),
        dict(commit_step=0,commit_monotonic=0.))
    actions=[action()]*18+[action(.7)]*8+[action(1.2)]*24
    r.committed(dict(prediction_id=1,observation_step=20,observation_monotonic=1.,actions=actions),
        dict(commit_step=28,commit_monotonic=1.4))
    r.covered_events=[dict(kind='action_revision',target_step=38),dict(kind='gripper_close',target_step=45)]
    d=r.decide(now=1.4,step=28,queue={20+i:(a,1,20,i) for i,a in enumerate(actions)},
        rows=[],inflight=False,rtc_delay_steps=10)
    assert d['action_signal_review']['candidate']['target_step']==45
    assert d['action_signal_added']
    saved=json.loads((tmp_path/'action_signals.jsonl').read_text())
    assert saved['review']==d['action_signal_review']
    assert saved['review']['point_rejected']>0


def test_v3_does_not_blanket_suppress_other_event_types_in_a_grasp_phase():
    actions=[action()]*18+[action(grip=1.)]*32
    kw=dict(actions=actions,origin=0,source_step=8,history={},
        reference=dict(origin=0,actions=[action()]*50),config=EventConfig(),period=.05,
        response_seconds=.5,margin_seconds=.025,prediction_count=1,
        settings=dict(settings(),version=PAIRED_VERSION,refresh_budget_enabled=False))
    phase=dict(start_step=15,stop_step=30,kinds=['gripper_open'],event_source='remaining_v2')
    p=prepare(**kw,covered_phases=[phase])
    assert p['review']['candidate']['target_step']==18
    assert p['review']['candidate']['evidence']['disagreement']==['gripper_disagreement']
    assert p['review']['phase_rejected']==0


def test_phase_only_preserves_v2_evidence_while_honoring_same_kind_coverage():
    actions=[action()]*18+[action(.7)]*18+[action(1.2)]*14
    old=[[-1.,*a[1:]] for a in actions]
    kw=dict(actions=actions,origin=0,source_step=8,history={},
        reference=dict(origin=0,actions=old),config=EventConfig(),period=.05,
        response_seconds=.5,margin_seconds=.025,prediction_count=1)
    cfg=dict(settings(),version=PAIRED_VERSION,pair_components=False,refresh_budget_enabled=False)
    plain=prepare(**kw,settings=cfg)
    legacy=prepare(**kw,settings=dict(settings(),version=ALIGNED_VERSION,refresh_budget_enabled=False))
    assert plain['review']['candidate']==legacy['review']['candidate']
    phase=dict(start_step=15,stop_step=30,kinds=['action_revision'],event_source='remaining_v2')
    fixed=prepare(**kw,settings=cfg,covered_phases=[phase])
    assert fixed['review']['candidate']['target_step']==35
    assert fixed['review']['evidence_pairing']=='same_target_window'
    assert fixed['review']['phase_rejected']>0


def test_gripper_conflict_is_optional_and_needs_opposite_confirmed_predictions():
    cfg=dict(settings(),version=ALIGNED_VERSION,refresh_budget_enabled=False)
    kw=dict(actions=[action(grip=1.)]*50,origin=20,source_step=28,history={},
        reference=dict(origin=0,actions=[action(grip=-1.)]*50),config=EventConfig(),
        period=.05,response_seconds=.5,margin_seconds=.025,prediction_count=2)
    assert prepare(**kw,settings=cfg)['review']['candidate'] is None
    cfg['gripper_conflict']=True
    p=prepare(**kw,settings=cfg)
    cue=p['review']['candidate']
    assert cue['target_step']==28 and cue['evidence']['steps']==[28,29]
    assert cue['evidence']['geometry']==['gripper_plan_conflict']
    assert cue['evidence']['disagreement']==['gripper_disagreement']
    # An isolated opposite command, agreed commands and absent overlap cannot
    # supply a sustained disagreement. Never manufacture an old tail.
    for old in ([action(grip=1.)]*8+[action(grip=-1.)]+[action(grip=1.)]*41,
                [action(grip=1.)]*50):
        assert prepare(**dict(kw,reference=dict(origin=20,actions=old)),
                       settings=cfg)['review']['candidate'] is None
    assert prepare(**dict(kw,reference=None),settings=cfg)['review']['candidate'] is None
