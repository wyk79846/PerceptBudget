import json
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.action_events import EventConfig, analyze_episode, detect_events


def action(x=1., y=0., grip=-1.):
    return [x, y, 0., 0., 0., 0., grip]


def window_requests(new_target=7):
    old=dict(prediction_id=0,observation_step=0,
             observation_monotonic=0.,
             actions=[action(x=1+i*.1,grip=-1 if i<7 else 1) for i in range(12)])
    new=dict(prediction_id=1,observation_step=3,
             observation_monotonic=.5,
             actions=[action(x=1+(i+3)*.1,grip=-1 if i+3<new_target else 1) for i in range(12)],
             rtc_guidance={'delay_estimate_steps':6})
    return old,new


def test_gripper_windows_align_absolute_steps_and_exclude_consumed_actions():
    from perceptbudget.action_events import compare_gripper_windows
    old,new=window_requests()
    result=compare_gripper_windows(old,new,5,radius=2)[0]
    assert result['status']=='matched' and result['event_step_shift']==0
    assert result['local_window']['start_step']==5 and result['local_window']['end_step_exclusive']==10
    assert result['local_window']['xyz_rmse']==0 and result['local_window']['gripper_sign_disagreement_steps']==0
    assert result['beyond_estimated_delay']['start_step']==9
    old['actions'][4][0]=999
    assert compare_gripper_windows(old,new,5,radius=2)[0]==result


def test_same_gripper_event_position_does_not_hide_different_local_actions():
    from perceptbudget.action_events import compare_gripper_windows
    old,new=window_requests()
    new['actions'][4][0]+=.4
    new['actions'][4][3]+=.2
    result=compare_gripper_windows(old,new,5,radius=2)[0]
    assert result['event_step_shift']==0
    assert result['local_window']['xyz_max_l2']==pytest.approx(.4)
    assert result['local_window']['rotation_rmse']>0
    assert result['local_window']['max_cumulative_xyz_command_difference']==pytest.approx(.4)


def test_gripper_window_reports_event_shift_and_unavailable_overlap():
    from perceptbudget.action_events import compare_gripper_windows
    old,new=window_requests(new_target=8)
    result=compare_gripper_windows(old,new,5,radius=2)[0]
    assert result['event_step_shift']==1 and result['local_window']['gripper_sign_disagreement_steps']==1
    assert compare_gripper_windows(old,new,8)[0]['status']=='new_event_not_in_old_forecast'
    old,new=window_requests(new_target=13)
    assert compare_gripper_windows(old,new,5)[0]['status']=='outside_common_future'


def test_gripper_window_marks_multiple_same_kind_transitions_ambiguous():
    from perceptbudget.action_events import compare_gripper_windows
    old,new=window_requests()
    for i in range(12):old['actions'][i][6]=1 if 3<=i<5 or i>=7 else -1
    result=compare_gripper_windows(old,new,3)[0]
    assert result['status']=='ambiguous_match'


def test_gripper_window_rejects_future_observation_and_invalid_radius():
    from perceptbudget.action_events import compare_gripper_windows
    old,new=window_requests()
    with pytest.raises(ValueError):compare_gripper_windows(old,new,2)
    with pytest.raises(ValueError):compare_gripper_windows(old,new,5,radius=-1)


def open_probe_fixture():
    from perceptbudget.action_events import select_open_refresh_probe
    old,new=window_requests()
    for q in (old,new):
        for a in q['actions']:a[6]*=-1
    following=dict(prediction_id=2,observation_step=5,
        m3_timing=dict(reason='action_event',critical_step=7,candidate_event_kinds=['gripper_open']))
    commits=[dict(prediction_id=0,observation_step=0,commit_step=0),
             dict(prediction_id=1,observation_step=3,commit_step=5)]
    plan=select_open_refresh_probe([old,new,following],commits)
    return old,new,plan


def test_open_probe_absent_target_is_not_an_execution_error():
    from perceptbudget.action_events import select_open_refresh_probe
    old,new,_=open_probe_fixture()
    old['actions']=old['actions'][:6]  # Coverage ends before the new open event.
    following=dict(prediction_id=2,observation_step=5,
        m3_timing=dict(reason='action_event',critical_step=7,candidate_event_kinds=['gripper_open']))
    commits=[dict(prediction_id=0,observation_step=0,commit_step=0),
             dict(prediction_id=1,observation_step=3,commit_step=5)]
    assert select_open_refresh_probe([old,new,following],commits) is None


@pytest.mark.parametrize('gap,intervening,expected',[(8,None,True),(9,None,False),(14,None,False),(3,'turn',False),(3,'slowdown',False)])
def test_nearby_slowdown_only_merges_adjacent_gripper(gap,intervening,expected):
    from perceptbudget.action_events import merge_nearby_slowdowns
    import copy
    events=[dict(kind='slowdown',offset=10),dict(kind='gripper_close',offset=10+gap)]
    if intervening:events.append(dict(kind=intervening,offset=11))
    before=copy.deepcopy(events)
    kept,merged=merge_nearby_slowdowns(events,8)
    assert events==before
    assert (events[0] not in kept)==expected
    assert any(e['kind']=='gripper_close' for e in kept)
    if expected:assert merged[0]['replacement_offset']==10+gap


def test_nearby_slowdown_keeps_simultaneous_and_unconfirmed_tail():
    from perceptbudget.action_events import merge_nearby_slowdowns,detect_events
    events=[dict(kind='slowdown',offset=10),dict(kind='turn',offset=10),dict(kind='gripper_open',offset=12)]
    assert merge_nearby_slowdowns(events,8)==(events,[])
    actions=[[1. if i<4 else .1,0.,0.,0.,0.,0.,-1. if i<9 else 1.] for i in range(10)]
    ev=detect_events(actions)
    assert not any(e['kind'].startswith('gripper') for e in ev)
    assert merge_nearby_slowdowns(ev,8)==(ev,[])


def test_single_slowdown_probe_selects_real_refresh_and_retains_other_signals():
    from perceptbudget.action_events import select_slowdown_refresh_probe
    from perceptbudget.closed_loop import DeadlineRuntime
    actions=[[1. if i<8 else .1,0.,0.,0.,0.,0.,-1. if i<12 else 1.] for i in range(20)]
    old=dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions,dvac={'prefix_actions':16})
    refresh=dict(prediction_id=1,observation_step=3,rtc_guidance={'old_chunk_origin_step':0},
        m3_timing=dict(reason='action_event',critical_step=8,candidate_event_kinds=['slowdown']))
    plan=select_slowdown_refresh_probe([old,refresh],[dict(prediction_id=0,commit_step=0)])
    assert plan['prediction_id']==0 and plan['kind']=='slowdown' and plan['target_step']==8
    cfg=dict(H=20,m3_scheduler_mode='deadline_high',m3_dvac_enabled=True,m3_open_probe_plan=plan,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=20,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(cfg)
    runtime.committed(old,dict(commit_step=0,commit_monotonic=.1))
    assert [e['kind'] for e in runtime.events]==['gripper_close']
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    decision=runtime.decide(now=0.,step=0,queue=queue,rows=[],inflight=False)
    assert decision['critical_step']==12 and decision['dvac_step']==16
    assert decision['diagnostic_open_probe']['kind']=='slowdown'
    fresh=dict(old,prediction_id=1,observation_step=3)
    runtime.committed(fresh,dict(commit_step=3,commit_monotonic=.3))
    assert 'slowdown' in [e['kind'] for e in runtime.events]  # Only one source hint is removed.
    refresh['m3_timing']['reason']='denoising_variance'
    assert select_slowdown_refresh_probe([old,refresh],[dict(prediction_id=0,commit_step=0)]) is None


def test_open_probe_selects_one_repeated_event_and_preserves_variance_deadline():
    from perceptbudget.closed_loop import DeadlineRuntime
    old,new,plan=open_probe_fixture()
    assert plan['prediction_id']==1 and plan['target_step']==7 and plan['kind']=='gripper_open'
    cfg=dict(H=50,m3_scheduler_mode='deadline_high',m3_dvac_enabled=True,m3_open_probe_plan=plan,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},rtc_initial_control_period_seconds=.05)
    runtime=DeadlineRuntime(cfg)
    runtime.committed(old,dict(commit_step=0,commit_monotonic=.1))
    new['dvac']={'prefix_actions':5}
    runtime.committed(new,dict(commit_step=5,commit_monotonic=1.))
    queue={new['observation_step']+i:(a,1,3,i) for i,a in enumerate(new['actions'])}
    decision=runtime.decide(now=1.,step=5,queue=queue,rows=[],inflight=False)
    assert decision['critical_step'] is None and decision['reason']=='denoising_variance'
    assert decision['start_now'] and decision['margin_seconds']==.05
    assert decision['diagnostic_open_probe']['target_step']==7
    unchanged=DeadlineRuntime(dict(cfg,m3_open_probe_plan=None))
    unchanged.committed(new,dict(commit_step=5,commit_monotonic=1.))
    assert unchanged.decide(now=1.,step=5,queue=queue,rows=[],inflight=False)['reason']=='action_event'
    with pytest.raises(RuntimeError,match='context changed'):
        DeadlineRuntime(cfg).committed(new,dict(commit_step=6,commit_monotonic=1.))


@pytest.mark.parametrize('changed',[None,'noise','control','marker'])
def test_open_probe_prefix_audit_rejects_preintervention_differences(tmp_path,changed):
    import copy
    from perceptbudget.action_events import audit_open_probe_prefix
    old,new,plan=open_probe_fixture()
    for q in (old,new):
        q.update(observation_raw_state=[0.]*8,model_state=[0.]*8,
            actions_raw=[[0.]*32 for _ in range(12)],rtc_solver_inputs={'noise':[[0.]*32 for _ in range(12)]})
    paths=[]
    for i in range(2):
        p=tmp_path/str(i);(p/'rtc').mkdir(parents=True);paths.append(p)
        qs=copy.deepcopy([old,new]);trace=[dict(event='control_step',step=s,
            action=list(old['actions'][s]),before={'raw_state':[0.]*8}) for s in range(5)]
        if i:
            trace.append(dict(event='m3_decision',step=5,diagnostic_open_probe={
                k:plan[k] for k in ('prediction_id','commit_step','target_step','kind')}))
            if changed=='noise':qs[1]['rtc_solver_inputs']['noise'][0][0]=1e-7
            if changed=='control':trace[2]['action'][0]+=1e-7
            if changed=='marker':trace.pop()
        (p/'rtc/q5_requests.json').write_text(json.dumps(qs))
        (p/'rtc/trace.jsonl').write_text('\n'.join(json.dumps(x) for x in trace))
    assert audit_open_probe_prefix(*paths,plan)['passed']==(changed is None)


def test_offline_episode_attaches_aligned_window_metrics_without_modifying_records(tmp_path):
    old,new=window_requests()
    (tmp_path/'rtc').mkdir()
    (tmp_path/'rtc/q5_requests.json').write_text(json.dumps([old,new]))
    (tmp_path/'run.json').write_text(json.dumps({'config_snapshot':{
        'rtc_suite':'libero_spatial','rtc_initial_control_period_seconds':.05}}))
    trace=[dict(event='prediction_committed',prediction_id=0,observation_step=0,commit_step=0,commit_monotonic=10.),
           dict(event='prediction_committed',prediction_id=1,observation_step=3,commit_step=5,commit_monotonic=10.25)]
    for i in range(12):
        q=old if i<5 else new
        trace.append(dict(event='control_step',step=i,q=5,task_id=0,applied_arm_gain=1.,
            action=q['actions'][i-q['observation_step']],action_start_monotonic=10+i*.05,
            action_end_monotonic=10+i*.05+.04,source={'prediction_id':q['prediction_id']},after={'contacts':[]}))
    (tmp_path/'rtc/trace.jsonl').write_text('\n'.join(json.dumps(e) for e in trace))
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=analyze_episode(tmp_path)
    comparison=result['gripper_window_comparisons'][0]
    assert comparison['status']=='matched' and comparison['local_window']['xyz_rmse']==0
    assert comparison['local_window']['start_step']>=5
    assert all(p.read_bytes()==content for p,content in before.items())


def test_gripper_hysteresis_ignores_jitter_and_labels_both_directions():
    values = [-1, -1, .05, .3, -.1, 1, 1, -.1, -1, -1]
    events = detect_events([action(grip=g) for g in values])
    grip = [e for e in events if e['kind'].startswith('gripper')]
    assert [(e['kind'], e['offset']) for e in grip] == [('gripper_close', 5), ('gripper_open', 8)]
    assert detect_events([action(grip=g) for g in values], start_offset=7) == [grip[-1]]


def test_motion_hints_use_translation_only_and_require_sustained_change():
    actions = [action(), action(), action(x=.4), action(x=.3), action(x=0,y=1), action(x=0,y=1)]
    events = detect_events(actions)
    assert [(e['kind'], e['offset']) for e in events] == [('slowdown', 2), ('turn', 4)]
    actions = [[0.,0.,0.,1.,0.,0.,-1.] for _ in range(5)]
    assert detect_events(actions) == []  # Rotation commands are not xyz motion.


def test_tiny_or_single_tick_turns_do_not_trigger():
    assert detect_events([action(), action(x=0,y=1), action()]) == []
    assert detect_events([action(x=.001), action(x=0,y=.001), action(x=0,y=.001)]) == []


@pytest.mark.parametrize('motion,offsets',[
    ([1,1,1,.3,.1,.3,1,1,1],[4]),
    ([1,1,1,3,1,1,1],[]),
    ([1,1,1,.1,1,1,1],[]),
    ([.03,.02,.01,.02,.03],[]),
    ([1,.8,.6,.4,.2,.1,0],[]),
    ([0,0,0,0,0],[]),
])
def test_phase_valley_uses_surrounding_motion_and_filters_isolated_noise(motion,offsets):
    acts=[action(x=v) for v in motion]
    events=[e for e in detect_events(acts,EventConfig(slowdown_detector='valley')) if e['kind']=='slowdown']
    assert [e['offset'] for e in events]==offsets
    assert all(e['relative_depth']>=.5 and e['detector']=='phase_valley' for e in events)
    assert all(e['offset']<=e['confirmed_offset']<len(acts) for e in events)


def test_phase_valley_changes_only_slowdown_and_keeps_absolute_onset_after_crop():
    acts=[action(x=v,grip=-1 if i<6 else 1) for i,v in enumerate([1,1,1,.3,.1,.3,1,1,1])]
    original=detect_events(acts)
    cfg=EventConfig(slowdown_detector='valley')
    new=detect_events(acts,cfg)
    assert [e for e in original if e['kind']!='slowdown']==[e for e in new if e['kind']!='slowdown']
    assert detect_events(acts,cfg,start_offset=5)==[e for e in new if e['offset']>=5]
    assert EventConfig().slowdown_detector=='drop'


def test_phase_valley_spacing_selects_one_of_two_close_boundaries():
    from perceptbudget.action_events import _phase_valleys
    cfg=EventConfig(slowdown_detector='valley',valley_spacing_steps=12)
    events=_phase_valleys([1,1,1,.3,.1,.3,1,1,1,.2,0,.2,1,1,1],cfg)
    assert len(events)==1 and events[0]['offset']==10


@pytest.mark.parametrize('motion,baseline,sustained',[
    ([1,1,.4,.3,1,1],[2],[]),
    ([1,1,.4,.3,.2,1],[2],[2]),
    ([1,1,.4,.3],[2],[]),  # Insufficient future tail is not a confirmed third step.
    ([.01,.01,.003,.003,.003],[],[]),
])
def test_slowdown_duration_changes_only_predicted_persistence(motion,baseline,sustained):
    acts=[action(x=v) for v in motion]
    def events(n):return [e for e in detect_events(acts,EventConfig(slowdown_confirm_steps=n)) if e['kind']=='slowdown']
    assert [e['offset'] for e in events(2)]==baseline
    result=events(3)
    assert [e['offset'] for e in result]==sustained
    assert all(e['confirmed_offset']==e['offset']+2 for e in result)


def test_slowdown_confirmation_does_not_change_gripper_turn_or_start_offset():
    acts=[action(x=1),action(x=1),action(x=.4,grip=1),action(x=.3,grip=1),
          action(x=.2,grip=1),action(x=0,y=1,grip=1),action(x=0,y=1,grip=1)]
    old=detect_events(acts)
    new=detect_events(acts,EventConfig(slowdown_confirm_steps=3))
    assert [e for e in old if e['kind']!='slowdown']==[e for e in new if e['kind']!='slowdown']
    assert detect_events(acts,EventConfig(slowdown_confirm_steps=3),start_offset=3)==[e for e in new if e['offset']>=3]
    assert next(e for e in new if e['kind']=='gripper_close')['confirmed_offset']==3
    assert EventConfig().confirm_steps==2 and EventConfig().slowdown_confirm_steps is None


@pytest.mark.parametrize('value',[0,-1,True,3.0])
def test_slowdown_duration_validation(value):
    with pytest.raises(ValueError):EventConfig(slowdown_confirm_steps=value)


@pytest.mark.parametrize('data', [[[0]*32], [[float('nan')]*7]])
def test_normalized_padded_or_nonfinite_actions_rejected(data):
    with pytest.raises(ValueError):
        detect_events(data)


def make_episode(path):
    (path/'rtc').mkdir(parents=True)
    acts = [action(), action(), action(grip=1), action(grip=1), action(grip=1)]
    req = [dict(prediction_id=0, actions=acts), dict(prediction_id=1, actions=acts)]
    (path/'rtc/q5_requests.json').write_text(json.dumps(req))
    metadata = dict(config_snapshot=dict(rtc_suite='libero_spatial', rtc_initial_control_period_seconds=.05))
    (path/'run.json').write_text(json.dumps(metadata))
    trace = [dict(event='prediction_committed', prediction_id=0, observation_step=0, commit_step=1, commit_monotonic=10.)]
    for i, a in enumerate(acts):
        trace.append(dict(event='control_step', step=i, q=5, task_id=0, action=a, applied_arm_gain=1.,
                          action_start_monotonic=10.+i*.07, action_end_monotonic=10.05+i*.07,
                          source=dict(prediction_id=0 if i<2 else 2), after={'contacts': []}))
    (path/'rtc/trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))


def test_offline_review_excludes_consumed_prefix_and_marks_overwritten_tail(tmp_path):
    make_episode(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report = analyze_episode(tmp_path)
    close = next(e for e in report['forecasts'] if e['kind']=='gripper_close')
    assert close['target_step']==2 and close['known_step']==1
    assert close['disposition']=='target_overwritten_or_underflow'
    assert close['command_step_error']==0
    assert close['predicted_lead_seconds']==pytest.approx(.05)  # No future tick leakage.
    assert report['summary']['uncommitted_requests']==1
    assert report['chunks'][0]['remaining_output_actions']==4
    assert all(p.read_bytes()==contents for p,contents in before.items())


def test_recording_session_uses_one_resident_runner_without_policy_changes(tmp_path):
    from perceptbudget.closed_loop import recording_session
    event_path = tmp_path/'events.json'
    report = dict(run='episode',suite='libero_spatial',task_id=0,q=5,control_steps=5,mode='record_only',
                  summary=dict(committed_chunks=1, forecast_counts={}))
    event_path.write_text(json.dumps(report))
    record = dict(episodes=[dict(action_events=str(event_path))],model_loads=1,suffix_loads=1)
    settings = dict(calibration_tasks=[['libero_spatial',0]], events={})
    with patch('perceptbudget.device_quality.resident_quality', return_value=record) as runner, \
         patch('perceptbudget.closed_loop.compiled_environments', return_value=nullcontext({})):
        result=recording_session({},settings,tmp_path,Path('gate'))
    assert runner.call_args.kwargs['record_only']
    assert runner.call_args.kwargs['task_plan']==[('libero_spatial',0)]
    assert not result['policy_changed'] and not result['clocks_changed']
    assert result['model_loads']==1


def test_recording_dry_run_never_spawns_worker(capsys):
    from scripts.run_m3 import main
    with patch('sys.argv',['run_m3','--dry-run']), patch('subprocess.run') as launch:
        assert main()==0
    d=json.loads(capsys.readouterr().out)
    assert len(d['tasks'])==3 and d['model_loads']==1
    assert not d['policy_changed'] and not d['clocks_changed']
    launch.assert_not_called()


def test_dvfs_session_loads_measured_recovery_samples_before_resident_model(tmp_path):
    from perceptbudget.closed_loop import recording_session
    from perceptbudget.deadline_controller import FullServiceTimes
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(status='completed',samples_seconds=[.634,.581,.604])))
    event_path=tmp_path/'events.json'
    event_path.write_text(json.dumps(dict(run='episode',suite='libero_spatial',task_id=0,q=5,
        control_steps=5,mode='deadline_dvfs',summary=dict(committed_chunks=1,forecast_counts={}))))
    record=dict(episodes=[dict(action_events=str(event_path))],model_loads=1,suffix_loads=1)
    settings=dict(mode='deadline_dvfs',calibration_tasks=[['libero_spatial',0]],events={},
        timing=dict(source_summary='mock-m2',service_quantile=.95,recovery_calibration_source=str(calibration)))
    measured=FullServiceTimes({h:[.553]*20 for h in ('hot','short','long')})
    with patch('perceptbudget.closed_loop.FullServiceTimes.from_m2_summary',return_value=measured), \
         patch('perceptbudget.device_quality.resident_quality',return_value=record) as runner, \
         patch('perceptbudget.closed_loop.compiled_environments',return_value=nullcontext({})):
        result=recording_session({},settings,tmp_path,Path('gate'))
    cfg=runner.call_args.args[0]
    assert cfg['m3_recovery_samples']==[.634,.581,.604]
    assert cfg['m3_recovery_calibration']['initial_estimate_seconds']==.634
    assert len(cfg['m3_recovery_calibration']['sha256'])==64
    assert result['clocks_changed'] and cfg['m3_service_samples']['hot']==[.553]*20


def test_long_environment_parity_uses_fixed_controls_without_m1_quality_dependency():
    from perceptbudget.closed_loop import environment_replay
    actions,source=environment_replay('libero_10',0,{'cells':{}})
    assert len(actions)==20 and all(len(a)==7 for a in actions)
    assert max(abs(x) for a in actions for x in a[:6])==.02
    assert {a[6] for a in actions}=={-1.,1.}
    assert source=='fixed20-small-motion-cache-parity'
    with pytest.raises(ValueError):environment_replay('libero_spatial',0,{'cells':{}})
