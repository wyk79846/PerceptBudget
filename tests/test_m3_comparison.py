import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.m3_comparison import GROUPS, comparison_settings, configure_group, group_order


def settings_fixture(tmp_path):
    proof=tmp_path/'watchdog.json'
    proof.write_text(json.dumps(dict(status='completed',checks={'passed':True},
        cases=[{'resumed_execution_step':51},{'resumed_execution_step':62},{'case':'hang'}])))
    return dict(timing=dict(maximum_action_offset=20,maximum_observation_age_seconds=2.),
        formal=dict(trigger_strategy='events_only',suites=['libero_spatial','libero_object','libero_goal'],task_ids=[1,3,6],
            request_timeout_seconds=3.,cleanup_grace_seconds=3.,watchdog_reference=str(proof)))


def test_formal_grid_is_nine_tasks_and_four_rotating_groups(tmp_path):
    source=settings_fixture(tmp_path);before=copy.deepcopy(source)
    plan=comparison_settings(source)
    assert len(plan['calibration_tasks'])==9 and all(t[1] in (1,3,6) for t in plan['calibration_tasks'])
    assert comparison_settings(source,screen=True)['calibration_tasks']==[['libero_spatial',0]]
    assert source==before and plan['timing']['maximum_action_offset']==50
    assert plan['timing']['maximum_observation_age_seconds'] is None
    assert [group_order(i)[0] for i in range(4)]==list(GROUPS)


def test_formal_plan_rejects_unverified_recovery_and_changed_timeout(tmp_path):
    s=settings_fixture(tmp_path);s['formal']['request_timeout_seconds']=2.
    with pytest.raises(ValueError):comparison_settings(s)
    s['formal']['request_timeout_seconds']=3.
    proof=s['formal']['watchdog_reference']
    with open(proof,'w') as f:json.dump(dict(status='completed',checks={'passed':True},cases=[{},{}]),f)
    with pytest.raises(ValueError,match='recovered'):comparison_settings(s)


@pytest.mark.parametrize('strategy',['events_only','variance_only','events_and_variance'])
def test_group_configuration_only_changes_declared_refresh_and_device_policy(strategy):
    original=dict(H=50,N=5,state_enabled=True,precision='bfloat16',m3_scheduler_mode='deadline_dvfs',m3_formal_signal=strategy)
    configs=[]
    for group in GROUPS:
        cfg=copy.deepcopy(original)
        assert configure_group(cfg,group)==5
        assert all(cfg[k]==original[k] for k in ('H','N','state_enabled','precision'))
        assert cfg['m3_merge_slowdown_steps'] is None and cfg['m3_open_probe_plan'] is None
        configs.append(cfg)
    assert [c['m3_scheduler_mode'] for c in configs]==['record_only','deadline_high','deadline_dvfs','deadline_dvfs']
    assert [c['m3_dvac_enabled'] for c in configs]==[False]+[strategy!='events_only']*3
    assert [c['m3_action_events_enabled'] for c in configs]==[False]+[strategy!='variance_only']*3
    assert [c['m3_hold_high_clocks'] for c in configs]==[True,True,False,False]
    assert configs[2]['m3_dvfs_policy']=='independent' and configs[3]['m3_dvfs_policy']=='joint'


def test_independent_restores_at_hot_start_joint_reserves_recovery_time():
    from perceptbudget.closed_loop import DeadlineRuntime
    cfg=dict(H=50,m3_scheduler_mode='deadline_dvfs',m3_dvac_enabled=True,m3_action_events_enabled=False,
        m3_timing=dict(service_quantile=.95,safety_margin_seconds=.05,maximum_action_offset=50,
            maximum_observation_age_seconds=None,minimum_low_idle_seconds=.2,margin_updates=False),m3_events={},
        m3_service_samples={h:[.3]*20 for h in ('hot','short','long')},m3_recovery_samples=[.5],
        rtc_initial_control_period_seconds=.05)
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50;queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    starts=[]
    for policy in ('independent','joint'):
        runtime=DeadlineRuntime(dict(cfg,m3_dvfs_policy=policy))
        runtime.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=0.,actions=actions,
            dvac={'prefix_actions':40}),dict(commit_step=0,commit_monotonic=.1))
        switches=[]
        runtime.clocks=SimpleNamespace(profile='gpu_emc',pending=None,poll=lambda:None,switch=switches.append)
        plan=runtime.decide(now=1.5,step=30,queue=queue,rows=[],inflight=False,rtc_delay_steps=5)
        assert plan['critical_step'] is None and plan['dvac_step']==40
        runtime.control_clocks(plan,inflight=False)
        if policy=='independent':
            assert plan['service_budget_history']=='hot' and not switches
        else:
            assert plan['service_budget_history']=='recovery' and switches==['baseline']
        starts.append(plan['latest_start'])
    assert starts[0]-starts[1]==pytest.approx(.2)


def test_formal_cli_screen_and_full_grid_do_not_load_models(capsys):
    from scripts.run_m3 import main
    for extra,episodes,tasks in ((['--comparison-screen'],4,1),([],36,9)):
        with patch('sys.argv',['m3','--mode','comparison','--dry-run',*extra]),patch('subprocess.run') as launch:
            assert main()==0
        plan=json.loads(capsys.readouterr().out)
        assert plan['episodes']==episodes and len(plan['tasks'])==tasks and plan['model_loads']==1
        assert plan['formal_groups']==list(GROUPS) and plan['denoising_triggers']==[False]*4
        assert plan['action_event_triggers']==[False,True,True,True] and plan['formal_trigger_strategy']=='events_only'
        assert plan['request_timeout_seconds']==3. and plan['exact_bootstrap_pair']
        launch.assert_not_called()


def test_summary_includes_failed_attempt_energy_and_flags_bad_initial_pairs(tmp_path):
    from perceptbudget.m3_comparison import summarize_comparison
    episodes=[]
    for i,group in enumerate(GROUPS):
        run=tmp_path/group;(run/'rtc').mkdir(parents=True)
        (run/'summary.json').write_text(json.dumps(dict(episode=dict(including_drain_seconds=2.,
            including_drain_power=dict(total_energy_mj=10000+1000*i,coverage_fraction=1.),underflow_steps=[]))))
        (run/'rtc/q5_requests.json').write_text('[{}]')
        episodes.append(dict(run=str(run),suite='libero_spatial',task_id=1,view=group,
            outcome='timeout' if i==0 else 'success',steps=100,wall_seconds=1.9))
    record=dict(episodes=episodes,paired_bootstrap_audit=dict(cases=[dict(suite='libero_spatial',task_id=1,passed=False)]))
    report=summarize_comparison(tmp_path,record,screen=True)
    assert report['exact_pair_tasks']==0 and all(not r['exact_initial_pair'] for r in report['rows'])
    assert report['groups'][0]['mean_attempt_energy_j']==10. and report['groups'][0]['energy_per_success_j'] is None
    assert report['groups'][1]['energy_per_success_j']==11.
    assert not any(pair['both_success'] for pair in report['pairs'] if pair['before']=='A_fixed_q5')
    assert (tmp_path/'comparison.csv').exists()


@pytest.mark.parametrize('broken',[None,'low_full','missing_power','no_restore','high_lowered'])
def test_formal_measurements_reject_wrong_clocks_or_missing_energy(tmp_path,broken):
    from perceptbudget.m3_comparison import audit_group_clocks
    rtc=tmp_path/'rtc';rtc.mkdir()
    (rtc/'q5_requests.json').write_text(json.dumps([dict(observation_monotonic=1.,ready_monotonic=2.)]))
    events=[dict(status='SAMPLE',monotonic=1.5,actual={'gpu/cur_freq':816000000 if broken=='low_full' else 1173000000,
        'emc/rate':3199000000,'cpu/scaling_cur_freq':1497600})]
    if broken=='high_lowered':events.append(dict(status='APPLIED',profile='gpu_emc'))
    text='\n'.join(json.dumps(e) for e in events)+'\n'
    if broken!='no_restore':text+='RESTORED\n'
    (rtc/'runtime_clocks.jsonl').write_text(text)
    (tmp_path/'summary.json').write_text(json.dumps(dict(episode=dict(including_drain_power=dict(
        total_energy_mj=10000,coverage_fraction=.9 if broken=='missing_power' else 1.,missing_count=0)))))
    assert audit_group_clocks(tmp_path,'B_high')['passed']==(broken is None)
