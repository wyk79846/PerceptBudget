from copy import deepcopy
from types import SimpleNamespace
import pytest

from perceptbudget.full_energy import HISTORIES,hold,price,select,audit_record
from perceptbudget.three_frequency import ThreeFullRuntime
from test_three_frequency import config


def costs():
    cells=[]
    for entry in ('baseline','gpu_emc','deep_idle'):
        for p,seconds,joules in (('baseline',.4,11.),('gpu_emc',.8,10.)):
            for h in HISTORIES:
                cells.append(dict(mode='Full',profile=p,from_profile=entry,history=h,
                    mean_seconds=seconds,mean_request_energy_j=joules,observations=4))
    for after in ('baseline','gpu_emc'):
        for p in ('baseline','gpu_emc','deep_idle'):
            for h in HISTORIES:
                cells.append(dict(mode='Continue',profile=p,after_full_profile=after,history=h,
                    mean_power_w=.1 if p=='deep_idle' else 12.))
    switches=[dict(from_profile=p,profile='deep_idle',during_task=True,milliseconds=50.,switch_window_energy_j=.1)
              for p in ('baseline','gpu_emc')]
    return dict(source='synthetic',cells=cells,transitions=switches)


def inputs(window=.95):
    rows=[dict(profile='gpu_emc',feasible=True),dict(profile='baseline',feasible=True)]
    return dict(candidates=rows,selected=rows[0]),dict(decision_monotonic=10.,deadline=10.+window+.025,
        margin_seconds=.025,current_profile='deep_idle',admitted_remaining_actions=round(window/.05),
        estimated_control_period_seconds=.05)


def test_higher_request_energy_can_win_by_the_complete_window():
    f,d=inputs();c=costs()
    r=select(f,d,c,quiet_age=.5,minimum_idle=.2)
    assert r['profile']=='baseline' and r['selected']['segments'][0]['energy_j']==11.
    assert r['candidates'][0]['segments'][0]['energy_j']==10.
    assert r['selected']['energy_j']==pytest.approx(11.15)
    assert not audit_record(r,c,current_profile='deep_idle',minimum_idle=.2)


def test_infeasible_frequency_cannot_win_even_with_lower_energy():
    f,d=inputs();f['candidates'][1]['feasible']=False
    r=select(f,d,costs(),quiet_age=.5,minimum_idle=.2)
    assert r['profile']=='gpu_emc'
    assert not r['candidates'][1]['complete']


def test_critical_arrival_does_not_truncate_the_energy_accounting_window():
    f,d=inputs();d['deadline']=10.85
    r=select(f,d,costs(),quiet_age=.5,minimum_idle=.2)
    assert r['window_end']==pytest.approx(10.95)
    assert r['arrival_deadline']==10.85


def test_entry_recovery_not_added_again_and_lowering_replaces_hold_time():
    f,d=inputs();c=costs()
    r=select(f,d,c,quiet_age=.5,minimum_idle=.2)
    c['transitions'].append(dict(from_profile='deep_idle',profile='baseline',during_task=True,
        milliseconds=30.,switch_window_energy_j=1000.))
    other=select(f,d,c,quiet_age=.5,minimum_idle=.2)
    assert other['selected']['energy_j']==r['selected']['energy_j']
    assert sum(p['seconds'] for p in r['selected']['segments'])==pytest.approx(.95)
    damaged=deepcopy(r);damaged['candidates'][1]['history_options'][0]['segments'][0]['energy_j']+=1
    assert audit_record(damaged,c,current_profile='deep_idle',minimum_idle=.2)


def test_post_full_hold_integrates_three_response_bins():
    c=costs()
    for r in c['cells']:
        if r['mode']=='Continue':r['mean_power_w']={HISTORIES[0]:15.,HISTORIES[1]:9.,HISTORIES[2]:7.}[r['history']]
    assert hold(c,'deep_idle','baseline',2.)['energy_j']==pytest.approx(.2*15+.8*9+7)


def test_missing_entry_route_retains_original_without_zero_cost():
    f,d=inputs();c=costs();c['cells']=[r for r in c['cells'] if r['mode']!='Full']
    r=select(f,d,c,quiet_age=.5,minimum_idle=.2)
    assert r['profile']=='gpu_emc' and r['selected'] is None
    assert not r['energy_used_for_selection']


def test_other_history_keeps_energy_and_time_from_the_same_observation():
    f,d=inputs();c=costs()
    c['cells']=[r for r in c['cells'] if not (r['mode']=='Full' and r['history']==HISTORIES[1])]
    for row in c['cells']:
        if row['mode']=='Full' and row['profile']=='baseline':
            row['mean_seconds']=.4 if row['history']==HISTORIES[0] else .8
            row['mean_request_energy_j']=11. if row['history']==HISTORIES[0] else 10.
    r=select(f,d,c,quiet_age=.5,minimum_idle=.2)
    high=r['candidates'][1]
    assert not high['history_match']
    assert high['energy_j']==max(o['energy_j'] for o in high['history_options'])
    assert high['cell_history']==HISTORIES[2]  # cheaper request but hotter retained tail


def test_runtime_keeps_original_trigger_and_full_bounds_when_switching_to_high():
    cfg=config();c=costs()
    for row in c['cells']:
        if row['mode']=='Full' and row['profile']=='baseline':row['mean_request_energy_j']=1.
    original=ThreeFullRuntime(cfg)
    candidate=ThreeFullRuntime(dict(cfg,m3_full_energy=True,m3_full_energy_costs=c))
    actions=[[1.,0.,0.,0.,0.,0.,-1.]]*50
    for r in (original,candidate):
        r.clocks=SimpleNamespace(profile='deep_idle',pending=None,last_low=9.,poll=lambda:None)
        r.committed(dict(prediction_id=0,observation_step=0,observation_monotonic=9.,actions=actions),
                    dict(commit_step=0,commit_monotonic=9.5))
    queue={i:(a,0,0,i) for i,a in enumerate(actions)}
    args=dict(now=11.,step=33,queue=queue,rows=[],inflight=False,rtc_delay_steps=10)
    a=original.decide(**args);b=candidate.decide(**args)
    assert a['start_now'] and a['compute_profile']=='gpu_emc'
    assert b['start_now'] and b['compute_profile']=='baseline'
    assert b['latest_start']==a['latest_start']
    assert b['deadline']==a['deadline'] and b['margin_seconds']==a['margin_seconds']
    assert b['rtc_delay_steps']==10 and b['rtc_latest_submit_step']==40
    assert b['service_seconds']<a['service_seconds']
    assert b['start_constraint']=='full_energy_fixed_prepare'


def test_same_frequency_pair_changes_only_energy_flag():
    from perceptbudget.device_quality import configure_frequency_variant,frequency_pair_views
    cfg=dict(m3_full_energy=True,m3_three_frequency=True,m3_start_floor_ms=25,H=50,N=5)
    assert frequency_pair_views(cfg,0)==('full_three_medium','full_three_energy')
    for name,enabled in [('full_three_medium',False),('full_three_energy',True)]:
        child=deepcopy(cfg);configure_frequency_variant(child,name)
        assert child['m3_full_energy'] is enabled
        assert child['m3_start_floor_ms']==25 and child['H']==50 and child['N']==5
