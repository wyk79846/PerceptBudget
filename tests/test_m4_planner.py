from perceptbudget.m4_planner import plan
from test_m4_candidates import CFG,WAIT,services

TRANS={('baseline','gpu_emc'):{'seconds':.25,'energy_j':3},('gpu_emc','baseline'):{'seconds':.25,'energy_j':4}}


def test_complete_plans_have_one_refresh_and_identical_window():
    r=plan(CFG,services(),WAIT,current_profile='baseline',quiet_age_seconds=.5,window_seconds=2,
        deadline_seconds=2,transitions=TRANS)
    assert r['selected'] and len(r['plans'])<=18
    for p in r['plans']:
        assert sum(s['kind']=='service' for s in p['segments'])==1
        assert abs(sum(s['seconds'] for s in p['segments'])-2)<1e-8
        assert abs(sum(s['energy_j'] for s in p['segments'])-p['energy_j'])<1e-8
        assert p['service_start_seconds']+p['service_p95_seconds']+.025<=2+1e-8
    # No Continue-only cheap option: equivalent deferrals include a real Full.
    deferred=[p for p in r['plans'] if p['policy_type'].startswith('Continue') or
        any(label.startswith('Continue') for label in p.get('equivalent_labels',[]))]
    assert deferred
    assert all(p['energy_j']>=12 for p in deferred)


def test_now_and_latest_are_compared_and_bootstrap_never_deferred():
    kwargs=dict(current_profile='baseline',quiet_age_seconds=.5,window_seconds=2,deadline_seconds=2,transitions=TRANS)
    r=plan(CFG,services(),WAIT,**kwargs)
    assert any(p['label'].endswith('@now') for p in r['plans'])
    assert any(p['label'].endswith('@latest') for p in r['plans'])
    r=plan(CFG,services(),WAIT,bootstrap=True,**kwargs)
    assert all(p['service_mode']=='Full' and p['prepare_start_seconds']==0 for p in r['plans'])


def test_busy_and_unreachable_deadline_do_not_create_extra_inference():
    kwargs=dict(current_profile='baseline',quiet_age_seconds=.5,window_seconds=1,transitions=TRANS)
    assert plan(CFG,services(),WAIT,deadline_seconds=1,inflight=True,**kwargs)['status']=='waiting_for_inflight'
    r=plan(CFG,services(),WAIT,deadline_seconds=.1,unexpected_change=True,**kwargs)
    assert r['selected'] is None
