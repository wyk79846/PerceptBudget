import pytest
from perceptbudget.m4_candidates import build,hold_energy,templates

CFG=dict(compute_profiles=['baseline','balanced','gpu_emc'],hold_profiles=['baseline','gpu_emc'],timing={'safety_margin_seconds':.025})
WAIT={'baseline':[16,12,10,9],'gpu_emc':[14,10,8,7]}


def services():
    out=[]
    for mode in ('Full','Fast'):
        for initial,profile,history in [('baseline','baseline','hot'),('gpu_emc','baseline','short'),('gpu_emc','baseline','long'),('gpu_emc','gpu_emc','hot')]:
            out.append(dict(mode=mode,profile=profile,from_profile=initial,history=history,
                p95_service_seconds=.6 if mode=='Full' else .4,mean_service_seconds=.5 if mode=='Full' else .3,
                mean_absolute_service_energy_j=12 if mode=='Full' else 8))
    return out


@pytest.mark.parametrize('state,count',[('baseline',3),('gpu_emc',5)])
def test_fourteen_templates_but_only_measured_state_paths_enabled(state,count):
    r=build(CFG,services(),WAIT,current_profile=state,quiet_age_seconds=.3,window_seconds=1,deadline_seconds=.9)
    assert len(templates(CFG))==r['template_count']==14
    assert r['measured_count']==count
    assert all(c['missing_reason'] for c in r['catalog'] if not c['costs_complete'])
    assert not any(c['mode']=='Continue' for c in r['candidates'])
    assert r['selected']['mode']=='Fast'


def test_critical_events_bootstrap_and_consecutive_fast_force_full():
    for flags in ({'critical_seconds':.625},{'bootstrap':True},{'consecutive_fast':1},{'unexpected_change':True}):
        r=build(CFG,services(),WAIT,current_profile='baseline',quiet_age_seconds=.3,window_seconds=1,deadline_seconds=.9,**flags)
        assert not r['fast_allowed'] and r['selected']['mode']=='Full'


def test_continue_leaves_enough_time_for_next_refresh():
    args=dict(current_profile='baseline',quiet_age_seconds=.3,window_seconds=.3)
    r=build(CFG,services(),WAIT,deadline_seconds=.6,**args)
    assert not any(c['mode']=='Continue' for c in r['candidates'])
    r=build(CFG,services(),WAIT,deadline_seconds=2,**args)
    assert r['selected']['mode']=='Continue'


def test_wait_cost_integrates_age_bins_without_resetting_age():
    assert hold_energy(WAIT,'baseline',.1,.5)==pytest.approx(.1*16+.3*12+.1*10)
    assert hold_energy(WAIT,'baseline',.6,.2)==pytest.approx(2)


@pytest.mark.parametrize('state',['baseline','gpu_emc'])
def test_directional_preparation_enables_all_ten_high_low_policies(state):
    transitions={('baseline','gpu_emc'):{'seconds':.25,'energy_j':3},
                 ('gpu_emc','baseline'):{'seconds':.25,'energy_j':4}}
    result=build(CFG,services(),WAIT,current_profile=state,quiet_age_seconds=.5,window_seconds=2,
        deadline_seconds=2,transitions=transitions)
    assert result['measured_count']==10
    missing=[c for c in result['catalog'] if not c['costs_complete']]
    assert len(missing)==4 and all(':balanced->' in c['label'] for c in missing)
    down=next(c for c in result['catalog'] if c['label']=='Full:baseline->gpu_emc')
    assert down['costs_complete']
    assert sum(s['seconds'] for s in down['segments'])==pytest.approx(2)
    assert sum(s['kind']=='transition' for s in down['segments'])==1
    # Low->high Full recovery is in service already; only the post-service down is charged.
    assert next(s['energy_j'] for s in down['segments'] if s['kind']=='transition')==3


def test_warm_post_compute_up_window_is_not_priced_as_cold_wait_recovery():
    transitions={('baseline','gpu_emc'):{'seconds':.25,'energy_j':3,'reference_quiet_age_seconds':0.},
        ('gpu_emc','baseline'):{'seconds':.25,'energy_j':2,'reference_quiet_age_seconds':1.}}
    r=build(CFG,services(),WAIT,current_profile='gpu_emc',quiet_age_seconds=1.,window_seconds=2,
        deadline_seconds=2,transitions=transitions)
    candidate=next(c for c in r['catalog'] if c['label']=='Full:gpu_emc->baseline')
    # Current warm low-side quarter second: .2*14+.05*10=3.3J;
    # cold reference: .25*7=1.75J. Only the positive history difference is added.
    energy=next(s['energy_j'] for s in candidate['segments'] if s['kind']=='transition')
    assert energy==pytest.approx(2+3.3-1.75)
