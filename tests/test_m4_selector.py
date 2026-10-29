from perceptbudget.m4_selector import Candidate,Segment,choose


def refresh(label,mode='Full',joules=12,profile='baseline',complete=True):
    return Candidate(label,mode,profile,'baseline',.2,.5,
        (Segment('hold',.2,2),Segment('service',.5,joules),Segment('hold',.3,3)),
        costs_complete=complete)


def test_quality_filters_before_energy_and_does_not_use_unknown_costs():
    result=choose([refresh('full'),refresh('fast','Fast',7),refresh('unknown',joules=0,complete=False)],
        window_seconds=1,deadline_seconds=1)
    assert result['selected']['label']=='full'
    assert {r['reason'] for r in result['rejected']}=={'quality','missing_measurement'}
    result=choose([refresh('full'),refresh('fast','Fast',7)],window_seconds=1,deadline_seconds=1,fast_allowed=True)
    assert result['selected']['label']=='fast' and result['selected']['energy_j']==12


def test_equal_window_can_favor_fast_high_compute_followed_by_low_hold():
    high=Candidate('fast-high','Fast','baseline','gpu_emc',0,.4,
        (Segment('service',.35,7.8),Segment('transition',.05,.7),Segment('hold',.6,4.2)))
    low=Candidate('slow-low','Fast','gpu_emc','gpu_emc',0,.6,
        (Segment('service',.5,7.0),Segment('hold',.5,7.0)))
    result=choose([high,low],window_seconds=1,deadline_seconds=1,fast_allowed=True)
    assert result['selected']['label']=='fast-high'
    assert result['selected']['energy_j']==12.7


def test_deadline_uses_p95_and_margin_not_shorter_mean_service_duration():
    result=choose([refresh('full')],window_seconds=1,deadline_seconds=.71,margin_seconds=.025)
    assert result['selected'] is None and result['rejected'][0]['reason']=='deadline'


def test_continue_is_not_always_low_and_cannot_extend_buffer():
    high=Candidate('hold-high','Continue','none','baseline',None,None,(Segment('hold',1,8),))
    low=Candidate('switch-low','Continue','none','gpu_emc',None,None,
        (Segment('transition',.1,2),Segment('hold',.9,7)))
    assert choose([high,low],window_seconds=1,deadline_seconds=2)['selected']['label']=='hold-high'
    assert choose([high],window_seconds=1,deadline_seconds=.8)['selected'] is None


def test_inconsistent_window_or_service_start_is_rejected():
    c=refresh('full')
    assert choose([c],window_seconds=2,deadline_seconds=3)['rejected'][0]['reason']=='window_mismatch'
    c=Candidate('wrong','Full','baseline','baseline',0,.5,c.segments)
    assert choose([c],window_seconds=1,deadline_seconds=2)['rejected'][0]['reason']=='service_start_mismatch'
