import pytest
from scripts.update_m4_costs import aggregate, continue_segments


def test_continue_partition_keeps_acknowledged_profiles_and_all_elapsed_time():
    clocks=[dict(status='READY',monotonic=0.,profile='baseline'),
            dict(status='APPLIED',observed_match_monotonic=1.,profile='deep_idle')]
    segments=list(continue_segments(.6,2.6,clocks,[.5]))
    assert sum(b-a for a,b,_,_ in segments)==pytest.approx(2.)
    assert segments[0][0]==.6 and segments[-1][1]==2.6
    assert all(left[1]==right[0] for left,right in zip(segments,segments[1:]))
    assert all(p=='baseline' for a,b,p,h in segments if b<=1.)
    assert all(p=='deep_idle' for a,b,p,h in segments if a>=1.)
    assert {h for a,b,p,h in segments if p=='deep_idle'}=={
        'under_0.2s','0.2_to_1s','at_least_1s'}


def test_continue_power_uses_time_weighting_not_average_of_window_powers():
    rows=[dict(mode='Continue',profile='deep_idle',from_profile='deep_idle',
               history='at_least_1s',run='a',seconds=t,energy_j=10.) for t in (1.,3.)]
    cell=aggregate(rows)[0]
    assert cell['mean_power_w']==5.
    assert cell['total_energy_j']==20. and cell['mean_request_energy_j'] is None


def test_bootstrap_is_separate_from_guided_full_service_and_source_profile():
    rows=[dict(mode=m,profile='baseline',from_profile=p,history=h,run='a',
               seconds=t,energy_j=j) for m,p,h,t,j in [
        ('Bootstrap','baseline','bootstrap',.8,12.),
        ('Full','baseline','under_0.2s',.5,8.),
        ('Full','deep_idle','at_least_1s',.55,9.)]]
    cells=aggregate(rows)
    assert len(cells)==3
    assert sorted(c['mean_request_energy_j'] for c in cells if c['mode']=='Full')==[8.,9.]


def test_continue_after_high_and_medium_full_are_not_mixed():
    rows=[dict(mode='Continue',profile='deep_idle',from_profile='deep_idle',
        history='under_0.2s',after_full_profile=source,run='a',seconds=1.,energy_j=j)
        for source,j in [('baseline',12.),('gpu_emc',9.)]]
    cells=aggregate(rows)
    assert len(cells)==2 and sorted(c['mean_power_w'] for c in cells)==[9.,12.]
