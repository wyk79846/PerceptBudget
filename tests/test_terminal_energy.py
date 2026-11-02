from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from perceptbudget import terminal_energy as terminal
from perceptbudget.service_ledger import build_ledger
from scripts import probe_delay_buffer as probe


def test_terminal_low_waits_for_ack_then_holds_without_restoring(monkeypatch):
    clock=SimpleNamespace(now=10.)
    monkeypatch.setattr(terminal.time,'monotonic',lambda:clock.now)
    monkeypatch.setattr(terminal.time,'sleep',lambda s:setattr(clock,'now',clock.now+s))
    clocks=SimpleNamespace(profile='baseline',pending=None)
    def switch(profile):clocks.pending=profile
    def poll():
        if clocks.pending is not None and clock.now>=10.02:
            clocks.profile=clocks.pending;clocks.pending=None
    clocks.switch=Mock(side_effect=switch);clocks.poll=poll
    row=terminal.settle_low(clocks)
    assert row['stop_monotonic']-row['low_acknowledged_monotonic']>=3.
    assert clocks.profile=='gpu_emc'
    clocks.switch.assert_called_once_with('gpu_emc')


def test_tail_is_separate_from_continue_and_partitions_total():
    readings=[dict(timestamp_monotonic=t,power_mw=2000.) for t in (0.,1.,2.,3.,4.,5.)]
    e=dict(including_drain_start_monotonic=0.,including_drain_stop_monotonic=5.,
        including_drain_power={'raw_readings':readings},commits=[],terminal_low={'start_monotonic':2.})
    ledger=build_ledger(e,[],max_gap_seconds=1.1)
    assert ledger['totals']['Continue']['covered_energy_j']==4.
    assert ledger['totals']['Terminal LOW tail']['covered_energy_j']==6.
    assert ledger['total_energy_j']==10.


def test_drain_stops_energy_at_low_endpoint_not_later_restoration(monkeypatch):
    monkeypatch.setattr(probe.time,'monotonic',lambda:99.)
    monitor=Mock()
    monkeypatch.setattr(probe.base,'_power_result',lambda m,a,b:{'total_energy_mj':(b-a)*2000})
    tail=dict(start_monotonic=12.,stop_monotonic=15.)
    result=probe._drain_executor(Mock(),monitor,10.,11.,terminal_low=tail)
    assert result['including_drain_stop_monotonic']==15.
    assert result['terminal_low']['task_plus_inference_power']['total_energy_mj']==4000.
    assert result['terminal_low']['power']['total_energy_mj']==6000.
    assert result['including_drain_power']['total_energy_mj']==10000.
    monitor.stop.assert_called_once()


def test_pair_report_does_not_call_final_commit_executed(tmp_path):
    import json
    from perceptbudget.full_frequency_pairs import summarize_pairs
    episodes=[]
    for group in ('full_high','full_low'):
        run=tmp_path/group;(run/'rtc').mkdir(parents=True)
        episode=dict(real_seconds=4.,steps=76,power={'total_energy_mj':10000},
            including_drain_power={'total_energy_mj':18000},
            terminal_low={'power':{'total_energy_mj':6000},'task_plus_inference_power':{'total_energy_mj':12000}},
            commits=[{'prediction_id':0,'commit_step':0},{'prediction_id':1,'commit_step':76}])
        (run/'summary.json').write_text(json.dumps({'episode':episode}))
        (run/'rtc/q5_requests.json').write_text(json.dumps([{'prediction_id':i} for i in (0,1)]))
        episodes.append(dict(run=str(run),suite='libero_spatial',task_id=0,view=group,outcome='success',steps=76))
    rows=summarize_pairs(episodes,tmp_path)['rows']
    assert all(r['unexecuted_requests']==1 for r in rows)
    assert all(r['task_plus_inference_j']+r['terminal_low_j']==r['common_endpoint_j'] for r in rows)
