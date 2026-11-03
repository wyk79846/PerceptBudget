import pytest
from perceptbudget.service_ledger import build_ledger


def episode():
    return dict(including_drain_start_monotonic=0.,including_drain_stop_monotonic=4.,
        commits=[dict(prediction_id=0,commit_monotonic=1.),dict(prediction_id=1,commit_monotonic=3.)],
        including_drain_power=dict(raw_readings=[dict(timestamp_monotonic=i/10,power_mw=10000.,error=None) for i in range(41)]))


def requests():
    return [dict(prediction_id=0,observation_monotonic=0.,m3_fast_choice=dict(mode='Full')),
            dict(prediction_id=1,observation_monotonic=2.2,m3_timing=dict(service_start_monotonic=2.),m3_fast_choice=dict(mode='Fast'))]


def test_partition():
    ledger=build_ledger(episode(),requests(),forbid_hot_fast=True)
    assert ledger['total_energy_j']==pytest.approx(40.)
    assert ledger['totals']['Continue']['seconds']==pytest.approx(2.)
    assert ledger['totals']['Recovery Fast']['covered_energy_j']==pytest.approx(10.)


def test_overlap_and_hot_fast_rejected():
    r=requests();r[1]['m3_timing']['service_start_monotonic']=.5
    with pytest.raises(ValueError,match='Overlapping'):build_ledger(episode(),r)
    r=requests();r[1].pop('m3_timing')
    with pytest.raises(ValueError,match='Hot Fast'):build_ledger(episode(),r,forbid_hot_fast=True)


def test_missing_samples_and_uncommitted_drain():
    e=episode();e['commits'].pop();r=requests();r[1]['ready_monotonic']=3.
    e['including_drain_power']['raw_readings'][25]['power_mw']=None
    ledger=build_ledger(e,r)
    assert ledger['total_energy_j'] is None
    assert not ledger['totals']['Recovery Fast']['complete']
    assert ledger['partition_verified']


def test_saved_q30_ledger_reads_the_q30_request_file(tmp_path):
    import json
    from perceptbudget.service_ledger import write_run_ledger
    (tmp_path/'rtc').mkdir()
    (tmp_path/'run.json').write_text(json.dumps({'config_snapshot':{'candidate_q':[30]}}))
    (tmp_path/'summary.json').write_text(json.dumps({'episode':episode()}))
    (tmp_path/'rtc/q30_requests.json').write_text(json.dumps(requests()))
    result=write_run_ledger(tmp_path)
    assert result['partition_verified']
    assert result['totals']['Continue']['covered_energy_j']==pytest.approx(20.)
