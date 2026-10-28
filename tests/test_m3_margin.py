import copy
import json
from pathlib import Path
from perceptbudget.m3_margin import scan_settings,configure_margin
from perceptbudget.closed_loop import comparison_variants


def test_margin_scan_rotates_three_candidates_without_changing_default():
    root=Path(__file__).resolve().parents[1]
    source=json.loads((root/'m3.json').read_text());before=copy.deepcopy(source)
    settings=scan_settings(source)
    assert source==before and settings['timing']['service_quantile']==.95
    assert settings['timing']['maximum_action_offset']==50
    assert settings['timing']['maximum_observation_age_seconds'] is None
    assert len(settings['calibration_tasks'])==3
    from perceptbudget.deadline_controller import SafetyMargin
    first=[]
    for task in range(3):
        labels=comparison_variants(dict(m3_margin_scan=True,m3_trigger_task_index=task))
        first.append(labels[0])
        assert len(labels)==3
        for label in labels:
            cfg=dict(m3_timing=copy.deepcopy(settings['timing']))
            repeat=int(label.split('__repeat')[1])
            ms=configure_margin(cfg,repeat)
            assert ms==(50,25,0)[repeat-1]
            assert cfg['m3_timing']['safety_margin_seconds']==ms/1000
            assert cfg['m3_dvfs_policy']=='joint'
            assert SafetyMargin(initial=ms/1000.,floor=0.).seconds==ms/1000.
    assert len(set(first))==3
