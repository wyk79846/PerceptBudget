import json
import pytest
from perceptbudget.rtc_initial_delay import reference_delay


def test_reference_keeps_initialization_separate_from_online_delay(tmp_path):
    run=tmp_path/'episode';run.mkdir()
    cfg=dict(H=50,N=5,state_enabled=True,precision='bfloat16',checkpoint_path='model',rtc_initial_delay_steps=10)
    (run/'run.json').write_text(json.dumps({'config_snapshot':dict(cfg,rtc_initial_delay_steps=9)}))
    (tmp_path/'recording.json').write_text(json.dumps({'episodes':[{'suite':'libero_10','task_id':3,'run':str(run)}]}))
    record=reference_delay(tmp_path,'libero_10',3,cfg)
    assert record['initial_delay_steps']==9
    assert cfg['rtc_initial_delay_steps']==10
    assert 'rtc_fixed_delay_estimate' not in cfg
    with pytest.raises(ValueError,match='configuration differs'):
        reference_delay(tmp_path,'libero_10',3,dict(cfg,state_enabled=False))
    with pytest.raises(ValueError,match='exactly one'):
        reference_delay(tmp_path,'libero_10',0,cfg)
