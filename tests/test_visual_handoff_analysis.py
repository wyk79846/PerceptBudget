import json
import numpy as np
import pytest
from scripts.analyze_visual_handoffs import handoff_metrics,latest_label


def test_disagreement_aligns_same_target_step_not_old_chunk_start():
    requests=[dict(prediction_id=0,observation_step=0,actions=[[i]*7 for i in range(10)],context_refresh={'mode':'full'}),
              dict(prediction_id=1,observation_step=2,actions=[[10+i]*7 for i in range(10)],context_refresh={'mode':'pool2'})]
    controls=[dict(step=4,action=[4]*7,source={'prediction_id':0}),
              dict(step=5,action=[13]*7,source={'prediction_id':1}),
              dict(step=6,action=[14]*7,source={'prediction_id':1})]
    rows,summary=handoff_metrics(controls,requests)
    assert len(rows)==1 and rows[0]['transition']=='full->pool2'
    assert rows[0]['arm_delta_l2']==pytest.approx(np.sqrt(6)*9)
    assert rows[0]['same_target_step_prediction_disagreement_arm_l2']==pytest.approx(np.sqrt(6)*8)
    assert summary[0]['count']==1


def test_latest_label_does_not_silently_select_old_success_or_child(tmp_path):
    for name,time,purpose in (('old','1','fresh-pooled-alternation'),('new','2','fresh-pooled-alternation'),
                              ('child','3','fast-quality-pool2_alternate')):
        p=tmp_path/name;p.mkdir()
        (p/'run.json').write_text(json.dumps(dict(created_at_utc=time,purpose=purpose,
            config_snapshot={'m2_test_label':'less-fast'})))
    assert latest_label(tmp_path,'less-fast')==tmp_path/'new'
