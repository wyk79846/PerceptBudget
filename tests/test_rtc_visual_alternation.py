from types import SimpleNamespace
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from perceptbudget.rtc_visual_alternation import FreshViewRTCAdapter
from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter


def test_fresh_visual_alternation_encodes_selected_current_image_and_state(monkeypatch):
    calls=[]
    def predict(self, inputs, prompt, previous, delay, executed, request):
        calls.append((self.pipeline.num_views, inputs['images'], inputs['state']))
        request['visual_context_seconds']=.01
        return np.zeros((50,7))
    monkeypatch.setattr(OriginalOpenPiRTCAdapter, '_predict', predict)
    adapter=FreshViewRTCAdapter.__new__(FreshViewRTCAdapter)
    def model(views):
        return SimpleNamespace(_pipe=SimpleNamespace(pipeline=SimpleNamespace(
            num_views=views,vision_seq_enc=256*views)))
    adapter._full_model=model(2)
    adapter.attach_wrist(model(1))
    adapter.config={'m2_visual_fast_enabled':True}
    adapter._guided_requests=0
    for episode in range(2):
        modes=[]
        for i in range(5):
            inputs={'images':[np.full((2,2,3),i),np.full((2,2,3),100+i)],'state':np.array([i])}
            planned=adapter.context_mode('task',i*5,guided=i>0)
            request={'observation_step':i*5,'prediction_id':i,'planned_context_mode':planned}
            adapter._predict(inputs,'task',None if i==0 else np.zeros((45,32)),5,5,request)
            info=request['context_refresh']
            modes.append(info['mode'])
            assert info['current_images_encoded'] and info['current_state_encoded']
            assert info['source_prediction_id']==i and info['source_observation_step']==i*5
            assert 'kv_recompute' not in request
            np.testing.assert_array_equal(calls[-1][2],inputs['state'])
            assert len(calls[-1][1])==(1 if planned=='wrist' else 2)
            np.testing.assert_array_equal(calls[-1][1][-1],inputs['images'][1])
        assert modes==['full','wrist','full','wrist','full']
    assert len(calls)==10  # Every request runs fresh context computation.
    with pytest.raises(RuntimeError,match='Planned visual'):
        adapter._predict(inputs,'task',np.zeros((45,32)),5,5,{'planned_context_mode':'full'})


def test_visual_cross_suite_cli_uses_fresh_wrist_not_old_kv():
    script=Path(__file__).resolve().parents[1]/'scripts/eval_expert_recompute.py'
    result=subprocess.run([sys.executable,str(script),'--visual-alternation','--cross-suite','--dry-run'],
                          capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['episodes']==6 and plan['model_loads']==1
    assert plan['conditions']==['full','wrist_alternate']
    assert plan['visual_alternation'] and plan['q']==5 and plan['N']==5
    assert plan['fast_inputs']=='fresh wrist image and current state'
