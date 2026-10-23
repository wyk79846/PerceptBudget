from types import SimpleNamespace
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from perceptbudget.rtc_visual_alternation import FreshViewRTCAdapter
from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter


def test_dual_geometry_builds_two_bounded_banks_and_dispatches_without_fallback(monkeypatch):
    calls=[]
    class Bank:
        fallback='root'
        def __init__(self,lengths):
            self.entries={n:('expert'+str(n),'graph'+str(n)) for n in lengths}
        def select(self,valid):
            return self.entries.get(valid,(self.fallback,None))
    def prepare(self,lengths,*,progress=None):
        assert len(lengths)<=17
        calls.append((list(lengths),self._step_bank_stream))
        self.step_bank=Bank(lengths)
        return dict(passed=True,prepare_seconds=1.,lengths=list(lengths))
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'prepare_step_bank',prepare)
    adapter=FreshViewRTCAdapter.__new__(FreshViewRTCAdapter)
    adapter.config={'source_prefix_coverage_rule':'h50_state8_tokenizer_bounds_v1'}
    adapter._step_bank_stream=object()
    full=list(range(650,667)); wrist=list(range(394,411))
    result=adapter.prepare_step_bank(full+wrist)
    assert result['passed'] and len(result['lengths'])==34
    assert [r[0] for r in calls]==[full,wrist]
    assert calls[0][1] is calls[1][1]
    assert adapter.step_bank.select(665)==('expert665','graph665')
    assert adapter.step_bank.select(409)==('expert409','graph409')
    assert adapter.step_bank.select(500)==('root',None)
    with pytest.raises(ValueError,match='at most17'):
        adapter.prepare_step_bank(list(range(650,668))+wrist)


@pytest.mark.parametrize('fast_mode',['wrist','pool2','pool2_agent'])
@pytest.mark.parametrize('fast_no_state',[False,True])
def test_fresh_visual_alternation_encodes_selected_current_image_and_state(monkeypatch,fast_mode,fast_no_state):
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
    adapter.config={'m2_visual_fast_enabled':True,'m2_visual_fast_mode':fast_mode,'m2_fast_no_state':fast_no_state}
    from perceptbudget.rtc_visual_alternation import visual_geometry
    geometry=visual_geometry(fast_mode)
    fast=model(geometry['num_views'])
    fast._pipe.pipeline.vision_seq_enc=geometry['visual_tokens']
    fast._pipe.pipeline.vision_pool_factor=geometry['vision_pool_factor']
    adapter.attach_wrist(fast)
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
            assert info['current_images_encoded']
            assert info['current_state_encoded']==(planned=='full' or not fast_no_state)
            assert info['source_prediction_id']==i and info['source_observation_step']==i*5
            assert 'kv_recompute' not in request
            if planned!='full' and fast_no_state:
                assert calls[-1][2] is None
            else:
                np.testing.assert_array_equal(calls[-1][2],inputs['state'])
            assert inputs['state'] is not None  # Never mutate snapshots used by Full.
            assert len(calls[-1][1])==(1 if planned=='wrist' else 2)
            np.testing.assert_array_equal(calls[-1][1][-1],inputs['images'][1])
        assert modes==['full',fast_mode,'full',fast_mode,'full']
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


def test_unseen_visual_prefix_stops_before_expert_without_padding(monkeypatch):
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'_prepare_context',lambda *a,**k:500)
    adapter=FreshViewRTCAdapter.__new__(FreshViewRTCAdapter)
    adapter.step_bank=SimpleNamespace(select=lambda valid: ('fallback',None))
    with pytest.raises(RuntimeError,match='stopped before expert'):
        adapter._prepare_context({},'task',{},guided=True)


def test_visual_subset_does_not_repeat_completed_spatial_pair():
    script=Path(__file__).resolve().parents[1]/'scripts/eval_expert_recompute.py'
    result=subprocess.run([sys.executable,str(script),'--visual-alternation','--cross-suite',
        '--suites','libero_object','libero_goal','--dry-run'],capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['episodes']==4 and plan['model_loads']==1
    assert plan['task_plan']==[['libero_object',0],['libero_goal',0]]


def test_pooled_alternation_cli_is_six_fresh_dual_image_paired_episodes():
    script=Path(__file__).resolve().parents[1]/'scripts/eval_expert_recompute.py'
    result=subprocess.run([sys.executable,str(script),'--pooled-alternation','--cross-suite','--dry-run'],
        capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['episodes']==6 and plan['model_loads']==1
    assert plan['conditions']==['full','pool2_alternate'] and plan['fast_visual_mode']=='pool2'
    assert plan['q']==5 and plan['N']==5 and plan['state']
    assert 'fresh dual images' in plan['fast_inputs']


def test_mixed_geometry_metadata_does_not_claim_full_only_tokens():
    from perceptbudget.rtc_visual_alternation import episode_geometry
    mixed=episode_geometry('pool2',True)
    assert mixed['visual_tokens'] is None and mixed['vision_pool_factor'] is None
    assert mixed['geometry_by_mode']['full']['visual_tokens']==512
    assert mixed['geometry_by_mode']['pool2']['visual_tokens']==128
    assert mixed['geometry_by_mode']['pool2']['vision_pool_factor']==2
    assert episode_geometry('pool2',False)['visual_tokens']==512


def test_prefix_offset_guard_checks_native_actual_valid_length(monkeypatch):
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'_prepare_context',lambda *a,**kw:273)
    adapter=FreshViewRTCAdapter.__new__(FreshViewRTCAdapter)
    adapter.step_bank=None
    adapter.pipeline=SimpleNamespace(vision_seq_enc=128)
    adapter.pipe=SimpleNamespace(current_prompt_len=144)
    with pytest.raises(RuntimeError,match='prefix offset'):
        adapter._prepare_context({},'task',{},guided=True)


def test_two_full_one_fast_schedule_and_labeled_cli_plans():
    from perceptbudget.rtc_visual_alternation import scheduled_visual_mode
    assert [scheduled_visual_mode(i,'pool2',True,2) for i in range(9)]==['full','full','pool2']*3
    script=Path(__file__).resolve().parents[1]/'scripts/eval_expert_recompute.py'
    for options,mode,ratio in ((['--mixed-pooled-alternation','--label','gentle-compression'],'pool2_agent',1),
                              (['--pooled-alternation','--full-per-fast','2','--label','less-fast'],'pool2',2)):
        r=subprocess.run([sys.executable,str(script),*options,'--suite','libero_object','--dry-run'],
            capture_output=True,text=True,check=True)
        plan=json.loads(r.stdout)
        assert plan['episodes']==2 and plan['model_loads']==1 and plan['q']==5
        assert plan['fast_visual_mode']==mode and plan['full_per_fast']==ratio
