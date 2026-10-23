import json
import subprocess
import sys
from pathlib import Path
import pytest
from perceptbudget.rtc_kv_reuse import RecomputeRTCAdapter
from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter


def test_alternation_keeps_state_and_source_metadata_stale_without_context_call(monkeypatch):
    seen=[]
    def context(self,snapshot,prompt,request,*,guided):
        seen.append(request.get('observation_step',0))
        return 650+len(seen)
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'_prepare_context',context)
    adapter=RecomputeRTCAdapter.__new__(RecomputeRTCAdapter)
    adapter.config=dict(m2_expert_recompute=True,m2_reuse_enabled=True)
    adapter.horizon=50
    requests=[]
    for i,step in enumerate((0,5,10,15)):
        request=dict(observation_step=step,prediction_id=i)
        adapter._prepare_context(dict(raw_state=[step]*8),'task',request,guided=bool(i))
        requests.append(request)
    assert seen==[0,10]
    assert [r['kv_recompute']['mode'] for r in requests]==['full','expert_only']*2
    assert requests[1]['kv_recompute']['source_raw_state']==[0.]*8
    assert requests[1]['kv_recompute']['current_state_encoded'] is False
    assert requests[3]['kv_recompute']['source_prediction_id']==2
    adapter.validate_execution(requests[1],49)
    with pytest.raises(RuntimeError):adapter.validate_execution(requests[1],50)
    # A new episode bootstrap must refresh even after expert-only inference.
    request=dict(observation_step=0,prediction_id=0)
    adapter._prepare_context(dict(raw_state=[1]*8),'task',request,guided=False)
    assert request['kv_recompute']['mode']=='full' and seen[-1]==0


def test_stale_guard_and_disabled_comparison_force_full(monkeypatch):
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'_prepare_context',lambda *a,**k:657)
    adapter=RecomputeRTCAdapter.__new__(RecomputeRTCAdapter)
    adapter.config=dict(m2_expert_recompute=True,m2_reuse_enabled=True);adapter.horizon=50
    for step,guided in ((0,False),(30,True)):
        r=dict(observation_step=step)
        adapter._prepare_context(dict(raw_state=[0]*8),'task',r,guided=guided)
        assert r['kv_recompute']['mode']=='full'
    adapter.config['m2_reuse_enabled']=False
    r=dict(observation_step=35)
    adapter._prepare_context(dict(raw_state=[0]*8),'task',r,guided=True)
    assert r['kv_recompute']['mode']=='full'
    adapter.config['m2_expert_recompute']=False
    with pytest.raises(ValueError):adapter._prepare_context({},'task',{},guided=True)


def test_recompute_cli_is_two_full_view_unpooled_paired_episodes():
    script=Path(__file__).resolve().parents[1]/'scripts/eval_expert_recompute.py'
    r=subprocess.run([sys.executable,str(script),'--dry-run'],capture_output=True,text=True,check=True)
    plan=json.loads(r.stdout)
    assert plan['episodes']==2 and plan['model_loads']==1
    assert plan['num_views']==2 and plan['pool']==1 and plan['q']==5
    assert plan['original_kv_expiry_enforced']


def test_mode_planning_is_pure_and_actual_execution_must_match(monkeypatch):
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'_prepare_context',lambda *a,**k:657)
    adapter=RecomputeRTCAdapter.__new__(RecomputeRTCAdapter)
    adapter.config=dict(m2_expert_recompute=True,m2_reuse_enabled=True);adapter.horizon=50
    adapter._prepare_context(dict(raw_state=[0]*8),'task',dict(observation_step=0),guided=False)
    assert adapter.context_mode('task',5,guided=True)=='expert_only'
    assert adapter.context_mode('task',5,guided=True)=='expert_only'
    with pytest.raises(RuntimeError,match='Planned KV mode'):
        adapter._prepare_context(dict(raw_state=[1]*8),'task',
            dict(observation_step=5,planned_kv_mode='full'),guided=True)
    assert adapter.context_mode('task',5,guided=True)=='expert_only'
    r=dict(observation_step=5,planned_kv_mode='expert_only')
    adapter._prepare_context(dict(raw_state=[1]*8),'task',r,guided=True)
    assert r['kv_recompute']['mode']=='expert_only'
    assert adapter.context_mode('task',10,guided=True)=='full'


def test_split_delay_comparison_is_three_episodes_without_changing_denoising():
    script=Path(__file__).resolve().parents[1]/'scripts/eval_expert_recompute.py'
    r=subprocess.run([sys.executable,str(script),'--split-delay','--dry-run'],
        capture_output=True,text=True,check=True)
    plan=json.loads(r.stdout)
    assert plan['episodes']==3 and plan['model_loads']==1
    assert plan['N']==5 and plan['conditions']==['full','kv_reuse','kv_reuse_split']


def test_single_insertion_is_consumed_once_and_resets_only_at_bootstrap(monkeypatch):
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'_prepare_context',lambda *a,**k:657)
    adapter=RecomputeRTCAdapter.__new__(RecomputeRTCAdapter)
    adapter.config=dict(m2_expert_recompute=True,m2_reuse_enabled=True,m2_single_reuse=True)
    adapter.horizon=50
    for episode in range(2):
        modes=[]
        for i,step in enumerate((0,5,10,15,20,25)):
            request=dict(observation_step=step,prediction_id=i)
            planned=adapter.context_mode('task',step,guided=bool(i))
            adapter._prepare_context(dict(raw_state=[step]*8),'task',request,guided=bool(i))
            assert request['kv_recompute']['mode']==planned
            modes.append(planned)
        assert modes==['full','expert_only','full','full','full','full']


def test_single_insertion_plan_is_two_episodes_with_separate_delay_histories():
    script=Path(__file__).resolve().parents[1]/'scripts/eval_expert_recompute.py'
    r=subprocess.run([sys.executable,str(script),'--single-insertion','--dry-run'],
        capture_output=True,text=True,check=True)
    plan=json.loads(r.stdout)
    assert plan['episodes']==2 and plan['model_loads']==1
    assert plan['conditions']==['full','kv_once'] and plan['split_delay_comparison']
    assert plan['single_insertion'] and plan['N']==5 and plan['q']==5
