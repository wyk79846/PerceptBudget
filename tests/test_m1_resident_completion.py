from types import SimpleNamespace
import numpy as np


def test_only_six_planned_disturbance_cases():
    from scripts.complete_m1_h50 import plan
    assert plan()==[(t,m,q) for t in (0,3) for m,q in
                   (('sync_no_rtc',5),('actual_history',5),('actual_history',40))]


def test_exceptional_resume_keeps_completed_cases_without_rerunning():
    import pytest
    from scripts.complete_m1_h50 import plan,pending_cases
    saved=[dict(task_id=t,mode=m,q=q) for t,m,q in plan()[:4]]
    assert pending_cases(saved)==[(3,'actual_history',5),(3,'actual_history',40)]
    with pytest.raises(ValueError):pending_cases(saved+saved[:1])


def test_guard_failure_propagates_to_service_exit_code(monkeypatch):
    import pathlib,runpy,subprocess,sys,pytest
    monkeypatch.setattr(sys,'argv',['complete_m1_h50.py','--main-batch','/tmp/not-read-by-parent'])
    monkeypatch.setattr(subprocess,'call',lambda *a,**k:125)
    script=pathlib.Path(__file__).resolve().parents[1]/'scripts/complete_m1_h50.py'
    with pytest.raises(SystemExit) as exc:runpy.run_path(str(script),run_name='__main__')
    assert exc.value.code==125


def test_initial_info_preserves_scene_and_frame_fingerprints(tmp_path):
    from scripts.complete_m1_h50 import initial_info
    import hashlib,struct
    ep=dict(initial_raw_state=[0.]*8,initial_sim_state_sha256='sim',initial_frames_sha256='frame')
    info=initial_info(tmp_path,ep)
    assert info['initial_hashes']['initial_sim_state_sha256']=='sim'
    assert info['initial_hashes']['initial_frames_sha256']=='frame'
    assert info['initial_hashes']['initial_raw_state_sha256']==hashlib.sha256(struct.pack('<8d',*[0.]*8)).hexdigest()


def test_repeated_bank_preparation_reuses_frozen_conditioning(monkeypatch):
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget import rtc_condition_cache,rtc_source_step_graph
    seen=[]
    streams=[]
    monkeypatch.setattr(rtc_condition_cache,'enable_condition_cache',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not reinstall')))
    def bank(expert,lengths,noise,**kwargs):
        seen.append(kwargs['max_lengths'])
        streams.append(kwargs['stream'])
        return SimpleNamespace(entries={n:None for n in lengths},audits=[])
    monkeypatch.setattr(rtc_source_step_graph,'SourceStepBank',bank)
    adapter=OriginalOpenPiRTCAdapter.__new__(OriginalOpenPiRTCAdapter)
    adapter.torch=SimpleNamespace(randn=lambda *a,**k:object(),float32='float32',
                                 cuda=SimpleNamespace(Stream=object))
    adapter.horizon=50;adapter.beta=5.
    adapter.expert=SimpleNamespace(_condition_cache={'existing':True},_packed_condition_cache=object())
    adapter.config=dict(source_packed_conditioning=True,source_prefix_coverage_rule='h50_state8_tokenizer_bounds_v1')
    for _ in range(2):
        result=adapter.prepare_step_bank(range(645,662))
        assert result['packed_conditioning']['reused'] is True
    assert seen==[17,17]
    assert streams[0] is streams[1]
