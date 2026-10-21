import copy
import hashlib
import json

import numpy as np
import pytest

from scripts import run_m1_h50 as batch


def evidence(tmp_path,name,changes=(),scene_change=False):
    directory=tmp_path/name/'rtc'
    directory.mkdir(parents=True)
    frame=np.zeros((256,512,3),dtype=np.uint8)
    for y,x,value in changes:frame[y,x,:]=value
    np.save(directory/'q10_frames_spool.npy',frame[None])
    before=dict(sim_state=[0.],raw_state=[0.]*8,observations={'position':int(scene_change)},contacts=[],sim_time=0.)
    (directory/'trace.jsonl').write_text(json.dumps(dict(event='control_step',step=0,before=before))+'\n')
    return dict(run=str(directory.parent),outcome='timeout',initial_hashes=dict(
        initial_sim_state_sha256='a'*64,initial_raw_state_sha256='b'*64,
        initial_frames_sha256=hashlib.sha256(frame.tobytes()).hexdigest()))


def test_single_pixel_lsb_is_explicitly_qualified_not_identical(tmp_path):
    a=evidence(tmp_path,'baseline');b=evidence(tmp_path,'candidate',[(82,307,1)])
    proof=batch.compare_initial(a,b)
    assert proof['passed'] and proof['changed_pixels']==1 and proof['max_channel_delta']==1
    assert 'not bit-identical' in proof['limitation']
    assert batch.compare_initial(a,a) is None


@pytest.mark.parametrize('problem',['state','scene','large_delta','many_pixels','corrupt_frame'])
def test_tolerance_rejects_real_or_unverified_changes(tmp_path,problem):
    a=evidence(tmp_path,'baseline')
    changes=[(82,307,1)]
    if problem=='large_delta':changes=[(82,307,2)]
    if problem=='many_pixels':changes=[(0,x,1) for x in range(5)]
    b=evidence(tmp_path,'candidate',changes,scene_change=problem=='scene')
    if problem=='state':b['initial_hashes']['initial_sim_state_sha256']='c'*64
    if problem=='corrupt_frame':b['initial_hashes']['initial_frames_sha256']='d'*64
    with pytest.raises(ValueError):batch.compare_initial(a,b)


def test_review_retains_recorded_timeout_without_new_trial(tmp_path,monkeypatch):
    a=evidence(tmp_path,'baseline');b=evidence(tmp_path,'candidate',[(82,307,1)])
    baseline='libero_goal_task06_q5_sync_no_rtc';candidate='libero_goal_task06_q10_actual_history'
    saved=dict(frozen={'H':50},reference='ref',reference_hash='refhash',
               source_hashes={'scripts/run_m1_h50.py':'old','solver.py':'same'})
    current=copy.deepcopy(saved);current['source_hashes']['scripts/run_m1_h50.py']='new'
    index=dict(provenance=saved,completed=1,cells={
        baseline:dict(status='completed',result=a),
        candidate:dict(status='invalid',parent=str(tmp_path),cell=['libero_goal',6,10,'actual_history'],
                       error='initial sim/frame/raw-state mismatch: libero_goal/6',attempts=[{'exit_code':0}])})
    monkeypatch.setattr(batch,'validate',lambda *args,**kwargs:b)
    monkeypatch.setattr(batch,'independent_audit',lambda *args:dict(passed=True))
    reviewed=batch.migrate_render_review(index,current)
    assert reviewed['completed']==2
    assert reviewed['cells'][candidate]['result']['outcome']=='timeout'
    assert len(reviewed['cells'][candidate]['attempts'])==1
    assert reviewed['render_lsb_review']['original_record']['status']=='invalid'
    assert index['cells'][candidate]['status']=='invalid'
    current['source_hashes']['solver.py']='changed'
    with pytest.raises(ValueError,match='scheduler'):batch.migrate_render_review(index,current)
