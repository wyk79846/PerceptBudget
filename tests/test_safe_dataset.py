from collections import Counter

import pytest

from perceptbudget.safe_dataset import TRAINING_QUOTAS, select_training240


def inputs():
    episodes=[]
    for suite in TRAINING_QUOTAS:
        for task in range(7):
            for j in range(9 if suite=='libero_10' else 6):
                episodes.append(dict(episode=len(episodes),source_run=f'old/{suite}/{task}/{j}',
                    suite=suite,task_id=task,safe_split='train',failure=j==0))
    episodes.extend([
        dict(episode=len(episodes),source_run='blocked-validation',suite='libero_10',task_id=0,safe_split='validation',failure=False),
        dict(episode=len(episodes)+1,source_run='blocked-test',suite='libero_10',task_id=7,safe_split='test',failure=True)])
    candidates=[dict(run=f'new/{task}/{j}',suite='libero_10',task_id=task,method='candidate37',outcome='success')
                for task in range(7) for j in range(20)]
    candidates.extend([
        dict(run='blocked-validation',suite='libero_10',task_id=0,method='candidate37'),
        dict(run='blocked-test',suite='libero_10',task_id=7,method='candidate37'),
        dict(run='unseen',suite='libero_10',task_id=8,method='candidate37'),
        dict(run='q10',suite='libero_10',task_id=0,method='q10')])
    return episodes,candidates


def test_suite_quotas_retain_failures_and_exclude_validation_test_and_q10():
    episodes,candidates=inputs()
    retained,added=select_training240(episodes,candidates)
    counts=Counter(e['suite'] for e in retained)
    counts.update(r['suite'] for r in added)
    assert counts==TRAINING_QUOTAS
    assert all(e['safe_split']=='train' for e in retained)
    assert all(e['task_id']<7 for e in retained+added)
    assert not {'blocked-validation','blocked-test','unseen','q10'} & {r['run'] for r in added}
    assert {e['source_run'] for e in episodes if e['safe_split']=='train' and e['failure']} <= {e['source_run'] for e in retained}
    long_counts=Counter(e['task_id'] for e in retained if e['suite']=='libero_10')
    long_counts.update(r['task_id'] for r in added)
    assert [long_counts[i] for i in range(7)]==[18,17,17,17,17,17,17]


def test_selection_is_reproducible_without_using_new_outcomes():
    episodes,candidates=inputs()
    a,b=select_training240(episodes,candidates)
    changed=[dict(r,outcome='timeout') for r in reversed(candidates)]
    c,d=select_training240(episodes,changed)
    assert [e['source_run'] for e in a]==[e['source_run'] for e in c]
    assert [r['run'] for r in b]==[r['run'] for r in d]


def test_insufficient_long_data_does_not_borrow_validation_or_duplicate_samples():
    episodes,candidates=inputs()
    with pytest.raises(ValueError,match='Insufficient unused Long'):
        select_training240(episodes,[r for r in candidates if r['task_id']!=0])


def test_rebalancing_does_not_shift_validation_evaluation_time():
    from perceptbudget.safe_paper import evaluation_cutoffs
    episodes=[dict(suite='libero_10',task_id=0,steps=242),dict(suite='libero_spatial',task_id=6,steps=101)]
    assert evaluation_cutoffs(episodes)=={('libero_10',0):242,('libero_spatial',6):101}
    original={'libero_10:0':244,'libero_spatial:6':98}
    assert evaluation_cutoffs(episodes,original)=={('libero_10',0):244,('libero_spatial',6):98}
    with pytest.raises(ValueError,match='same tasks'):
        evaluation_cutoffs(episodes,{'libero_10:0':244})
