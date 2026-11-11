from types import SimpleNamespace

import numpy as np
import torch

from perceptbudget.safe_paper import aggregate_corners,compact_loss,compact_weights,conformal_bands,make_model,paper_split


def test_corner_combinations_follow_author_horizon_then_denoising_order():
    corners=np.array([[[1.,2.],[3.,4.],[5.,6.],[7.,8.]]])
    assert aggregate_corners(corners,'first','concat').tolist()==[[1.,2.,5.,6.]]
    assert aggregate_corners(corners,'concat','last').tolist()==[[5.,6.,7.,8.]]
    assert aggregate_corners(corners,'concat','concat').tolist()==[[1.,2.,3.,4.,5.,6.,7.,8.]]


def test_compact_loss_and_gradient_match_unmodified_author_loss_with_variable_lengths():
    torch.manual_seed(11)
    model=make_model(4,'cpu')
    episodes=[dict(failure=False),dict(failure=True),dict(failure=False)]
    lengths=[3,1,2];features=torch.randn(3,3,4)
    masks=torch.arange(3)[None,:]<torch.tensor(lengths)[:,None]
    batch=dict(features=features,valid_masks=masks.float(),success_labels=torch.tensor([1,0,1]))
    expected,_=model.forward_compute_loss(batch,[3/(1+1),3/(2+1)])
    expected_grad=torch.autograd.grad(expected,tuple(model.parameters()))
    flat=torch.cat([features[i,:n] for i,n in enumerate(lengths)])
    weights=compact_weights(episodes,{i:list(range(n)) for i,n in enumerate(lengths)},[0,1,2],'cpu')
    actual,_=compact_loss(model,flat,weights,0.)
    actual_grad=torch.autograd.grad(actual,tuple(model.parameters()))
    assert torch.allclose(actual,expected,atol=1e-7)
    for a,b in zip(actual_grad,expected_grad):assert torch.allclose(a,b,atol=1e-7)


def test_seen_validation_uses_separate_rollouts_and_unseen_tasks_stay_out_of_training():
    episodes=[dict(episode=task*5+j,suite='suite',task_id=task,failure=False) for task in range(10) for j in range(5)]
    splits=paper_split(episodes)
    assert len(splits['train'])==21 and len(splits['validation'])==14 and len(splits['test'])==15
    assert not set(splits['train'])&set(splits['validation'])
    assert {episodes[i]['task_id'] for i in splits['train']}=={episodes[i]['task_id'] for i in splits['validation']}
    assert {episodes[i]['task_id'] for i in splits['test']}=={7,8,9}


def test_functional_band_does_not_fit_held_out_scores():
    curves=[np.linspace(.1,.8,8)*(1+i/10) for i in range(12)]
    original=conformal_bands(curves,list(range(10)))
    curves[-1]=curves[-1]+1000
    changed=conformal_bands(curves,list(range(10)))
    for alpha in original:
        assert np.isfinite(original[alpha]).all()
        assert np.array_equal(original[alpha],changed[alpha])


def test_success_augmentation_preserves_all_existing_split_memberships():
    episodes=[dict(episode=task*5+j,suite='suite',task_id=task,failure=False) for task in range(10) for j in range(5)]
    original=paper_split(episodes)
    for role,ids in original.items():
        for order,i in enumerate(ids):episodes[i].update(safe_split=role,safe_split_order=order)
    episodes.append(dict(episode=50,suite='suite',task_id=0,failure=False,safe_split='train',safe_split_order=len(original['train'])))
    episodes.append(dict(episode=51,suite='suite',task_id=1,failure=False,safe_split='validation',safe_split_order=len(original['validation'])))
    current=paper_split(episodes)
    assert current['test']==original['test']
    for role in ('train','validation'):
        assert [i for i in current[role] if i<50]==original[role]
