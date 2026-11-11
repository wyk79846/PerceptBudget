"""SAFE author network/loss/functional CP with a saved-feature adapter.

Official source is vendored under third_party/safe (MIT). The compact training
loss below is algebraically the author's per-rollout mean loss: padding and
the cumulative-sum matrix can be omitted for the independent MLP.
"""
import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


CHOICES=('first','last','concat')
AUTHOR_ROOT=Path(__file__).resolve().parents[1]/'third_party/safe'


def aggregate_corners(features, horizon, denoise):
    """Horizon selection first, then diffusion selection, as in author loader."""
    import numpy as np
    if horizon not in CHOICES or denoise not in CHOICES:
        raise ValueError('Unknown SAFE feature selection')
    h=(0,) if horizon=='first' else (1,) if horizon=='last' else (0,1)
    k=(0,) if denoise=='first' else (1,) if denoise=='last' else (0,1)
    return np.concatenate([features[:,2*step+pos,:] for step in k for pos in h],axis=-1)


@lru_cache(maxsize=1)
def author_model_type():
    """Execute the original class without its Hydra/WandB framework imports."""
    import torch
    from third_party.safe.model.utils import get_time_weight,aggregate_monitor_loss
    class BaseModel(torch.nn.Module):
        def __init__(self,cfg,input_dim):
            super().__init__()
            self.cfg,self.input_dim=cfg,input_dim
    tree=ast.parse((AUTHOR_ROOT/'model/indep.py').read_text())
    selected=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IndepModel')
    scope=dict(torch=torch,nn=torch.nn,BaseModel=BaseModel,Config=SimpleNamespace,
        get_time_weight=get_time_weight,aggregate_monitor_loss=aggregate_monitor_loss)
    exec(compile(ast.Module(body=[selected],type_ignores=[]),str(AUTHOR_ROOT/'model/indep.py'),'exec'),scope)
    return scope['IndepModel']


def make_model(width,device):
    cfg=SimpleNamespace(model=SimpleNamespace(n_history_steps=1,hidden_dim=256,n_layers=2,
        final_act_layer='sigmoid',cumsum=True,rmean=False,use_time_weighting=False,
        use_threshold=False,threshold=50))
    return author_model_type()(cfg,width).to(device)


def compact_weights(episodes,indices_by_episode,train_ids,device):
    import torch
    counts={label:sum(episodes[i]['failure']==label for i in train_ids) for label in (False,True)}
    weights=[]
    for i in train_ids:
        n=len(indices_by_episode[i]);failed=episodes[i]['failure']
        sign=-1 if failed else 1
        weights.extend(sign*(n-j)/n/(counts[failed]+1) for j in range(n))
    return torch.tensor(weights,device=device,dtype=torch.float32)


def compact_loss(model,features,weights,regularization):
    scores=model.projector(features).flatten()
    monitor=(scores*weights).sum()
    penalty=sum(parameter.square().sum() for name,parameter in model.named_parameters() if 'bias' not in name)
    return monitor+regularization*penalty,monitor


def paper_split(episodes,seed=11):
    """Fixed 7/3 tasks per suite; author's seeded 60/40 rollout split on seen tasks."""
    import torch
    groups={}
    for e in episodes:
        groups.setdefault((e['suite'],e['task_id']),[]).append(e['episode'])
    generator=torch.Generator().manual_seed(seed)
    splits=dict(train=[],validation=[],test=[])
    for (_,task),ids in sorted(groups.items()):
        if task>=7:
            splits['test'].extend(ids)
        else:
            order=torch.randperm(len(ids),generator=generator).tolist()
            n=int(.6*len(ids))
            splits['train'].extend(ids[j] for j in order[:n])
            splits['validation'].extend(ids[j] for j in order[n:])
    return splits


def conformal_bands(curves,calibration_success_ids,*,alpha_values=(.1,.15,.2),seed=11):
    """Author Tfunc/Mean band, successful seen validation split 30/70."""
    import numpy as np
    from third_party.safe.conformal.functional_predictor import FunctionalPredictor,ModulationType,RegressionType
    size=max(len(s) for s in curves)
    cal=np.stack([np.pad(curves[i],(0,size-len(curves[i])),mode='edge') for i in calibration_success_ids])
    if len(cal)<4:
        raise ValueError('Need at least4 successful validation rollouts for functional CP')
    np.random.RandomState(seed).shuffle(cal)
    boundary=int(.3*len(cal))
    predictor=FunctionalPredictor(ModulationType.Tfunc,RegressionType.Mean)
    return {str(alpha):predictor.get_one_sided_prediction_band(cal[:boundary],cal[boundary:],alpha,
        lower_bound=False).reshape(-1) for alpha in alpha_values}
