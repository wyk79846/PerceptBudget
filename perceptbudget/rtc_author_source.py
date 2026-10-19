"""Run author's unchanged RTC function body with a narrow Torch backend.

Not JAX/XLA validation or a copy of the author's policy. Only single-sample
vmap, fixed supplied noise, scan and input VJP are adapted to Torch so the
original OpenPI expert can be used without installing JAX or training.
"""
import ast
import functools
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace


def sample_author_source(expert, noise, previous, delay, executed, *, source, beta=5.):
    import torch
    if previous.shape != (expert.horizon-executed,32) or not 0 <= delay <= len(previous):
        raise ValueError('invalid author-source overlap')
    rows=[]
    current={}
    class Arrays:
        e=math.e
        def arange(self,n): return torch.arange(n,device=noise.device,dtype=torch.float32)
        def minimum(self,a,b):
            return torch.minimum(torch.as_tensor(a,device=noise.device),torch.as_tensor(b,device=noise.device))
        def clip(self,x,a,b): return x.clamp(a,b)
        def where(self,c,a,b): return torch.where(c,torch.as_tensor(a,device=noise.device),torch.as_tensor(b,device=noise.device))
        def expm1(self,x): return torch.expm1(x)
        def ones(self,n): return torch.ones(n,device=noise.device)
        def nan_to_num(self,x,**kwargs): return torch.nan_to_num(x,**kwargs)
    def vjp(fn,x,has_aux):
        assert has_aux
        x=x.detach().requires_grad_(True)
        with torch.enable_grad():
            clean,velocity=fn(x)
        current.update(input=x.detach().clone(),clean=clean.detach().clone(),velocity=-velocity.detach().clone())
        def pullback(error):
            correction,=torch.autograd.grad(clean,x,grad_outputs=error.detach())
            current.update(error=error.detach().clone(),vjp=correction.detach().clone())
            return (correction,)
        return clean,pullback,velocity
    def vmap(fn,in_axes):
        assert in_axes == (0,0,0,None)
        def mapped(obs,x,y,t):
            assert len(obs)==1
            return fn(obs[0],x[0],y[0],t)[None]
        return mapped
    def scan(fn,carry,length):
        carry=(carry[0],torch.tensor(carry[1],device=noise.device,dtype=torch.float32))
        for step in range(length):
            current.clear()
            (x,t),_=fn(carry,None)
            current['output']=x[0].detach().clone()
            rows.append(dict(current))
            carry=(x.detach(),t.detach())
        return carry,None
    path=Path(source)
    tree=ast.parse(path.read_text())
    mask=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_prefix_weights')
    action=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body
                if isinstance(n,ast.FunctionDef) and n.name=='realtime_action')
    scope=dict(jnp=Arrays(),functools=functools,
        jax=SimpleNamespace(vjp=vjp,vmap=vmap,lax=SimpleNamespace(scan=scan),
            random=SimpleNamespace(normal=lambda rng,shape:noise.detach().clone()[None])))
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,mask,action],type_ignores=[])),str(path),'exec'),scope)
    class Policy:
        action_chunk_size=expert.horizon
        action_dim=32
        simulated_delay=None
        def __call__(self,obs,x,t):
            # Author uses clean-at-one time; OpenPI uses clean-at-zero.
            return -expert.velocity(x[0],len(rows))[None]
    target=torch.zeros_like(noise)
    target[:len(previous)]=previous
    raw=scope['realtime_action'](Policy(),None,torch.zeros(1,1,device=noise.device),
        expert.num_steps,target[None],delay,expert.horizon-executed,'exp',beta)[0]
    weights=scope['get_prefix_weights'](delay,expert.horizon-executed,expert.horizon,'exp')
    return raw,rows,weights,hashlib.sha256(path.read_bytes()).hexdigest()
