"""One-step original expert/VJP replay, retaining author source arithmetic.

Exact prefix shape only. No attention replacement, padding or precision change.
Unlike a five-step graph, the expert time is a live scalar tensor input.
"""
import copy
import math


class SourceStepGraph:
    def __init__(self,expert,noise,*,pool=None,stream=None):
        import torch
        if getattr(expert,'_condition_cache',None) is not None:
            raise ValueError('step graph gate must use original uncached conditioning')
        expert.enable_graph_constants()
        self.times=[x.detach().clone() for x in expert.times]
        self.expert=copy.copy(expert)
        self.expert.times=[self.times[0].clone(),*self.times[1:]]
        self.context=(expert.keys,expert.values,expert.prefix_mask)
        self.binding=tuple(x.data_ptr() for x in self.context)
        self.x=noise.detach().clone().requires_grad_(True)
        self.target=torch.zeros_like(noise)
        self.mask=torch.zeros(expert.horizon,1,device=noise.device)
        self.tau=torch.zeros((),device=noise.device)
        self.beta=torch.tensor(5.,device=noise.device)
        stream=stream if stream is not None else torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2): self._step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool,stream=stream),torch.enable_grad():
            self.outputs=self._step()
        torch.cuda.synchronize()

    def _step(self):
        import torch
        with torch.enable_grad():
            x=self.x.detach().requires_grad_(True)
            velocity=-self.expert.velocity(x,0)
            clean=x+velocity*(1-self.tau)
            residual=((self.target-clean)*self.mask).detach()
            correction,=torch.autograd.grad(clean,x,grad_outputs=residual)
            inv_r2=(self.tau**2+(1-self.tau)**2)/((1-self.tau)**2)
            c=torch.nan_to_num((1-self.tau)/self.tau,posinf=5.)
            gain=torch.minimum(c*inv_r2,self.beta)
            updated=(x+(1/self.expert.num_steps)*(velocity+gain*correction)).detach()
        return updated,-velocity.detach(),clean.detach(),residual,correction.detach()

    def replay(self,noise,previous,delay,executed,*,trace=None):
        import torch
        h=self.expert.horizon
        if not 0<=delay<=h-executed or previous.shape!=(h-executed,32):
            raise ValueError('invalid source-step RTC overlap')
        if tuple(x.data_ptr() for x in (self.expert.keys,self.expert.values,self.expert.prefix_mask))!=self.binding:
            raise RuntimeError('source-step context replaced')
        if noise.shape!=self.x.shape:
            raise ValueError('invalid source-step noise shape')
        i=torch.arange(h,device=noise.device,dtype=torch.float32)
        end=h-executed
        start=torch.minimum(torch.tensor(delay,device=noise.device),torch.tensor(end,device=noise.device))
        weights=((start-1-i)/(end-start+1)+1).clamp(0,1)
        weights=torch.where(i>=end,0,weights*torch.expm1(weights)/(math.e-1))[:,None]
        norms=[]
        with torch.no_grad():
            self.x.copy_(noise)
            self.target.zero_();self.target[:len(previous)].copy_(previous)
            self.mask.copy_(weights);self.tau.zero_()
            for k,t in enumerate(self.times):
                self.expert.times[0].copy_(t)
                before=self.x.detach().clone() if trace is not None else None
                self.graph.replay()
                out,v,clean,error,g=self.outputs
                if trace is not None:
                    trace.append(dict(input=before,velocity=v.clone(),clean=clean.clone(),
                        error=error.clone(),vjp=g.clone(),output=out.clone()))
                norms.append(g.norm())
                self.x.copy_(out)
                self.tau.add_(1/self.expert.num_steps)
        return self.x.detach(),torch.stack(norms)


class SourceStepBank:
    """Sequential exact-length graphs sharing scratch pool; miss uses fallback.

    Only setup may build graphs. No runtime padding, recapture or LRU eviction.
    Returned buffers must be consumed before any other graph is replayed.
    """
    def __init__(self,expert,lengths,noise,*,progress=None):
        import time
        import torch
        from perceptbudget.openpi_reference import sample_original_rtc
        lengths=tuple(sorted(set(lengths)))
        if not lengths or len(lengths)>8 or any(type(n) is not int or not 1<=n<=592 for n in lengths):
            raise ValueError('prepare between one and eight exact prefix lengths')
        self.fallback=expert
        self.entries={}
        self.audits=[]
        pool=torch.cuda.graph_pool_handle()
        stream=torch.cuda.Stream()
        for valid in lengths:
            started=time.monotonic()
            bound=copy.copy(expert)
            bound.keys=torch.zeros(18,1,1,valid,256,device=noise.device,dtype=expert.keys.dtype)
            bound.values=torch.zeros_like(bound.keys)
            bound.prefix_mask=torch.ones(1,valid,device=noise.device,dtype=torch.bool)
            count=min(valid,expert.keys.shape[-2])
            bound.keys[...,:count,:].copy_(expert.keys[...,:count,:])
            bound.values[...,:count,:].copy_(expert.values[...,:count,:])
            reference,reference_norms=sample_original_rtc(bound,noise,noise[5:].clone(),5,5,author_order=True)
            graph=SourceStepGraph(bound,noise,pool=pool,stream=stream)
            got,norms=graph.replay(noise,noise[5:].clone(),5,5)
            raw_error=float((reference-got).abs().max())
            norm_error=float((norms-torch.tensor(reference_norms,device=noise.device)).abs().max())
            passed=math.isfinite(raw_error) and math.isfinite(norm_error) and max(raw_error,norm_error)<1e-4
            record=dict(valid=valid,raw_max_abs=raw_error,vjp_norm_max_abs=norm_error,
                passed=passed,prepare_seconds=time.monotonic()-started,
                allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20)
            self.audits.append(record)
            if progress is not None: progress(record)
            if not passed: raise RuntimeError('source-step bank parity failed')
            self.entries[valid]=(bound,graph)

    def select(self,valid):
        return self.entries.get(valid,(self.fallback,None))
