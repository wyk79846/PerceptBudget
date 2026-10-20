"""Opt-in original-precision AdaRMS fusion, not old BF16-style normalization."""
import ctypes
import hashlib
from pathlib import Path
import subprocess

_LIB=None


def load(output_root):
    global _LIB
    import torch
    if _LIB is not None:return
    src=Path(__file__).with_suffix('.cu')
    major,minor=torch.cuda.get_device_capability()
    flags=['-O3','--fmad=false','-shared','-Xcompiler','-fPIC',f'-arch=sm_{major}{minor}']
    key=hashlib.sha256(src.read_bytes()+repr(flags).encode()).hexdigest()[:16]
    directory=Path(output_root)/'build'/f'rtc-norm-{key}'
    directory.mkdir(parents=True,exist_ok=True)
    library=directory/'rtc_norm.so'
    if not library.exists():
        subprocess.run(['/usr/local/cuda/bin/nvcc',*flags,str(src),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library))
    p,i=ctypes.c_void_p,ctypes.c_int
    lib.pb_norm_forward.argtypes=[p,p,p,p,i,i,ctypes.c_float,p]
    lib.pb_norm_backward.argtypes=[p,p,p,p,p,i,i,p]
    lib.pb_norm_forward.restype=lib.pb_norm_backward.restype=i
    _LIB=lib


def fused_norm(x,modulation,eps):
    import torch
    # Define the autograd class lazily to keep project imports lightweight.
    class Norm(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,style):
            if (_LIB is None or not x.is_cuda or x.dtype!=torch.bfloat16 or
                x.ndim!=3 or x.shape[0]!=1 or not x.is_contiguous() or
                style.dtype!=torch.float32 or style.numel()!=3*x.shape[-1] or
                style.device!=x.device or not style.is_contiguous() or style.requires_grad):
                raise ValueError('fused norm requires contiguous batch-one BF16 hidden and frozen FP32 style')
            y=torch.empty_like(x)
            r=torch.empty(x.shape[1],device=x.device,dtype=torch.float32)
            code=_LIB.pb_norm_forward(x.data_ptr(),style.data_ptr(),y.data_ptr(),r.data_ptr(),
                x.shape[1],x.shape[2],eps,torch.cuda.current_stream().cuda_stream)
            if code:raise RuntimeError(f'norm forward launch failed: {code}')
            ctx.save_for_backward(x,style,r)
            return y

        @staticmethod
        def backward(ctx,dy):
            x,style,r=ctx.saved_tensors
            dy=dy.contiguous()
            dx=torch.empty_like(x)
            code=_LIB.pb_norm_backward(x.data_ptr(),style.data_ptr(),r.data_ptr(),dy.data_ptr(),dx.data_ptr(),
                x.shape[1],x.shape[2],torch.cuda.current_stream().cuda_stream)
            if code:raise RuntimeError(f'norm backward launch failed: {code}')
            return dx,None
    return Norm.apply(x,modulation)


def enable_fused_norm(expert,output_root):
    load(output_root)
    model=expert.suffix.paligemma_with_expert.gemma_expert.model
    if model.training or any(p.requires_grad for p in expert.suffix.parameters()):
        raise ValueError('fused norm requires frozen eval expert')
    modules=[]
    for layer in model.layers:modules.extend((layer.input_layernorm,layer.post_attention_layernorm))
    modules.append(model.norm)
    if any(m.dense is None for m in modules):raise ValueError('adaptive norms required')
    original=[m.forward for m in modules]
    for module in modules:
        def forward(x,cond=None,m=module):
            if cond is None or cond.requires_grad:
                raise ValueError('fused norm needs frozen time conditioning')
            style=m.dense(cond)
            out=fused_norm(x,style,m.eps)
            gate=style[...,2*x.shape[-1]:].to(x.dtype).unsqueeze(1)
            return out,gate
        module.forward=forward

    def restore():
        for m,forward in zip(modules,original):m.forward=forward
    return restore,dict(modules=len(modules),style_dtype='float32',hidden_dtype='bfloat16',
                        backward='full input VJP; separately rounded direct/variance branches')


def audit_fused_norm_same_input(expert,output_root,*,beta=5.):
    from perceptbudget.rtc_exact_flash_probe import audit_expert_change
    def install():return enable_fused_norm(expert,output_root)[0]
    return audit_expert_change(expert,install,beta=beta,backend='fused_norm_on_flash')
