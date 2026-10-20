"""Opt-in tiny CUDA VJP library; build outside repo before graph capture."""
import ctypes
import hashlib
from pathlib import Path
import subprocess

_LIB = None


def load(output_root):
    global _LIB
    import torch
    if _LIB is not None:
        return
    source = Path(__file__).with_suffix('.cu')
    major, minor = torch.cuda.get_device_capability()
    flags = ['-O3','--fmad=false','-shared','-Xcompiler','-fPIC',f'-arch=sm_{major}{minor}']
    key = hashlib.sha256(source.read_bytes()+repr(flags).encode()).hexdigest()[:16]
    directory = Path(output_root)/'build'/f'rtc-backward-{key}'
    directory.mkdir(parents=True,exist_ok=True)
    library = directory/'rtc_backward.so'
    if not library.exists():
        subprocess.run(['/usr/local/cuda/bin/nvcc',*flags,str(source),'-o',str(library)],check=True)
    lib = ctypes.CDLL(str(library))
    p,i = ctypes.c_void_p,ctypes.c_int
    lib.rtc_ada_dx.argtypes = [p,p,p,p,i,i,p]
    lib.rtc_geglu_dx.argtypes = [p,p,p,p,p,i,p]
    lib.rtc_geglu_tanh_dx.argtypes = [p,p,p,p,p,i,p]
    lib.rtc_residual.argtypes = [p,p,p,p,i,i,i,p]
    lib.rtc_qkv_rope.argtypes = [p,p,p,p,p,i,p]
    lib.rtc_qkv_rope_dx.argtypes = [p,p,p,p,p,i,p]
    lib.rtc_ada_dx.restype = lib.rtc_geglu_dx.restype = i
    lib.rtc_geglu_tanh_dx.restype = i
    _LIB = lib


def _validate(*tensors):
    import torch
    if _LIB is None:
        raise RuntimeError('load fused backward before capture')
    if any(t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous()
           or t.device != tensors[0].device for t in tensors):
        raise ValueError('fused backward requires contiguous same-device CUDA BF16 tensors')


def ada_backward(x,style,dy):
    import torch
    dy = dy.contiguous()
    _validate(x,style,dy)
    if x.ndim!=2 or style.shape!=(x.shape[0],3*x.shape[1]) or dy.shape!=x.shape:
        raise ValueError('invalid Ada VJP shapes')
    out = torch.empty_like(x)
    code = _LIB.rtc_ada_dx(x.data_ptr(),style.data_ptr(),dy.data_ptr(),out.data_ptr(),
                           *x.shape,torch.cuda.current_stream().cuda_stream)
    if code:
        raise RuntimeError(f'Ada VJP CUDA launch failed: {code}')
    return out


def geglu_backward(gate,up,dy):
    import torch
    dy = dy.contiguous()
    _validate(gate,up,dy)
    if gate.shape!=up.shape or gate.shape!=dy.shape:
        raise ValueError('invalid GeGLU VJP shapes')
    dg,du = torch.empty_like(gate),torch.empty_like(up)
    code = _LIB.rtc_geglu_dx(gate.data_ptr(),up.data_ptr(),dy.data_ptr(),dg.data_ptr(),
                             du.data_ptr(),gate.numel(),torch.cuda.current_stream().cuda_stream)
    if code:
        raise RuntimeError(f'GeGLU VJP CUDA launch failed: {code}')
    return dg,du


def geglu_tanh_backward(gate,up,dy):
    """tanh-form GELU VJP matching the pi05 suffix's gelu_pytorch_tanh."""
    import torch
    dy = dy.contiguous()
    _validate(gate,up,dy)
    if gate.shape!=up.shape or gate.shape!=dy.shape:
        raise ValueError('invalid GeGLU tanh VJP shapes')
    dg,du = torch.empty_like(gate),torch.empty_like(up)
    code = _LIB.rtc_geglu_tanh_dx(gate.data_ptr(),up.data_ptr(),dy.data_ptr(),dg.data_ptr(),
                                  du.data_ptr(),gate.numel(),torch.cuda.current_stream().cuda_stream)
    if code:
        raise RuntimeError(f'GeGLU tanh VJP CUDA launch failed: {code}')
    return dg,du


def residual(x,projected,style,backward=False):
    import torch
    _validate(x,projected,style)
    if x.ndim!=2 or projected.shape!=x.shape or style.shape!=(x.shape[0],3*x.shape[1]):
        raise ValueError('invalid residual shapes')
    out=torch.empty_like(x)
    code=_LIB.rtc_residual(x.data_ptr(),projected.data_ptr(),style.data_ptr(),out.data_ptr(),
                          *x.shape,int(backward),torch.cuda.current_stream().cuda_stream)
    if code: raise RuntimeError(f'residual CUDA error {code}')
    return out


def qkv_rope(x,rope):
    import torch
    _validate(x,rope)
    if x.ndim!=2 or x.shape[1]!=2560 or rope.shape!=(x.shape[0],256):
        raise ValueError('invalid QKV/RoPE shapes')
    rows=x.shape[0]
    q=torch.empty((rows,8,256),device=x.device,dtype=x.dtype)
    k=torch.empty((rows,1,256),device=x.device,dtype=x.dtype)
    v=torch.empty_like(k)
    code=_LIB.rtc_qkv_rope(x.data_ptr(),rope.data_ptr(),q.data_ptr(),k.data_ptr(),v.data_ptr(),
                          rows,torch.cuda.current_stream().cuda_stream)
    if code: raise RuntimeError(f'QKV RoPE CUDA error {code}')
    return q,k,v


def qkv_rope_backward(dq,dk,dv,rope):
    import torch
    dq,dk,dv=(v.contiguous() for v in (dq,dk,dv))
    _validate(dq,dk,dv,rope)
    rows=rope.shape[0]
    if dq.shape!=(rows,8,256) or dk.shape!=(rows,1,256) or dv.shape!=dk.shape:
        raise ValueError('invalid QKV gradient shapes')
    out=torch.empty((rows,2560),device=dq.device,dtype=dq.dtype)
    code=_LIB.rtc_qkv_rope_dx(dq.data_ptr(),dk.data_ptr(),dv.data_ptr(),rope.data_ptr(),
                             out.data_ptr(),rows,torch.cuda.current_stream().cuda_stream)
    if code: raise RuntimeError(f'QKV RoPE backward CUDA error {code}')
    return out
