"""Project-side native selector; build once before loading/capturing model."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
_LIB=None


def load():
    global _LIB
    if _LIB is not None:return _LIB
    import torch
    source=Path(__file__).with_name('selective_pool.cu')
    major,minor=torch.cuda.get_device_capability()
    flags=['-O3','--fmad=false','-shared','-Xcompiler','-fPIC',f'-arch=sm_{major}{minor}']
    key=hashlib.sha256(source.read_bytes()+repr(flags).encode()).hexdigest()[:16]
    directory=Path(__file__).resolve().parents[2]/'任务产出/perceptbudget/build'/('selective-pool-'+key)
    directory.mkdir(parents=True,exist_ok=True);library=directory/'selective_pool.so'
    if not library.exists():subprocess.run(['/usr/local/cuda/bin/nvcc',*flags,str(source),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library))
    for name in ('selective_pool416','selective_pool320','selective_pool464'):
        function=getattr(lib,name)
        function.argtypes=[ctypes.c_void_p]*6
        function.restype=ctypes.c_int
    lib.selective_rope.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.selective_rope.restype=ctypes.c_int
    _LIB=lib;return lib
