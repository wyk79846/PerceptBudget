#!/usr/bin/env python3
"""Tiny actual-kernel timestamp permission probe; no VLA model or clock writes.

May be run with sudo ONLY for this tiny probe. Does not grant permanent
profiling privileges, change driver configuration, or authorize a root model.
"""
import json
import os
from pathlib import Path
import tempfile
import time


def first_kernel_offset(trace,marker):
    events=trace.get('traceEvents',[])
    ranges=[e for e in events if e.get('name')==marker and e.get('ph')=='X']
    if len(ranges)!=1:
        raise RuntimeError('Missing unique CPU marker in profiler trace')
    r=ranges[0]
    kernels=[e for e in events if e.get('cat')=='kernel' and
             r['ts']<=e.get('ts',-1)<r['ts']+r['dur']]
    if not kernels:
        raise RuntimeError('No actual GPU kernel timestamps captured; CPU markers are not GPU-start evidence')
    first=min(kernels,key=lambda e:e['ts'])
    return dict(first_kernel_name=first['name'],kernel_count=len(kernels),
                first_kernel_after_marker_us=first['ts']-r['ts'],
                first_kernel_duration_us=first.get('dur'))


def main():
    import torch
    result=dict(status='running',effective_uid=os.geteuid(),model_loaded=False,
                clock_settings_changed=False,persistent_permissions_changed=False,
                scope='Tiny CUDA graph timestamp capability probe only; not VLA startup measurement')
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError('CUDA profiler activities unavailable')
        stream=torch.cuda.Stream()
        a=torch.ones((128,128),device='cuda')
        with torch.cuda.stream(stream):
            for _ in range(3):b=a@a
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):b=a@a
        torch.cuda.synchronize()
        marker='PB_ACTUAL_GPU_START_PERMISSION_PROBE'
        with tempfile.TemporaryDirectory(prefix='perceptbudget-gpu-start-') as tmp:
            output=Path(tmp)/'trace.json'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                before=time.monotonic_ns()
                with torch.profiler.record_function(marker):
                    inside=time.monotonic_ns()
                    graph.replay()
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(output))
            result.update(first_kernel_offset(json.loads(output.read_text()),marker))
            result.update(status='completed',cpu_marker_clock_bracket_ns=[before,inside],
                limitation='Profiler adds overhead; probe duration is not VLA startup cost')
    except Exception as exc:
        result.update(status='unavailable',error=f'{type(exc).__name__}: {exc}')
    print('GPU_START_PROBE='+json.dumps(result),flush=True)
    return 0 if result['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
