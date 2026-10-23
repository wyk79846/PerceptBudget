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
from collections import Counter


def trace_diagnostics(trace,marker):
    events=trace.get('traceEvents',[])
    return dict(trace_event_count=len(events),
        category_counts=dict(Counter(str(e.get('cat')) for e in events)),
        phase_counts=dict(Counter(str(e.get('ph')) for e in events)),
        marker_events=[e for e in events if marker in str(e.get('name',''))],
        actual_kernel_count=sum(e.get('cat')=='kernel' for e in events),
        event_name_examples=list(dict.fromkeys(str(e.get('name')) for e in events))[:15])


def first_kernel_offset(trace,marker):
    events=trace.get('traceEvents',[])
    ranges=[e for e in events if e.get('name')==marker and e.get('ph')=='X'
            and e.get('cat')=='user_annotation']
    if len(ranges)!=1:
        raise RuntimeError(f'Expected one complete CPU marker, found {len(ranges)}; inspect saved trace and diagnostic counts')
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
        # Keep this tiny probe's trace even if marker/kernel parsing fails.
        # A sudo-created private random directory is made readable intentionally:
        # it contains only these synthetic matrix events, never model/user data.
        tmp=Path(tempfile.mkdtemp(prefix='perceptbudget-gpu-start-'))
        tmp.chmod(0o755)
        output=tmp/'trace.json'
        result['trace_path']=str(output)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            # The first CUPTI collection can initialize lazily. Exercise it
            # outside the measured marker, then mark a second replay explicitly.
            with torch.profiler.record_function('PB_PROFILER_WARMUP'):
                graph.replay();torch.cuda.synchronize()
            prof.step()
            before=time.monotonic_ns()
            with torch.profiler.record_function(marker):
                inside=time.monotonic_ns()
                graph.replay()
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(output));output.chmod(0o644)
        trace=json.loads(output.read_text())
        result['trace_diagnostics']=trace_diagnostics(trace,marker)
        result['profiler_event_count']=len(prof.events())
        result.update(first_kernel_offset(trace,marker))
        result.update(status='completed',cpu_marker_clock_bracket_ns=[before,inside],
            limitation='Profiler adds overhead; probe duration is not VLA startup cost')
    except Exception as exc:
        result.update(status='unavailable',error=f'{type(exc).__name__}: {exc}')
    if result.get('trace_path'):
        summary=Path(result['trace_path']).with_name('summary.json')
        summary.write_text(json.dumps(result,indent=2));summary.chmod(0o644)
    print('GPU_START_PROBE='+json.dumps(result),flush=True)
    return 0 if result['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
