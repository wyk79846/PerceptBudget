"""Actual CUPTI kernel timing, scoped to one preparation/recovery pair."""
import json
from pathlib import Path
import time

from scripts.probe_gpu_start_timing import first_kernel_offset


def timing_from_trace(trace,marker,*,cpu_bracket_ns,issued,ready):
    kernel=first_kernel_offset(trace,marker)
    low,high=(n/1e9 for n in cpu_bracket_ns)
    if high<low or ready<issued:
        raise ValueError('Invalid CPU monotonic clock bracket')
    delta=kernel['first_kernel_after_marker_us']/1e6
    first_low,first_high=low+delta,high+delta
    first=(first_low+first_high)/2
    if first_high<issued or first_low>ready:
        raise ValueError('Actual kernel timestamp does not fit request interval')
    return dict(**kernel,request_to_first_kernel_ms=(first-issued)*1000,
        request_to_first_kernel_bounds_ms=[(first_low-issued)*1000,(first_high-issued)*1000],
        first_kernel_to_actions_ready_ms=(ready-first)*1000,
        request_to_actions_ready_ms=(ready-issued)*1000,
        cpu_marker_bracket_us=(high-low)*1e6,
        limitation='CPU marker entry bracket plus CUPTI-correlated trace timestamps; '
                   'bounds do not quantify CUPTI clock-correlation error or profiling overhead')


class PairCollector:
    def __init__(self,adapter,directory,torch):
        self.original=adapter.predict
        self.directory=Path(directory);self.torch=torch;self.prof=None;self.index=0

    def predict(self,*args,**kwargs):
        request=kwargs['request']
        if request.get('timing_reference'):
            return self.original(*args,**kwargs)
        torch=self.torch
        if 'caller_request_monotonic' not in request:
            if self.prof is not None:raise RuntimeError('Unmatched profiling preparation')
            self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                         torch.profiler.ProfilerActivity.CUDA])
            self.prof.start()
            try:return self.original(*args,**kwargs)
            except BaseException:
                self.close();raise
        if self.prof is None:raise RuntimeError('Recovery profiler was not armed during preparation')
        marker='PB_MODEL_RECOVERY_'+str(self.index)
        before=time.monotonic_ns()
        try:
            with torch.profiler.record_function(marker):
                inside=time.monotonic_ns()
                out=self.original(*args,**kwargs)
            self.prof.stop()
            path=self.directory/(marker+'.json')
            self.prof.export_chrome_trace(str(path))
            trace=json.loads(path.read_text())
            timing=timing_from_trace(trace,marker,cpu_bracket_ns=[before,inside],
                issued=request['caller_request_monotonic'],ready=request['inference_ended_monotonic'])
            timing['trace_path']=str(path)
            request['gpu_start_timing']=timing
            self.prof=None;self.index+=1
            return out
        except BaseException:
            self.close();raise

    def close(self):
        if self.prof is not None:
            try:self.prof.stop()
            except RuntimeError:pass
            self.prof=None
