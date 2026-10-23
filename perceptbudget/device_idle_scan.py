"""Fixed-frequency idle history characterization, never a hardware-state claim."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import statistics
import time

from perceptbudget.device_clock_screen import Lease,energy_window,frequency_check
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor

DELAYS=(0.,.05,.1,.2,.5,1.,2.)


def validate_recovery_power(window,*,startup_timing=False):
    complete=window['energy_j'] is not None
    if not complete and not startup_timing:
        raise RuntimeError('Recovery power coverage incomplete')
    return complete


def scan_plan(repeats=3):
    if repeats not in (1,3):
        raise ValueError('Idle scan supports one preflight or three measured blocks')
    rng=random.Random(7)
    plan=[]
    for block in range(repeats):
        values=list(DELAYS);rng.shuffle(values)
        plan.extend((block,value) for value in values)
    return plan


def describe(rows):
    out=[]
    for delay in DELAYS:
        group=[r for r in rows if r['requested_idle_seconds']==delay]
        if not group:continue
        tail=[r['idle_tail_mean_power_w'] for r in group if r['idle_tail_mean_power_w'] is not None]
        out.append(dict(requested_idle_seconds=delay,repeats=len(group),
            actual_idle_median_ms=statistics.median(r['actual_idle_seconds'] for r in group)*1000,
            recovery_median_ms=statistics.median(r['recovery_ms'] for r in group),
            recovery_min_ms=min(r['recovery_ms'] for r in group),
            recovery_max_ms=max(r['recovery_ms'] for r in group),
            idle_tail_mean_power_w=statistics.mean(tail) if tail else None,
            interpretation='Exploration point only; no state boundary inferred'))
    return out


class FixedScanLease(Lease):
    def __init__(self,directory):
        super().__init__('baseline',directory)
        self.duration=120

    def command(self):
        return super().command()+['--trace-interval','.02']


def runtime_status():
    # Runtime status is evidence, never equated with railgate from a timer.
    root=Path('/sys/devices/platform/bus@0/17000000.gpu/power')
    result={}
    for name in ('runtime_status','runtime_active_time','runtime_suspended_time'):
        try:result[name]=(root/name).read_text().strip()
        except OSError as exc:result[name]={'error':str(exc)}
    return result


def screen(adapter,snapshot,prompt,previous,torch,directory,metadata):
    import numpy as np
    from scripts import eval_q as base
    from scripts.eval_pooled_views import CLEANUP_GATE
    from perceptbudget.device_quality import install_render_buffer_cleanup
    from perceptbudget.continue_camera import suspend_cameras
    from perceptbudget.rtc_memory import memory_record
    directory=Path(directory)
    result=dict(metadata,status='running',rows=[],scope='Fixed high-frequency saved-input Full RTC idle scan; '
        'static simulation and ongoing proprioception/forward service, not action execution, live service L or safe buffer',
        requested_points_seconds=list(DELAYS),repeats=3,plan=scan_plan(),
        control_service_period_seconds=.05,hardware_state_classification=None,
        first_gpu_operator_timestamp=None,clock_settings_changed=False)
    if adapter.config.get('m2_gpu_start_trace'):
        result['scope']='Profiler-assisted request-to-first-real-GPU-kernel/action-return at fixed high clocks, saved Full inputs and static physical service; not uninstrumented latency, full live service or task energy'
        result['power_role']='Opportunistic diagnostics only; coverage gaps are retained as unavailable energy, never interpolated or used as calibrated energy'
    path=directory/'idle_scan.json'
    gate=json.loads((CLEANUP_GATE/'reset_reuse_audit.json').read_text())
    if not gate['passed'] or len(gate['cases'])!=6 or gate['render_cleanup']['errors']:
        raise RuntimeError('Require validated render cleanup gate')
    base._patch_egl_cleanup();install_render_buffer_cleanup(shared_egl=True)
    task=base._load_tasks([0],['libero_spatial'])[0]
    _,cls=base._libero_api()
    env=None;monitor=None;reference=None;ticks=[]
    try:
        base._seed_process(7,np,torch)
        env=base._make_env(cls,task);env._pb_lock_fixtures=True
        base._prepare_env_episode(env,task,np)
        original=env.env.sim.get_state().flatten().copy()
        mem=memory_record(torch,'idle_scan_ready');result['memory']=mem
        if mem['available_mib']<900:raise RuntimeError('Idle scan RAM below900MiB')
        def tick():
            start=time.monotonic()
            env.env.sim.forward()
            env.env._get_observations(force_update=True)
            if not np.array_equal(original,env.env.sim.get_state().flatten()):
                raise RuntimeError('Static idle scan physics changed')
            ticks.append(dict(start=start,stop=time.monotonic()))
        next_tick=[time.monotonic()]
        def pump(future=None,deadline=None):
            while (not future.done()) if future is not None else time.monotonic()<deadline:
                now=time.monotonic()
                if now>=next_tick[0]:
                    tick();next_tick[0]=max(next_tick[0]+.05,time.monotonic())
                time.sleep(.001)
            if future is not None:return future.result()
        def predict(issued=None,timing_reference=False):
            torch.manual_seed(20261023)
            request={} if issued is None else {'caller_request_monotonic':issued}
            if timing_reference:request['timing_reference']=True
            out=adapter.predict(snapshot,prompt,previous,delay=5,executed=5,request=request)
            if not np.isfinite(out).all() or request['rtc_guidance']['variable_length_fallback']:
                raise RuntimeError('Idle scan nonfinite actions or slow fallback')
            return request,out
        # Same high-frequency setting throughout; root trace checks actual clocks.
        with FixedScanLease(directory) as lease, suspend_cameras(env),ThreadPoolExecutor(max_workers=1) as executor:
            result['clock_settings_changed']=True
            monitor=PowerMonitor(.02,max_gap_seconds=.1);monitor.start()
            if adapter.config.get('m2_gpu_start_trace'):
                result['unprofiled_request_ms']=[]
                for _ in range(2):
                    request,_=pump(executor.submit(predict,None,True))
                    result['unprofiled_request_ms'].append(1000*(request['inference_ended_monotonic']-request['inference_started_monotonic']))
                result['profiling_intervention']=True
            for block,delay in result['plan']:
                lease.ensure(10)
                prepared,prep_actions=pump(executor.submit(predict))
                anchor=prepared['inference_ended_monotonic']
                # All bookkeeping counted in actual idle; no artificial nominal clock.
                pump(deadline=anchor+delay)
                status=runtime_status()
                issued=time.monotonic()
                recovery,out=pump(executor.submit(predict,issued))
                begin=recovery['inference_started_monotonic'];end=recovery['inference_ended_monotonic']
                if reference is None:reference=out.copy()
                diff=float(np.max(np.abs(out-reference)))
                if diff!=0:raise RuntimeError('Paired idle-scan actions changed')
                monitor.sample()
                samples=[r.power_mw/1000 for r in monitor.readings
                    if max(anchor,begin-.1)<=r.timestamp<begin and r.power_mw is not None]
                row=dict(block=block,requested_idle_seconds=delay,
                    preparation_start_monotonic=prepared['inference_started_monotonic'],
                    idle_start_monotonic=anchor,recovery_start_monotonic=begin,recovery_stop_monotonic=end,
                    actual_idle_seconds=begin-anchor,recovery_ms=(end-begin)*1000,
                    recovery=energy_window(monitor,begin,end),idle=energy_window(monitor,anchor,begin),
                    idle_tail_mean_power_w=statistics.mean(samples) if samples else None,
                    idle_tail_samples=len(samples),runtime_before_recovery=status,
                    action_max_abs_difference=diff,action_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                    guidance=recovery['rtc_guidance'])
                if adapter.config.get('m2_gpu_start_trace'):
                    row['gpu_start_timing']=recovery['gpu_start_timing']
                    row['request_issued_monotonic']=issued
                    row['request_to_actions_ready_ms']=(end-issued)*1000
                row['power_evidence_valid']=validate_recovery_power(row['recovery'],
                    startup_timing=bool(adapter.config.get('m2_gpu_start_trace')))
                if not row['power_evidence_valid']:
                    row['power_unavailable_reason']='Profiler-assisted run has incomplete sampling coverage; no valid recovery energy'
                    row['idle_tail_mean_power_w']=None
                row['frequency_check']=frequency_check(lease.events,'baseline',prepared['inference_started_monotonic'],end)
                if not row['frequency_check']['matches']:raise RuntimeError('Idle scan fixed frequency mismatch')
                result['rows'].append(row)
                _write_json_atomically(path,result)
                print('IDLE_SCAN block='+str(block)+' requested='+str(delay)+' actual_ms='+str((begin-anchor)*1000),flush=True)
            monitor.stop()
            result['raw_power']=[asdict(r) for r in monitor.readings]
        result.update(status='completed',restored=True,summary=describe(result['rows']),
            clock_trace=lease.events,control_service_ticks=ticks,
            limitations=['No automatic short/long threshold, railgate or sleep-state claim.',
                'Saved identical model inputs; synthetic frozen control service, not task continuation.',
                '20ms polling is not validated20ms sensor response; short points may be unresolved.',
                'Three repeated blocks are exploratory, not independent long-run confidence intervals.'])
        result['power_unavailable_rows']=sum(not r['power_evidence_valid'] for r in result['rows'])
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if monitor is not None:
            monitor.stop();result['raw_power']=[asdict(r) for r in monitor.readings]
        if env is not None:
            env.close()
        _write_json_atomically(path,result)
    return result
