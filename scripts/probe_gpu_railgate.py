#!/usr/bin/env python3
"""Tiny resident-CUDA railgate probe. No VLA model, force_idle, or frequency writes."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run,_write_json_atomically
from perceptbudget.gpu_railgate import state,confirmed_off,joint_rail_configuration
from perceptbudget.power import PowerMonitor


class RailgateUnavailable(RuntimeError):
    pass


class Lease:
    def __init__(self,directory):
        self.restored_event=None
        self.events=queue.Queue();self.log=(directory/'railgate_helper.jsonl').open('w')
        self.proc=subprocess.Popen(['sudo','-n','/usr/bin/python3',str(ROOT/'scripts/control_gpu_railgate.py')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        def read():
            for line in self.proc.stdout:
                self.log.write(line);self.log.flush()
                try:
                    event=json.loads(line)
                    if event.get('status')=='RESTORED':self.restored_event=event
                    self.events.put(event)
                except ValueError:pass
            self.events.put({'status':'EOF'})
        self.reader=threading.Thread(target=read,daemon=True);self.reader.start()
    def wait(self,status):
        end=time.monotonic()+5
        while time.monotonic()<end:
            event=self.events.get(timeout=max(.001,end-time.monotonic()))
            if event['status']==status:return event
            if event['status']=='REJECTED':
                raise RailgateUnavailable(f"Driver rejected railgate_enable={event['enabled']}: errno={event['errno']}, {event['error']}")
            if event['status']=='EOF':raise RuntimeError('Railgate helper exited; inspect helper log')
        raise TimeoutError(status)
    def set(self,value):
        self.proc.stdin.write(value+'\n');self.proc.stdin.flush()
        event=self.wait('APPLIED')
        if event['enabled']!=value:raise RuntimeError('Railgate command acknowledgement mismatch')
    def close(self):
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write('STOP\n');self.proc.stdin.flush();self.proc.stdin.close()
                except BrokenPipeError:pass
            code=self.proc.wait(timeout=5)
            self.reader.join(timeout=5)
            if self.restored_event is None:
                raise RuntimeError('No railgate restoration acknowledgement; inspect helper log')
            if code:raise RuntimeError('Railgate helper failed after restoring; inspect helper log')
        finally:
            self.reader.join(timeout=5);self.log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(scopes=['no CUDA context','resident tiny CUDA graph'],
            idle_seconds=[.2,1.,2.],repeats=2,changes=['railgate_enable only'],model_loaded=False)));return
    if os.geteuid()==0:parser.error('Run as normal user; only the helper uses sudo')
    cfg=load_config(repo_root=ROOT)
    if joint_rail_configuration():
        result=dict(status='unavailable',reason='Device tree declares joint XPU rail; GA10B driver disables independent railgate',
            model_loaded=False,power_settings_changed=False,clock_settings_changed=False,
            restoration_required=False,actual=state())
        run=create_run(dict(cfg,scope='Read-only joint-rail preflight'),'M4','gpu-railgate-preflight',repo_root=ROOT)
        _write_json_atomically(run/'summary.json',result)
        print('GPU_RAILGATE_STATUS='+json.dumps(dict(result,run=str(run))),flush=True)
        return 2
    subprocess.run(['sudo','-v'],check=True)
    with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=create_run(dict(cfg,scope='Tiny CUDA railgate capability, not VLA inference'),'M4','gpu-railgate-probe',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        result=dict(status='running',model_loaded=False,initial=state(),requests=[],no_context=[],restored=False,
            timing_scope='CPU submission through first CUDA graph completion; not first-kernel startup latency',
            energy_scope='Module input, idle+tiny replay+2s tail; not full-model energy or pure switching energy')
        samples=[];stop=threading.Event()
        def poll():
            while not stop.is_set():samples.append(state());stop.wait(.02)
        reader=threading.Thread(target=poll,daemon=True);monitor=PowerMonitor(.02,max_gap_seconds=.1)
        lease=Lease(run)
        try:
            lease.wait('READY');reader.start();monitor.start()
            # Check driver behaviour before importing torch or creating a CUDA context.
            for enabled in ('0','1'):
                lease.set(enabled);a=time.monotonic();time.sleep(2.);b=time.monotonic()
                seen=[x for x in samples if a<=x['monotonic']<=b]
                result['no_context'].append(dict(enabled=enabled,confirmed_off=any(map(confirmed_off,seen)),last=seen[-1] if seen else state()))
            lease.set('0')
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            stream=torch.cuda.Stream();x=torch.ones((128,128),device='cuda');y=torch.empty_like(x)
            with torch.cuda.stream(stream):
                for _ in range(3):torch.mm(x,x,out=y)
            stream.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):torch.mm(x,x,out=y)
            torch.cuda.synchronize()
            for idle in (.2,1.,2.):
                for repeat in range(2):
                    for enabled in (('0','1') if repeat==0 else ('1','0')):
                        lease.set(enabled);graph.replay();torch.cuda.synchronize()
                        a=time.monotonic();time.sleep(idle);before=state()
                        launch=time.monotonic();graph.replay();torch.cuda.synchronize();ready=time.monotonic()
                        time.sleep(2.);end=time.monotonic();monitor.sample()
                        idle_states=[s for s in samples if a<=s['monotonic']<=launch]
                        energy=asdict(monitor.integrate_report(a,end))
                        correct=bool(torch.equal(y.cpu(),torch.full((128,128),128.)))
                        row=dict(enabled=enabled,idle_seconds=idle,repeat=repeat+1,
                            start_monotonic=a,launch_monotonic=launch,ready_monotonic=ready,stop_monotonic=end,
                            first_replay_completion_ms=(ready-launch)*1000,
                            confirmed_off_during_idle=any(map(confirmed_off,idle_states)),
                            off_immediately_before_replay=confirmed_off(before),before_replay=before,
                            first_off_after_work_seconds=next((s['monotonic']-a for s in idle_states if confirmed_off(s)),None),
                            cycle_power=energy,output_matches=correct)
                        result['requests'].append(row);_write_json_atomically(run/'summary.json',result)
                        print('RAILGATE_TRIAL='+json.dumps({k:row[k] for k in ['enabled','idle_seconds','first_replay_completion_ms','confirmed_off_during_idle','output_matches']}),flush=True)
                        if not correct:raise RuntimeError('CUDA output changed after idle')
            result.update(status='completed',
                railgate_observed_without_cuda_context=any(r['confirmed_off'] for r in result['no_context']),
                railgate_observed_with_resident_cuda_graph=any(r['confirmed_off_during_idle'] for r in result['requests']),
                all_outputs_match=all(r['output_matches'] for r in result['requests']))
        except RailgateUnavailable as exc:
            result.update(status='unavailable',error=str(exc),
                reason='Driver rejected enabling automatic railgate before CUDA test; no force-idle fallback')
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            try:
                lease.close();result['restored']=True
            finally:
                stop.set()
                if reader.is_alive():reader.join(timeout=2)
                monitor.stop()
                result['final']=state()
                _write_json_atomically(run/'summary.json',result)
                _write_json_atomically(run/'gpu_states.json',samples)
                _write_json_atomically(run/'power_readings.json',[asdict(x) for x in monitor.readings])
        print('GPU_RAILGATE_STATUS='+json.dumps(dict(status=result['status'],restored=result['restored'],run=str(run))),flush=True)
        return 0 if result['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
