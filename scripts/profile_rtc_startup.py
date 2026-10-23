#!/usr/bin/env python3
"""Initialize CUPTI with sudo, irrevocably drop privileges BEFORE VLA loading.

No driver/permission changes. If collection does not survive privilege drop,
abort before loading weights. Only the tiny synthetic initialization is root.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import signal
import sys
import tempfile
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def identity():
    values={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key=line.split(':',1)[0]
        if key in ('Uid','Gid','CapEff','CapPrm','CapAmb'):
            values[key]=line.split(':',1)[1].strip()
    return values


def drop_to_invoker():
    uid=int(os.environ.get('SUDO_UID','0'))
    if os.geteuid()!=0 or uid==0:
        raise RuntimeError('Use sudo from the normal project user, not a root login')
    account=pwd.getpwuid(uid)
    os.initgroups(account.pw_name,account.pw_gid)
    os.setresgid(account.pw_gid,account.pw_gid,account.pw_gid)
    os.setresuid(uid,uid,uid)
    if os.getresuid()!=(uid,uid,uid):raise RuntimeError('UID drop incomplete')
    status=identity()
    if any(int(status[k],16)!=0 for k in ('CapEff','CapPrm','CapAmb')):
        raise RuntimeError('Profiling worker retained capabilities; refusing model load')
    # Restore the actual invoking user's identity, rather than sudo's /root caches.
    os.environ.update(HOME=account.pw_dir,USER=account.pw_name,LOGNAME=account.pw_name)
    return status


def initialize_and_drop():
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from scripts.probe_gpu_start_timing import first_kernel_offset
    if os.geteuid()!=0 or not int(os.environ.get('SUDO_UID','0')):
        raise RuntimeError('Run using sudo from the normal project account')
    a=torch.ones((128,128),device='cuda');stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):b=a@a
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):b=a@a
    torch.cuda.synchronize()
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities):
        graph.replay();torch.cuda.synchronize()
    status=drop_to_invoker()
    # Check collection in the same kind of ordinary-user worker thread used later.
    def check():
        with tempfile.TemporaryDirectory(prefix='perceptbudget-unpriv-cupti-') as tmp:
            path=Path(tmp)/'trace.json';marker='PB_AFTER_PRIVILEGE_DROP'
            with torch.profiler.profile(activities=activities) as prof:
                with torch.profiler.record_function(marker):
                    graph.replay();torch.cuda.synchronize()
            prof.export_chrome_trace(str(path))
            return first_kernel_offset(json.loads(path.read_text()),marker)
    with ThreadPoolExecutor(max_workers=1) as executor:
        proof=[executor.submit(check).result() for _ in range(2)]
    return torch,dict(measurement_identity=status,model_uid=os.geteuid(),
        root_model_allowed=False,persistent_permissions_changed=False,
        unprivileged_kernel_capture=proof)


def run_scan(config,run,torch):
    from unittest.mock import patch
    from perceptbudget.device_profile import fixed_view_probe
    from perceptbudget.device_idle_scan import screen
    from perceptbudget.gpu_start_trace import PairCollector
    def instrument(adapter,*args,**kwargs):
        collector=PairCollector(adapter,run,torch)
        adapter.predict=collector.predict
        try:
            result=screen(adapter,*args,**kwargs)
            if collector.index!=21:raise RuntimeError('Incomplete21GPU timing traces')
            return result
        finally:
            adapter.predict=collector.original;collector.close()
    with patch('perceptbudget.device_idle_scan.screen',instrument):
        return fixed_view_probe(config,run,
            Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d',
            'full',clock_screen=True,idle_state_scan=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--permission-only',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(model_loads=1,root_stage='tiny CUPTI initialization only',
            model_uid='invoking user, real/effective/saved UID all dropped, capabilities0',
            kernel_permission_gate_before_model=True,points_seconds=[0,.05,.1,.2,.5,1,2],
            repeats=3,baseline_no_profiler_requests=2,profiled_recovery_requests=21,
            profiler_armed_before_idle=True,driver_settings_changed=False)))
        return 0
    run=None;stop=threading.Event()
    try:
        torch,proof=initialize_and_drop()
        print('PROFILE_PRIVILEGE_GATE='+json.dumps(proof),flush=True)
        if args.permission_only:return 0
        from perceptbudget.experiment import load_config,create_run,_write_json_atomically
        from perceptbudget.baseline import validate
        from scripts.profile_clock_screen import clock_preflight
        from scripts.guarded_run import available_ram_bytes,RAM_FLOOR_BYTES
        config=load_config(repo_root=ROOT);validate(config)
        config=dict(config,m2_gpu_start_trace=True,m2_startup_root_model_allowed=False)
        if available_ram_bytes()<RAM_FLOOR_BYTES:raise RuntimeError('RAM below768MiB')
        run=create_run(config,'M2','rtc-startup-timing',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        summary=dict(status='running',privilege_proof=proof,
            scope='Profiler-assisted request-to-first-kernel and action-return diagnostics; not normal production latency')
        _write_json_atomically(run/'summary.json',summary)
        def memory_guard():
            while not stop.wait(.1):
                if available_ram_bytes()<RAM_FLOOR_BYTES:
                    _write_json_atomically(run/'memory_guard.json',dict(status='invalid',reason='RAM below768MiB'))
                    os.kill(os.getpid(),signal.SIGINT)
                    if not stop.wait(5):os._exit(125)
                    return
        guard=threading.Thread(target=memory_guard,daemon=True);guard.start()
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            clock_preflight(run,('baseline',),summary)
            result=run_scan(config,run,torch)
        summary.update(status='completed',model_loads=result['model_loads'],restored=result['restored'],
            conditions=result['summary'],unprofiled_request_ms=result['unprofiled_request_ms'],
            startup_rows=[dict(requested_idle_seconds=r['requested_idle_seconds'],block=r['block'],
                **r['gpu_start_timing']) for r in result['rows']])
        _write_json_atomically(run/'summary.json',summary)
        return 0
    except BaseException as exc:
        print('STARTUP_TIMING_ERROR='+json.dumps(dict(error=f'{type(exc).__name__}: {exc}',
            uid=os.geteuid(),run=str(run) if run else None,
            model_loaded=False if run is None else 'inspect run metadata')),flush=True)
        if run is not None:
            from perceptbudget.experiment import _write_json_atomically
            s=json.loads((run/'summary.json').read_text());s.update(status='invalid',error=str(exc))
            _write_json_atomically(run/'summary.json',s)
        return 2
    finally:
        stop.set()
        if run is not None:
            from perceptbudget.experiment import _write_json_atomically
            r=json.loads((run/'run.json').read_text());r['status']=json.loads((run/'summary.json').read_text())['status']
            _write_json_atomically(run/'run.json',r)


if __name__=='__main__':raise SystemExit(main())
