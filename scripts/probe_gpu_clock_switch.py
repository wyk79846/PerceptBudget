#!/usr/bin/env python3
"""GPU-only1173/306MHz switching, fixed EMC2133MHz and CPU1497.6MHz."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
HIGH=1173000000
LOW=306000000


def switch(control, target):
    """Command to first matching sysfs read; not an actual GPU kernel timestamp."""
    before=control.snapshot()
    advertised=list(map(int,(control.gpu/'available_frequencies').read_text().split()))
    if target not in (HIGH,LOW) or target not in advertised:
        raise ValueError('Unapproved GPU frequency')
    start=time.monotonic()
    control.bounds(control.gpu/'min_freq',control.gpu/'max_freq',target,target)
    written=time.monotonic()
    while True:
        actual=control.read(control.gpu/'cur_freq')
        observed=time.monotonic()
        if actual==target:
            break
        if observed-written>.5:
            raise RuntimeError('GPU current frequency did not match within500ms')
        time.sleep(.001)
    after=control.snapshot()
    unchanged={k:v for k,v in before.items() if not k.startswith('class/devfreq/17000000.gpu/')}
    if any(after[k]!=v for k,v in unchanged.items()):
        raise RuntimeError('Non-GPU settings changed during timed transition')
    return dict(target_hz=target,actual_hz=actual,command_monotonic=start,
                settings_monotonic=written,observed_monotonic=observed,
                settings_ms=(written-start)*1000,command_to_readback_ms=(observed-start)*1000,
                non_gpu_settings_unchanged=True)


def helper():
    if os.geteuid()!=0:
        raise RuntimeError('Helper requires sudo')
    from perceptbudget import device_clocks
    # Reuse root journal, locking, fixed40W gate, signals and finally restoration.
    from scripts import control_device_clocks
    device_clocks.PROFILES['baseline']=(HIGH,2133000000,1497600)
    device_clocks.PROFILES['gpu_emc']=(LOW,2133000000,1497600)
    def cycles(control,profile,emit):
        for index in range(10):
            for direction,target in [('down',LOW),('up',HIGH)]:
                record=switch(control,target)
                emit(dict(status='TRANSITION',index=index,direction=direction,**record))
                time.sleep(.2)
    control_device_clocks.cycle_trace=cycles
    sys.argv=[sys.argv[0],'cycles','--profile','gpu_emc']
    control_device_clocks.main()


def summarize(events):
    rows=[e for e in events if e.get('status')=='TRANSITION']
    result={}
    for direction in ('down','up'):
        subset=[r for r in rows if r['direction']==direction]
        if len(subset)!=10 or {r['index'] for r in subset}!=set(range(10)):
            raise ValueError('Expected10 transitions in each direction')
        if any(r['actual_hz']!=r['target_hz'] or not r['non_gpu_settings_unchanged'] for r in subset):
            raise ValueError('Invalid transition readback')
        values=sorted(r['command_to_readback_ms'] for r in subset)
        result[direction]=dict(samples=10,median_ms=statistics.median(values),
            p95_ms=values[9],max_ms=max(values),min_ms=min(values),
            median_settings_ms=statistics.median(r['settings_ms'] for r in subset))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.helper:
        return helper()
    if args.dry_run:
        print(json.dumps(dict(high_hz=HIGH,low_hz=LOW,emc_hz=2133000000,cpu_khz=1497600,
            pairs=10,model_loads=0,timing_scope='GPU bounds command to current-frequency readback',
            production_defaults_changed=False)))
        return 0
    if os.geteuid()==0:
        parser.error('Run normally; only bounded helper uses sudo')
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically as write
    config=load_config(repo_root=ROOT)
    subprocess.run(['sudo','-v'],check=True)
    run=create_run(config,'M4','gpu-clock-switch',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    summary=dict(status='running',model_loads=0,
        scope='Static GPU-only sysfs command/readback timing; not resident-model wake or first-kernel timing',
        fixed_emc_hz=2133000000,fixed_cpu_khz=1497600)
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with (run/'clocks.log').open('w') as log:
                code=subprocess.run(['sudo','-n','/usr/bin/python3',str(Path(__file__).resolve()),'--helper'],
                                    stdout=log,stderr=subprocess.STDOUT).returncode
            lines=(run/'clocks.log').read_text().splitlines()
            restored='RESTORED' in lines
            summary['restored']=restored
            if code or not restored:
                raise RuntimeError('Clock helper failed; inspect '+str(run/'clocks.log'))
            events=[]
            for line in lines:
                try:events.append(json.loads(line))
                except ValueError:pass
            summary.update(status='completed',conditions=summarize(events))
    except BaseException as exc:
        summary.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(run/'summary.json',summary)
        print('GPU_SWITCH_STATUS='+json.dumps(summary),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
