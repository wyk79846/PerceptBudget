#!/usr/bin/env python3
"""Matched40/50ms idle windows before an already planned Full clock change."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
GAPS=(.04,.05)
REPEATS=3
ROUTES=(('baseline','gpu_emc'),('gpu_emc','baseline'))


def switch(clocks,profile):
    if clocks.profile==profile and clocks.pending is None:return None
    clocks.switch(profile)
    stop=time.monotonic()+2.
    while clocks.pending is not None:
        clocks.poll()
        if time.monotonic()>stop:raise TimeoutError('Short-idle clock acknowledgement')
        time.sleep(.001)
    return dict(clocks.last_change)


def screen(adapter,snapshot,prompt,previous,torch,directory,metadata):
    import numpy as np
    from perceptbudget.three_frequency import ThreeClocks
    from perceptbudget.device_clock_screen import energy_window,frequency_check
    from perceptbudget.power import PowerMonitor
    from perceptbudget.experiment import _write_json_atomically as write
    result=dict(metadata,status='running',rows=[],model_loads=1,repeats=REPEATS,
        gaps_seconds=GAPS,routes=ROUTES,after_gap_seconds=3.,
        scope='Prepared Full inputs and real bounded clock changes; static resident model, no env.step or task-success claim. '
              'Common window starts after identical home-frequency warm inference and ends gap+3s at the same target frequency.',
        power_boundary='INA3221 VDD_IN module input',production_defaults_changed=False)
    references={}
    try:
        for home,target in ROUTES:
            label=home+'-to-'+target
            route_dir=directory/label;route_dir.mkdir()
            clock=ThreeClocks(route_dir)
            entered=False;monitor=None
            try:
                clock.__enter__()
                entered=True
                with PowerMonitor(.02,max_gap_seconds=.1) as monitor:
                    time.sleep(.05)
                    for index in range(REPEATS):
                        # Reverse both orderings between repeats; each pair has
                        # identical input, noise, preceding load and endpoints.
                        gaps=GAPS if index%2==0 else GAPS[::-1]
                        policies=('hold','deep') if index%2==0 else ('deep','hold')
                        for gap in gaps:
                            for policy in policies:
                                switch(clock,home)
                                seed=20261103+index
                                torch.manual_seed(seed)
                                warm={}
                                adapter.predict(snapshot,prompt,previous,delay=5,executed=5,request=warm)
                                if warm['rtc_guidance'].get('variable_length_fallback'):
                                    raise RuntimeError('Unexpected preparation fallback')
                                torch.manual_seed(seed)
                                monitor.sample()
                                origin=time.monotonic()
                                down=switch(clock,'deep_idle') if policy=='deep' else None
                                if time.monotonic()>origin+gap:
                                    raise RuntimeError('Downclock consumed entire requested gap')
                                time.sleep(max(0.,origin+gap-time.monotonic()))
                                up=switch(clock,target)
                                if up is None:raise RuntimeError('Test route needs a real target clock change')
                                request={};begin=time.monotonic()
                                actions=adapter.predict(snapshot,prompt,previous,delay=5,executed=5,request=request)
                                ready=time.monotonic()
                                if not np.isfinite(actions).all() or request['rtc_guidance'].get('variable_length_fallback'):
                                    raise RuntimeError('Nonfinite actions or fallback')
                                key=(home,target,gap,index)
                                references.setdefault(key,actions.copy())
                                diff=float(np.max(np.abs(actions-references[key])))
                                if diff:raise RuntimeError('Matched Full actions changed')
                                stop=origin+gap+3.
                                if ready>origin+gap+1.2:raise RuntimeError('Full exceeded1.2s measurement allowance')
                                time.sleep(max(0.,stop-time.monotonic()))
                                monitor.sample();time.sleep(.025);monitor.sample();clock.poll()
                                full_check=frequency_check(clock.lease.events,target,begin,ready)
                                if not full_check['matches']:raise RuntimeError('Full clock mismatch')
                                windows={name:energy_window(monitor,origin,end) for name,end in
                                         (('common',stop),('early',stop-1.))}
                                if any(w['energy_j'] is None for w in windows.values()):
                                    raise RuntimeError('Incomplete power coverage')
                                result['rows'].append(dict(home=home,target=target,gap_ms=gap*1000,
                                    index=index,policy=policy,origin=origin,down=down,up=up,
                                    low_dwell_ms=None if down is None else
                                        1000*(up['command_monotonic']-down['observed_match_monotonic']),
                                    recovery_to_actions_ms=1000*(ready-up['command_monotonic']),
                                    full_ms=1000*(ready-begin),actions_ready_from_origin_ms=1000*(ready-origin),
                                    window_dispatch_lateness_ms=1000*(up['command_monotonic']-(origin+gap)),
                                    max_abs_action_difference=diff,full_frequency_check=full_check,windows=windows,
                                    short_dwell_note='APPLIED receipt confirms readback;0.2s sampling does not continuously cover40/50ms'))
                                write(directory/'short_idle.json',result)
                                print('SHORT_IDLE '+json.dumps(dict(route=label,gap_ms=gap*1000,index=index,policy=policy)),flush=True)
            finally:
                if monitor is not None:write(route_dir/'power.json',[asdict(r) for r in monitor.readings])
                if entered:clock.close()
            if not any(e.get('message')=='RESTORED' for e in clock.lease.events):
                raise RuntimeError('Missing restoration receipt')
        pairs=[]
        for home,target in ROUTES:
            for gap in GAPS:
                values=[];early=[];extras=[]
                for index in range(REPEATS):
                    a,b=(next(r for r in result['rows'] if r['home']==home and r['target']==target
                              and r['gap_ms']==gap*1000 and r['index']==index and r['policy']==p)
                         for p in ('hold','deep'))
                    values.append(a['windows']['common']['energy_j']-b['windows']['common']['energy_j'])
                    early.append(a['windows']['early']['energy_j']-b['windows']['early']['energy_j'])
                    extras.append(b['actions_ready_from_origin_ms']-a['actions_ready_from_origin_ms'])
                pairs.append(dict(home=home,target=target,gap_ms=gap*1000,samples=REPEATS,
                    saved_j=values,mean_saved_j=statistics.mean(values),positive_pairs=sum(x>0 for x in values),
                    early_mean_saved_j=statistics.mean(early),actions_ready_extra_ms=extras))
        result.update(status='completed',restored=True,pairs=pairs)
        return result
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(directory/'short_idle.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(gaps_ms=[40,50],routes=ROUTES,repeats=REPEATS,model_loads=1,
            measured_requests=24,preparation_requests=24,common_tail_seconds=3.,live_control=False)))
        return 0
    if os.geteuid()==0:parser.error('Run as ordinary user; existing finite helper alone changes clocks')
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically as write
    from perceptbudget.baseline import validate
    config=load_config(repo_root=ROOT);validate(config)
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent!=(Path(config['output_root'])/'M4').resolve():
            parser.error('Expected M4 run directory')
        from perceptbudget.device_profile import fixed_view_probe
        source=Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
        with patch('perceptbudget.device_idle_payback.screen',screen):
            fixed_view_probe(config,args.run_dir,source,'full',clock_screen=True,idle_payback=True)
        return 0
    from perceptbudget.clock_access import authorize
    from perceptbudget.three_frequency import preflight
    from scripts.profile_clock_screen import monitored_worker
    authorize();run=create_run(config,'M4','short-idle40-50',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    status=dict(status='running',model_loads=0)
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            preflight(run)
            status['model_loads']=None
            with (run/'worker.log').open('w') as log:
                code=monitored_worker([config['python_executable'],'-u',str(Path(__file__).resolve()),
                    '--worker','--run-dir',str(run)],log)
            if code:raise RuntimeError('Worker failed; inspect worker.log and finite-helper restoration')
            status=json.loads((run/'short_idle.json').read_text())
            if status['status']!='completed' or len(status['rows'])!=24 or not status['restored']:
                raise RuntimeError('Incomplete short idle probe')
    except BaseException as exc:
        status.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(run/'summary.json',status)
        meta=json.loads((run/'run.json').read_text());meta['status']=status['status'];write(run/'run.json',meta)
        print('SHORT_IDLE_STATUS='+status['status'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
