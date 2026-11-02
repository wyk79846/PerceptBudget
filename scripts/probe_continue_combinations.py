#!/usr/bin/env python3
"""Small306MHz Continue screen: lower EMC, then CPU, one resident model."""
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
CANDIDATES={
    'idle306':(306000000,2133000000,1497600),
    'emc1600':(306000000,1600000000,1497600),
    'cpu1190low':(306000000,2133000000,1190400),
    'cpu1036low':(306000000,2133000000,1036800),
    'cpu806low':(306000000,2133000000,806400),
}


def command(*args):
    return ['sudo','-n','/usr/bin/python3',str(Path(__file__).resolve()),'--helper',*args]


def helper():
    if os.geteuid()!=0:
        raise RuntimeError('Root helper requires sudo')
    from perceptbudget import device_clocks as clocks
    clocks.PROFILES.update(CANDIDATES)
    from scripts import control_device_clocks as control
    if '--timing-profile' in sys.argv:
        i=sys.argv.index('--timing-profile')
        target=sys.argv[i+1]
        if target not in CANDIDATES:
            raise ValueError('Unknown combination')
        def cycles(instance,unused,emit):
            instance.validate(target)
            for index in range(5):
                for direction,profile,current in [('down',target,'baseline'),('up','baseline',target)]:
                    row=clocks.observed_switch(instance,profile,current)
                    emit(dict(status='TRANSITION',index=index,direction=direction,profile=profile,**row))
                    time.sleep(.2)
        control.cycle_trace=cycles
        sys.argv=[sys.argv[0],'cycles','--profile','gpu_emc']
    else:
        sys.argv.remove('--helper')
    control.main()


def screen(adapter,snapshot,prompt,previous,torch,directory,metadata,*,original_screen):
    from unittest.mock import patch
    from perceptbudget import device_clocks as clocks,device_continue as cont
    from perceptbudget.experiment import _write_json_atomically as write
    accepted=json.loads((directory/'preflight.json').read_text())['accepted']
    class ProbeLease(cont.ContinueLease):
        def command(self):
            return command('lease','--profile',self.profile,'--seconds',str(self.duration),
                           '--trace','--trace-interval','.02')
    with patch.dict(clocks.PROFILES,CANDIDATES),patch.object(cont,'MATRIX_PROFILES',tuple(accepted)), \
         patch.object(cont,'REPEATS',2),patch.object(cont,'ContinueLease',ProbeLease):
        record=original_screen(adapter,snapshot,prompt,previous,torch,directory,metadata,
                           history_windows=True,history_points=(1.,),report_cadence_misses=True,paired_fresh_inputs=True)
    for row in record['summary']:
        row['cadence_usable']=row['deadline_overruns']==0
    record.update(candidate_profiles={p:CANDIDATES[p] for p in accepted},
                  scope='Spatial0 live Continue: same20 original actions, measure last5 after15 at70ms; '
                        'no inference during execution, one resident model, not whole-task quality',
                  production_defaults_changed=False)
    write(directory/'continue_control.json',record)
    return record


def main():
    if '--helper' in sys.argv:
        return helper()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(candidates=CANDIDATES,model_loads=1,windows_per_accepted_profile=2,
            maximum_windows=10,history_actions=15,measured_actions=5,period_seconds=.07,
            skip_rejected_frequencies_before_loading=True,full_combo_switch_pairs=5,
            production_defaults_changed=False)))
        return 0
    if os.geteuid()==0:
        parser.error('Run as normal user; only bounded helper uses sudo')
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically as write
    from perceptbudget.baseline import validate
    config=load_config(repo_root=ROOT);validate(config)
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent!=(Path(config['output_root'])/'M4').resolve():
            parser.error('Expected M4 directory')
        from unittest.mock import patch
        from perceptbudget.device_profile import fixed_view_probe
        source=Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
        from functools import partial
        from perceptbudget.device_continue import screen as original_screen
        with patch('perceptbudget.device_continue.screen',partial(screen,original_screen=original_screen)):
            fixed_view_probe(config,args.run_dir,source,'full',clock_screen=True,continue_control=True)
        return 0
    subprocess.run(['sudo','-v'],check=True)
    run=create_run(config,'M4','continue-lower-combos',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    summary=dict(status='running',model_loads=0)
    preflight=dict(accepted=[],rejected=[],timings={})
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for profile in CANDIDATES:
                path=run/(profile+'_preflight.log')
                with path.open('w') as log:
                    code=subprocess.run(command('lease','--profile',profile,'--seconds','1','--trace'),
                                        stdout=log,stderr=subprocess.STDOUT).returncode
                lines=path.read_text().splitlines()
                if 'RESTORED' not in lines:
                    raise RuntimeError('Restoration not confirmed; inspect '+str(path))
                if code:
                    preflight['rejected'].append(dict(profile=profile,log=str(path)))
                else:
                    preflight['accepted'].append(profile)
                write(run/'preflight.json',preflight)
            if 'idle306' not in preflight['accepted']:
                raise RuntimeError('Reference306MHz combination unavailable')
            for profile in preflight['accepted']:
                path=run/(profile+'_switch.log')
                with path.open('w') as log:
                    code=subprocess.run(command('--timing-profile',profile),stdout=log,
                                        stderr=subprocess.STDOUT).returncode
                lines=path.read_text().splitlines()
                if code or 'RESTORED' not in lines:
                    raise RuntimeError('Combination timing failed; inspect '+str(path))
                events=[json.loads(line) for line in lines if line.startswith('{')]
                stats={}
                for direction in ('down','up'):
                    rows=[e for e in events if e.get('status')=='TRANSITION' and e['direction']==direction]
                    if len(rows)!=5:raise RuntimeError('Missing switch samples')
                    times=[(e['observed_match_monotonic']-e['command_monotonic'])*1000 for e in rows]
                    stats[direction]=dict(samples=5,median_ms=statistics.median(times),max_ms=max(times))
                preflight['timings'][profile]=stats
                write(run/'preflight.json',preflight)
            from scripts.profile_clock_screen import monitored_worker
            summary['model_loads']=None
            with (run/'worker.log').open('w') as log:
                code=monitored_worker([config['python_executable'],'-u',str(Path(__file__).resolve()),
                                       '--worker','--run-dir',str(run)],log)
            if code:raise RuntimeError('Continue worker failed; inspect worker.log and restore logs')
            record=json.loads((run/'continue_control.json').read_text())
            if record['status']!='completed' or len(record['windows'])!=2*len(preflight['accepted']):
                raise RuntimeError('Incomplete Continue screening')
            summary.update(status='completed',model_loads=1,conditions=record['summary'],
                           transitions=preflight['timings'],rejected=preflight['rejected'],
                           paired_final_states=record['all_final_states_equal'])
    except BaseException as exc:
        summary.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(run/'summary.json',summary)
        print('CONTINUE_COMBINATIONS_STATUS='+json.dumps(summary),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
