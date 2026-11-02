#!/usr/bin/env python3
"""Resident-model lower-idle screening; production clock profiles are unchanged."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILES = {
    'gpu_emc': (816000000, 2133000000, 1497600),
    'idle408': (408000000, 1600000000, 1497600),
    'idle306': (306000000, 665600000, 1497600),
}


def helper_command(*args):
    return ['sudo', '-n', '/usr/bin/python3', str(Path(__file__).resolve()),
            '--clock-helper', *args]


def clock_helper():
    # Extend only this diagnostic process. Existing production profiles/defaults
    # remain unchanged; validation, root snapshot, lock and restoration are reused.
    if os.geteuid() != 0:
        raise RuntimeError('Clock helper requires sudo')
    from perceptbudget import device_clocks
    device_clocks.PROFILES.update(PROFILES)
    device_clocks.MATRIX_PROFILES = ('baseline', *PROFILES)
    from scripts.control_device_clocks import main
    sys.argv.remove('--clock-helper')
    main()


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata):
    import numpy as np
    from perceptbudget.device_clock_screen import Lease, energy_window, frequency_check
    from perceptbudget.device_clock_recovery import RecoveryLease
    from perceptbudget.experiment import _write_json_atomically as write
    from perceptbudget.power import PowerMonitor
    # frequency_check uses the shared dictionary for readback expectations only.
    from perceptbudget.device_clocks import PROFILES as known_profiles

    class ProbeLease(RecoveryLease):
        def __init__(self, profile, label):
            Lease.__init__(self, profile, directory)
            self.path = directory/(label+'_clocks.jsonl')
            self.duration = 60

        def command(self):
            return helper_command('recovery', '--profile', self.profile,
                                  '--idle-seconds', '1.0')

    saved = dict(known_profiles)
    result = dict(metadata, status='running', rows=[], model_loads=1,
        scope='Resident static no-inference hold and prepared-input Full recovery; '
              'not live Continue, task energy, or success-rate validation',
        profiles=PROFILES, idle_seconds=1., common_cycle_seconds=3.5,
        power_boundary='VDD_IN module total', production_defaults_changed=False)
    reference = {}
    try:
        known_profiles.update(PROFILES)
        # Forward/reverse blocks reduce systematic order bias: 30 requests,
        # five per helper transaction, ten samples for each profile.
        for block, order in enumerate((tuple(PROFILES), tuple(reversed(PROFILES)))):
            for profile in order:
                label = f'b{block}_{profile}'
                print('IDLE_PROFILE_START='+label, flush=True)
                with PowerMonitor(.02, max_gap_seconds=.1) as monitor:
                    with ProbeLease(profile, label) as lease:
                        for index in range(5):
                            seed = 20261102+index
                            torch.manual_seed(seed)
                            adapter.predict(snapshot, prompt, previous, delay=5,
                                            executed=5, request={})
                            torch.manual_seed(seed)
                            lease.send('RUN')
                            event = lease.receive('SERVE', index)
                            origin = event['down']['command_monotonic']
                            start = time.monotonic()
                            request = {}
                            actions = adapter.predict(snapshot, prompt, previous,
                                delay=5, executed=5, request=request)
                            end = time.monotonic()
                            if not np.isfinite(actions).all():
                                raise RuntimeError('Nonfinite actions')
                            if request['rtc_guidance'].get('variable_length_fallback'):
                                raise RuntimeError('Unexpected RTC fallback')
                            reference.setdefault(index, actions.copy())
                            difference = float(np.max(np.abs(actions-reference[index])))
                            if difference != 0:
                                raise RuntimeError('Matched action outputs changed')
                            stop = origin+3.5
                            if end > stop-1.:
                                raise RuntimeError('Full recovery exceeded common allowance')
                            time.sleep(max(0., stop-time.monotonic()))
                            monitor.sample()
                            hold_start = event['down']['observed_match_monotonic']+.2
                            hold_end = event['up']['command_monotonic']
                            windows = dict(cycle=energy_window(monitor, origin, stop),
                                hold=energy_window(monitor, hold_start, hold_end))
                            if any(w['energy_j'] is None for w in windows.values()):
                                raise RuntimeError('Power coverage incomplete')
                            hold_check = frequency_check(lease.events, profile, hold_start, hold_end)
                            full_check = frequency_check(lease.events, 'baseline', start, end)
                            if not all(c['matches'] and c['sample_count'] for c in (hold_check, full_check)):
                                raise RuntimeError('Clock readback mismatch or no samples')
                            result['rows'].append(dict(profile=profile, block=block,
                                index=index, seed=seed, origin=origin, end=stop,
                                hold_start=hold_start, hold_end=hold_end, windows=windows,
                                hold_power_w=windows['hold']['energy_j']/(hold_end-hold_start),
                                recovery_to_actions_ms=(end-event['up']['command_monotonic'])*1000,
                                predict_ms=(end-start)*1000, max_abs_action_difference=difference,
                                hold_frequency_check=hold_check, full_frequency_check=full_check,
                                transition=event))
                            write(directory/'lower_idle.json', result)
                            lease.send('DONE')
                            lease.receive('REQUEST_ACK', index)
                    if not any(e.get('message') == 'RESTORED' for e in lease.events):
                        raise RuntimeError('Clock restoration missing')
                    write(directory/(label+'_power.json'), [asdict(r) for r in monitor.readings])
        conditions = {}
        for profile in PROFILES:
            rows = [r for r in result['rows'] if r['profile'] == profile]
            conditions[profile] = dict(samples=len(rows),
                hold_power_w=statistics.mean(r['hold_power_w'] for r in rows),
                mean_cycle_j=statistics.mean(r['windows']['cycle']['energy_j'] for r in rows),
                median_recovery_ms=statistics.median(r['recovery_to_actions_ms'] for r in rows))
        baseline = conditions['gpu_emc']
        for profile, condition in conditions.items():
            pairs = [next(b for b in result['rows'] if b['profile']=='gpu_emc'
                     and b['block']==r['block'] and b['index']==r['index'])['windows']['cycle']['energy_j']
                     - r['windows']['cycle']['energy_j']
                     for r in result['rows'] if r['profile']==profile]
            condition.update(hold_reduction_pct=100*(1-condition['hold_power_w']/baseline['hold_power_w']),
                mean_cycle_saved_j=statistics.mean(pairs), positive_pairs=sum(p>0 for p in pairs),
                cycle_saved_range_j=[min(pairs),max(pairs)])
        result.update(status='completed', restored=True, conditions=conditions)
        return result
    finally:
        known_profiles.clear()
        known_profiles.update(saved)
        write(directory/'lower_idle.json', result)


def main():
    if '--clock-helper' in sys.argv:
        return clock_helper()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(profiles=PROFILES, model_loads=1, measured_requests=30,
            preparation_requests=30, idle_seconds=1, common_cycle_seconds=3.5,
            live_continue=False, restore=True, changes_production_defaults=False)))
        return 0
    if os.geteuid()==0:
        parser.error('Run as normal user; only clock helper uses sudo')
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically as write
    from perceptbudget.baseline import validate
    config=load_config(repo_root=ROOT)
    validate(config)
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M4').resolve():
            parser.error('Expected M4 run directory')
        from unittest.mock import patch
        from perceptbudget.device_profile import fixed_view_probe
        source=Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
        with patch('perceptbudget.device_idle_payback.screen', screen):
            fixed_view_probe(config,args.run_dir,source,'full',clock_screen=True,idle_payback=True)
        return 0
    subprocess.run(['sudo','-v'],check=True)
    run=create_run(config,'M4','lower-idle-screen',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    summary=dict(status='running',model_loads=0)
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for profile in PROFILES:
                path=run/(profile+'_preflight.log')
                with path.open('w') as log:
                    code=subprocess.run(helper_command('lease','--profile',profile,'--seconds','1'),
                                        stdout=log,stderr=subprocess.STDOUT).returncode
                if code or 'RESTORED' not in path.read_text().splitlines():
                    raise RuntimeError('Clock preflight failed before model loading: '+str(path))
            from scripts.profile_clock_screen import monitored_worker
            summary['model_loads']=None
            with (run/'worker.log').open('w') as log:
                code=monitored_worker([config['python_executable'],'-u',str(Path(__file__).resolve()),
                                       '--worker','--run-dir',str(run)],log)
            if code:
                raise RuntimeError('Worker failed; inspect worker.log and restoration logs')
            record=json.loads((run/'lower_idle.json').read_text())
            if record['status']!='completed' or len(record['rows'])!=30 or not record['restored']:
                raise RuntimeError('Incomplete lower idle screen')
            summary.update(status='completed',model_loads=1,conditions=record['conditions'],restored=True)
    except BaseException as exc:
        summary.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(run/'summary.json',summary)
        print('LOWER_IDLE_STATUS='+json.dumps(summary),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
