"""Matched resident-model idle cycles; exploratory mean energy, no railgate claim."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import statistics
import time

from perceptbudget.device_clock_recovery import RecoveryLease
from perceptbudget.device_clock_screen import ROOT, energy_window, frequency_check
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor
from perceptbudget.rtc_memory import memory_record

DELAYS = (.05, .2, 1., 2.)
BOUNDARY_DELAYS = (.1, .15)
REPEATS = 5
# A common 1s recovery allowance plus 2s sampling tail. This is a measurement
# window, NOT a production deadline or a claim about sensor settling time.
AFTER_IDLE_SECONDS = 3.


def plan(delays=DELAYS):
    if tuple(delays) not in (DELAYS, BOUNDARY_DELAYS):
        raise ValueError('Idle plan outside measured or boundary whitelist')
    return [(delay, profile) for index, delay in enumerate(delays)
            for profile in (('baseline', 'gpu_emc') if index % 2 == 0
                            else ('gpu_emc', 'baseline'))]


class PairedIdleLease(RecoveryLease):
    def __init__(self, profile, delay, directory, label):
        if profile not in ('baseline', 'gpu_emc') or delay not in DELAYS+BOUNDARY_DELAYS:
            raise ValueError('Paired idle scenario outside whitelist')
        # Reuse the bounded protocol/EOF restoration, not its older dwell origin.
        super().__init__(profile, .2, directory, label)
        self.idle = delay

    def command(self):
        return ['sudo', '-n', '/usr/bin/python3', str(ROOT/'scripts/control_device_clocks.py'),
                'paired-idle', '--profile', self.profile, '--idle-seconds', str(self.idle)]


def summarize(rows, *, delays=DELAYS, repeats=REPEATS):
    result = []
    for delay in delays:
        paired = []
        for index in range(repeats):
            high = next(r for r in rows if r['idle_seconds'] == delay and r['profile'] == 'baseline' and r['index'] == index)
            low = next(r for r in rows if r['idle_seconds'] == delay and r['profile'] == 'gpu_emc' and r['index'] == index)
            paired.append(dict(index=index,
                energy_saved_j=high['matched_cycle']['energy_j']-low['matched_cycle']['energy_j'],
                early_window_energy_saved_j=high['early_cycle']['energy_j']-low['early_cycle']['energy_j'],
                actions_ready_extra_ms=low['actions_ready_from_origin_ms']-high['actions_ready_from_origin_ms']))
        result.append(dict(idle_seconds=delay, pairs=paired,
            mean_saved_j=statistics.mean(p['energy_saved_j'] for p in paired),
            early_mean_saved_j=statistics.mean(p['early_window_energy_saved_j'] for p in paired),
            mean_tail_sensitivity_j=statistics.mean(p['energy_saved_j']-p['early_window_energy_saved_j'] for p in paired),
            saved_j_min=min(p['energy_saved_j'] for p in paired),
            saved_j_max=max(p['energy_saved_j'] for p in paired),
            mean_ready_extra_ms=statistics.mean(p['actions_ready_extra_ms'] for p in paired),
            positive_pairs=sum(p['energy_saved_j'] > 0 for p in paired),
            interpretation='Screening only; no automatic break-even threshold or deadline acceptance'))
    return result


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata):
    import numpy as np
    directory = Path(directory)
    delays = BOUNDARY_DELAYS if adapter.config.get('m2_idle_boundary') else DELAYS
    path = directory/'idle_payback.json'
    result = dict(metadata, status='running', rows=[], model_loads=1,
        scope='Saved Full inputs, resident-model static idle down/up versus high hold; not live Continue or task energy',
        plan=plan(delays), repeats=REPEATS, after_idle_seconds=AFTER_IDLE_SECONDS,
        power_boundary='INA3221 VDD_IN module input', production_deadline_seconds=None,
        sensor_settling_validated=False, first_gpu_timestamp=None,
        limitations=['Five grouped matched repeats; order alternates between durations, not independent long runs.',
            'Common measurement tail reduces boundary sensitivity but does not prove complete sensor settling.',
            'No physical actions or live sensor acquisition; no valid-buffer or task-quality claim.',
            'Readback is not first GPU start. Readiness overhead is reported, not automatically accepted.'])
    reference = {}
    monitor = None
    try:
        if memory_record(torch, 'idle_payback_ready')['available_mib'] < 900:
            raise RuntimeError('Paired idle RAM below900MiB')
        for delay, profile in plan(delays):
            label = str(delay).replace('.', 'p')+'_'+profile
            monitor = PowerMonitor(.02, max_gap_seconds=.1)
            monitor.start()
            time.sleep(.05)
            try:
                with PairedIdleLease(profile, delay, directory, label) as lease:
                    for index in range(REPEATS):
                        # Identical preceding inference/noise establishes matched load history.
                        torch.manual_seed(20261024+index)
                        prep = {}
                        adapter.predict(snapshot, prompt, previous, delay=5, executed=5, request=prep)
                        if prep['rtc_guidance'].get('variable_length_fallback'):
                            raise RuntimeError('Unexpected preparation fallback')
                        torch.manual_seed(20261024+index)
                        lease.send('RUN')
                        event = lease.receive('SERVE', index)
                        origin = event['down']['command_monotonic']
                        requested = origin+delay
                        request = {}
                        start = time.monotonic()
                        actions = adapter.predict(snapshot, prompt, previous, delay=5, executed=5, request=request)
                        end = time.monotonic()
                        if not np.isfinite(actions).all() or request['rtc_guidance'].get('variable_length_fallback'):
                            raise RuntimeError('Nonfinite actions or unexpected fallback')
                        if index not in reference:
                            reference[index] = actions.copy()
                        diff = float(np.max(np.abs(actions-reference[index])))
                        if diff != 0:
                            raise RuntimeError('Matched actions changed across idle profiles')
                        stop = origin+delay+AFTER_IDLE_SECONDS
                        if end > origin+delay+1.:
                            raise RuntimeError('Inference exceeded common1s measurement allowance')
                        # Two common endpoints expose sensitivity to the sampling tail.
                        time.sleep(max(0., stop-time.monotonic()))
                        monitor.sample()
                        time.sleep(.025)
                        monitor.sample()
                        cycle = energy_window(monitor, origin, stop)
                        early = energy_window(monitor, origin, stop-1.)
                        ready = energy_window(monitor, origin, end)
                        if any(w['energy_j'] is None for w in (cycle, early, ready)):
                            raise RuntimeError('Paired cycle power coverage incomplete')
                        check = frequency_check(lease.events, 'baseline', start, end)
                        if not check['matches']:
                            raise RuntimeError('Inference clock mismatch')
                        low_start = event['down']['observed_match_monotonic']
                        low_stop = event['up']['command_monotonic']
                        hold_check = frequency_check(lease.events, profile, low_start, low_stop)
                        row = dict(idle_seconds=delay, profile=profile, index=index, timing=event,
                            preparation=prep, inference_started_monotonic=start, inference_ended_monotonic=end,
                            actions_ready_from_origin_ms=(end-origin)*1000,
                            request_dispatch_lateness_ms=(start-requested)*1000,
                            inference_ms=(end-start)*1000, matched_cycle=cycle, early_cycle=early,
                            through_actions_ready=ready, frequency_check=check, hold_frequency_check=hold_check,
                            short_hold_sampling_unresolved=hold_check['sample_count']==0,
                            action_max_abs_difference=diff, action_sha256=hashlib.sha256(actions.tobytes()).hexdigest())
                        if hold_check['mismatches']:
                            raise RuntimeError('Idle clock mismatch')
                        result['rows'].append(row)
                        _write_json_atomically(path, result)
                        lease.send('DONE')
                        lease.receive('REQUEST_ACK', index)
                        print('IDLE_PAYBACK '+label+' index='+str(index), flush=True)
                result.setdefault('restored_conditions', []).append(label)
            finally:
                monitor.stop()
                _write_json_atomically(directory/(label+'_power.json'), [asdict(r) for r in monitor.readings])
        result.update(status='completed', restored=True, summary=summarize(result['rows'], delays=delays))
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if monitor is not None:
            monitor.stop()
        _write_json_atomically(path, result)
    return result
