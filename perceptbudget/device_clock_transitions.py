"""Resident static switching diagnostic; not a closed-loop Continue measurement."""
from dataclasses import asdict
from pathlib import Path
import statistics

from perceptbudget.device_clock_screen import Lease, ROOT, energy_window, frequency_check
from perceptbudget.device_control import inventory
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor


class CycleLease(Lease):
    def __init__(self, profile, directory, label):
        super().__init__(profile, directory)
        self.path = directory/(label+'_clocks.jsonl')
        self.duration = 40

    def command(self):
        return ['sudo', '-n', '/usr/bin/python3', str(ROOT/'scripts/control_device_clocks.py'),
                'cycles', '--profile', self.profile]


def summarize_cycles(events, monitor, profile):
    rows = []
    for i in range(5):
        low = next(e for e in events if e.get('status') == 'PHASE' and
                   e['cycle'] == i and e['phase'] == 'low')
        high = next(e for e in events if e.get('status') == 'PHASE' and
                    e['cycle'] == i and e['phase'] == 'high')
        end = next(e['monotonic'] for e in events if e.get('status') == 'CYCLE_END' and e['cycle'] == i)
        low_start, low_stop = low['observed_match_monotonic']+.5, high['command_monotonic']-.2
        high_start, high_stop = high['observed_match_monotonic']+.5, end-.2
        if low_stop <= low_start or high_stop <= high_start:
            raise RuntimeError('Incomplete steady hold window')
        low_check = frequency_check(events, profile, low_start, low_stop)
        high_check = frequency_check(events, 'baseline', high_start, high_stop)
        for check in (low_check, high_check):
            check['note'] = '20ms software polling, not instantaneous hardware transition time'
            if not check['matches']:
                raise RuntimeError('Actual frequency mismatch or missing samples')
        windows = dict(cycle_energy=energy_window(monitor, low['command_monotonic'], end),
                       low_hold=energy_window(monitor, low_start, low_stop),
                       high_hold=energy_window(monitor, high_start, high_stop))
        if any(window['energy_j'] is None for window in windows.values()):
            raise RuntimeError('Power coverage incomplete; no complete transition energy estimate')
        rows.append(dict(cycle=i, low=low, high=high, end_monotonic=end,
            down_observed_readback_ms=1000*(low['observed_match_monotonic']-low['command_monotonic']),
            up_observed_readback_ms=1000*(high['observed_match_monotonic']-high['command_monotonic']),
            low_frequency_check=low_check, high_frequency_check=high_check, **windows))
    return rows


def mean(values):
    return statistics.mean(values)


def transition_estimates(conditions):
    pre = conditions[0]['cycles']
    post = conditions[-1]['cycles']
    baseline_powers = [r[phase]['mean_power_w'] for rows in (pre, post)
                       for r in rows for phase in ('low_hold', 'high_hold')]
    before = mean(r[phase]['mean_power_w'] for r in pre for phase in ('low_hold', 'high_hold'))
    after = mean(r[phase]['mean_power_w'] for r in post for phase in ('low_hold', 'high_hold'))
    uncertainty_w = max(abs(after-before)/2,
                        statistics.stdev(baseline_powers)/(len(baseline_powers)**.5))
    summaries = []
    for index, condition in enumerate(conditions):
        cycles = condition['cycles']
        low_power = mean(r['low_hold']['mean_power_w'] for r in cycles)
        high_power = before+(after-before)*index/(len(conditions)-1)
        out = dict(label=condition['label'], profile=condition['profile'], restored=True,
                   cycles=len(cycles), low_hold_mean_power_w=low_power,
                   high_hold_mean_power_w=mean(r['high_hold']['mean_power_w'] for r in cycles),
                   down_readback_median_ms=statistics.median(r['down_observed_readback_ms'] for r in cycles),
                   up_readback_median_ms=statistics.median(r['up_observed_readback_ms'] for r in cycles))
        if condition['profile'] != 'baseline':
            residuals = []
            for row in cycles:
                duration = row['cycle_energy']['duration_seconds']
                low_duration = row['high']['command_monotonic']-row['low']['command_monotonic']
                ideal = low_power*low_duration+high_power*(duration-low_duration)
                residuals.append(row['cycle_energy']['energy_j']-ideal)
            extra = mean(residuals)
            # Practical noise/drift screen, NOT a formal confidence interval.
            noise = 2*(statistics.stdev(residuals)/(len(residuals)**.5)+
                       uncertainty_w*mean(r['cycle_energy']['duration_seconds'] for r in cycles))
            difference = high_power-low_power
            resolved = extra > noise and difference > 0
            out.update(reference_high_power_w=high_power, hold_saving_w=difference,
                       transition_excess_energy_j_estimate=extra,
                       practical_noise_bound_j=noise,
                       transition_energy_status='positive_above_observed_noise' if resolved else 'unresolved',
                       break_even_seconds=extra/difference if resolved else None,
                       warning='Mean complete down/up cycle residual only; not per-direction energy, '
                               'single-event tail, GPU wake latency or real-control break-even')
        summaries.append(out)
    return summaries


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata):
    directory = Path(directory)
    result = dict(metadata, status='running', conditions=[], clock_settings_changed=False,
                  scope='resident model static hold and five down/up cycles per condition; '
                        'not true Continue, first-operator/new-action recovery, or the 27-cell table',
                  power_boundary='INA3221 VDD_IN module input',
                  frequency_poll_seconds=.02, power_poll_seconds=.02,
                  cycles_per_condition=5, phase_seconds=3)
    path = directory/'clock_transitions.json'
    try:
        # Bracket the two combinations to expose drift, no extra model load.
        plan = [('baseline_pre', 'baseline'), ('gpu_emc', 'gpu_emc'),
                ('gpu_emc_cpu', 'gpu_emc_cpu'), ('baseline_post', 'baseline')]
        for label, profile in plan:
            print('CLOCK_TRANSITION_START='+label, flush=True)
            temperature = inventory()['thermal']
            monitor = PowerMonitor(.02, max_gap_seconds=.1)
            monitor.start()
            try:
                with CycleLease(profile, directory, label) as lease:
                    result['clock_settings_changed'] = True
                    # Root runs only this fixed finite schedule. No inference
                    # during windows; model weights/graphs remain resident.
                    pass
            finally:
                monitor.stop()
                _write_json_atomically(directory/(label+'_power.json'),
                                       [asdict(r) for r in monitor.readings])
            cycles = summarize_cycles(lease.events, monitor, profile)
            condition = dict(label=label, profile=profile, cycles=cycles, restored=True,
                             thermal_at_start=temperature, thermal_after=inventory()['thermal'])
            result['conditions'].append(condition)
            _write_json_atomically(path, result)
            print('CLOCK_TRANSITION_DONE='+label, flush=True)
        result.update(status='completed', summary=transition_estimates(result['conditions']))
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(path, result)
    return result
