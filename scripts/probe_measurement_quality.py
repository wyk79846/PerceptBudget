#!/usr/bin/env python3
"""Read-only M2 sensor step response and paired sampler-overhead diagnostic."""
import fcntl
import json
from pathlib import Path
import resource
import statistics
import sys
import time
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perceptbudget.experiment import create_run, load_config, _write_json_atomically
from perceptbudget.power import PowerMonitor


class TimedMonitor(PowerMonitor):
    def __init__(self):
        super().__init__(.02, max_gap_seconds=.1)
        self.cpu_ns = 0
        self.calls = 0
    def sample(self):
        begin = time.thread_time_ns()
        try:
            return super().sample()
        finally:
            self.cpu_ns += time.thread_time_ns()-begin
            self.calls += 1


def observed_response(sensor):
    """Combined load/rail/sensor response, not a hardware update-rate estimate."""
    rows = []
    samples = [r for r in sensor['samples'] if r['power_mw'] is not None]
    for previous, block in zip(sensor['blocks'], sensor['blocks'][1:]):
        before = [r['power_mw']/1000 for r in samples
                  if previous['stop_monotonic']-.5 <= r['timestamp'] <= previous['stop_monotonic']]
        after = [r['power_mw']/1000 for r in samples
                 if block['stop_monotonic']-.5 <= r['timestamp'] <= block['stop_monotonic']]
        if not before or not after:
            rows.append(dict(mode=block['mode'], status='unavailable'))
            continue
        low, high = statistics.median(before), statistics.median(after)
        delta = high-low
        trace = [r for r in samples if block['start_monotonic'] <= r['timestamp'] < block['stop_monotonic']]
        row = dict(mode=block['mode'], before_w=low, after_w=high,
                   step_w=delta, status='observed' if abs(delta) >= 1. else 'small_or_unresolved')
        for fraction in (.1, .9):
            cross = next((r for r in trace if abs(delta) >= 1.
                          and (r['power_mw']/1000-low)/delta >= fraction), None)
            row['t'+str(int(fraction*100))+'_ms'] = None if cross is None else 1000*(cross['timestamp']-block['start_monotonic'])
        rows.append(row)
    return rows


def main():
    from perceptbudget.device_profile import sensor_steps
    from perceptbudget.device_control import inventory
    config = load_config(repo_root=ROOT)
    run = create_run(config, 'M2', 'measurement-quality', repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    result = dict(status='running', model_loaded=False, clock_settings_changed=False,
        persistent_permissions_changed=False,
        scope='Synthetic load response and idle sampler overhead, not VLA timing or absolute meter calibration',
        windows=[])
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result['device_before'] = inventory()
            result['sensor'] = sensor_steps(block_seconds=3., pairs=3)
            result['observed_response'] = observed_response(result['sensor'])
            _write_json_atomically(run/'measurement_quality.json', result)
            time.sleep(3.)
            # A common low-rate reference remains enabled in BOTH arms.
            # Therefore this estimates the incremental20ms sampler, not total monitoring.
            for pair in range(3):
                for enabled in ((False, True) if pair % 2 == 0 else (True, False)):
                    reference = PowerMonitor(.2, max_gap_seconds=.5)
                    measured = TimedMonitor()
                    reference.start()
                    if enabled:
                        measured.start()
                    before = resource.getrusage(resource.RUSAGE_SELF)
                    start = time.monotonic()
                    time.sleep(10.)
                    stop = time.monotonic()
                    after = resource.getrusage(resource.RUSAGE_SELF)
                    if enabled:
                        measured.stop()
                    reference.stop()
                    energy = reference.integrate_report(start, stop)
                    if energy.total_energy_mj is None:
                        raise RuntimeError('Reference power coverage incomplete')
                    cpu_seconds = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
                    row = dict(pair=pair, sampler_enabled=enabled, duration_seconds=stop-start,
                        process_cpu_seconds=cpu_seconds, process_one_core_fraction=cpu_seconds/(stop-start),
                        sampler_cpu_seconds=measured.cpu_ns/1e9, sampler_reads=measured.calls,
                        reference_energy_j=energy.total_energy_mj/1000,
                        reference_coverage=asdict(energy))
                    result['windows'].append(row)
                    _write_json_atomically(run/f'reference_{pair}_{int(enabled)}.json',
                                           [asdict(r) for r in reference.readings])
                    _write_json_atomically(run/'measurement_quality.json', result)
                    print('SAMPLER_WINDOW pair='+str(pair)+' enabled='+str(enabled), flush=True)
            pairs = []
            for pair in range(3):
                off = next(r for r in result['windows'] if r['pair']==pair and not r['sampler_enabled'])
                on = next(r for r in result['windows'] if r['pair']==pair and r['sampler_enabled'])
                pairs.append(dict(pair=pair,
                    extra_cpu_seconds=on['process_cpu_seconds']-off['process_cpu_seconds'],
                    extra_mean_power_w=on['reference_energy_j']/on['duration_seconds']-
                                       off['reference_energy_j']/off['duration_seconds']))
            result.update(status='completed', pairs=pairs,
                mean_extra_power_w=statistics.mean(r['extra_mean_power_w'] for r in pairs),
                energy_overhead_status='exploratory matched idle difference; drift may dominate, not a resolved single-event cost',
                limitations=['No model or clocks changed; overhead under VLA contention is not established.',
                    'Reference200ms sampler is present in both arms;20ms incremental sampler only.',
                    'Load-step response includes physical load ramp and sensor/driver filtering; no absolute calibration.',
                    'Three grouped idle pairs are not a tail model or precision energy certification.'])
            result['device_after'] = inventory()
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(run/'measurement_quality.json', result)
        meta=json.loads((run/'run.json').read_text()); meta['status']=result['status']
        _write_json_atomically(run/'run.json',meta)
        print('MEASUREMENT_QUALITY_STATUS='+json.dumps({k:result[k] for k in ('status',)}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
