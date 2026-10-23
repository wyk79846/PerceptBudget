"""Four-point prepared-input screen, one resident model, finite root leases."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import time

from perceptbudget.device_clocks import PROFILES, SCREEN_PROFILES, MATRIX_PROFILES
from perceptbudget.device_control import inventory
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 25


class Lease:
    """Root child writes no user paths; ordinary user captures its stdout."""
    def __init__(self, profile, directory):
        if profile not in PROFILES:
            raise ValueError('Unknown profile')
        self.profile = profile
        self.path = directory/(profile+'_clocks.jsonl')
        self.events = []
        self.lines = queue.Queue()
        self.proc = None
        self.duration = LEASE_SECONDS

    def command(self):
        return ['sudo', '-n', '/usr/bin/python3', str(ROOT/'scripts/control_device_clocks.py'),
                'lease', '--profile', self.profile, '--seconds', str(self.duration), '--trace']

    def __enter__(self):
        self.proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        def consume():
            with self.path.open('w') as out:
                for line in self.proc.stdout:
                    out.write(line)
                    out.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        event = {'message': line.strip()}
                    self.events.append(event)
                    self.lines.put(event)
            self.lines.put({'status': 'EOF'})
        self.thread = threading.Thread(target=consume, daemon=True)
        self.thread.start()
        try:
            while True:
                event = self.lines.get(timeout=10)
                if event.get('status') == 'READY':
                    self.ready = event
                    self.deadline = event['monotonic']+self.duration
                    return self
                if event.get('status') == 'EOF':
                    raise RuntimeError('No clock READY; inspect '+str(self.path))
        except BaseException:
            self.finish()
            raise

    def ensure(self, remaining=0):
        if self.proc.poll() is not None or time.monotonic()+remaining >= self.deadline:
            raise RuntimeError('Clock lease ended or insufficient time; measurement invalid')

    def finish(self):
        # No signals or root model execution: finite child naturally restores.
        try:
            code = self.proc.wait(timeout=self.duration+15)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Clock helper did not exit; manual restore required')
        self.thread.join(timeout=2)
        if not any(e.get('message') == 'RESTORED' for e in self.events):
            raise RuntimeError('Clock restoration unconfirmed; inspect '+str(self.path))
        if code != 0:
            raise RuntimeError(f'Clock helper failed (exit{code}) AFTER successful restoration; inspect {self.path}')

    def __exit__(self, *exc):
        self.finish()


def energy_window(monitor, start, stop):
    report = monitor.integrate_report(start, stop)
    energy = report.total_energy_mj
    return dict(start_monotonic=start, stop_monotonic=stop,
                duration_seconds=stop-start, power_coverage=asdict(report),
                energy_j=None if energy is None else energy/1000,
                mean_power_w=None if energy is None else energy/1000/(stop-start))


def frequency_check(events, profile, start, stop):
    gpu, emc, cpu = PROFILES[profile]
    rows = [e for e in events if e.get('status') == 'SAMPLE'
            and start <= e['monotonic'] <= stop]
    mismatches = []
    for event in rows:
        for name, actual in event['actual'].items():
            expected = gpu if name.endswith('/cur_freq') else emc if name.endswith('/rate') else cpu
            if actual != expected:
                mismatches.append(dict(monotonic=event['monotonic'], node=name,
                                       actual=actual, requested=expected))
    return dict(sample_count=len(rows), matches=bool(rows) and not mismatches,
                mismatches=mismatches, note='0.2s polling; not a hard real-time guarantee')


def prepared_prediction(adapter, snapshot, prompt, previous, *, fast_clocks, seed, torch):
    """Fresh Full before each isolated Fast, excluded from Fast compute timing."""
    if fast_clocks:
        adapter.config['m2_reuse_enabled'] = False
        adapter.bootstrap(snapshot, prompt, dict(observation_step=0, prediction_id='prepared_full'))
        adapter.config['m2_reuse_enabled'] = True
    torch.manual_seed(seed)
    request = dict(observation_step=5, prediction_id=1) if fast_clocks else {}
    return request


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata, *, fast_clocks=False, visual_clocks=False):
    import numpy as np
    directory = Path(directory)
    result = dict(metadata, status='running', conditions=[], clock_settings_changed=False,
                  scope='prepared Full input plus RTC compute and static resident hold; not service L, '
                        'task success, true Continue control load, or the 27-cell table',
                  repeats=5, fixed_delay_steps=5, fixed_executed_steps=5,
                  hold_seconds=3, lease_seconds=LEASE_SECONDS,
                  profile_order=list(MATRIX_PROFILES if fast_clocks or visual_clocks else SCREEN_PROFILES), power_boundary='INA3221 VDD_IN module input')
    if visual_clocks:
        result['scope'] = ('Prepared '+('dual-view factor2 token-pooling' if metadata.get('vision_pool_factor') == 2 else 'single-wrist-view')+
                           ' context and complete RTC compute; not live service, quality certification or task energy')
    if fast_clocks:
        result['scope'] = 'Isolated old-KV expert-only RTC compute; fresh Full before each sample excluded; not fresh perception, service L, success rate or task energy'
    _write_json_atomically(directory/'clock_screen.json', result)
    reference_actions = None
    try:
        for profile in result['profile_order']:
            print('CLOCK_PROFILE_START='+profile, flush=True)
            row = dict(profile=profile, target=PROFILES[profile], requests=[])
            outputs = []
            with Lease(profile, directory) as lease:
                result['clock_settings_changed'] = True
                row['ready'] = lease.ready
                row['thermal_at_start'] = inventory()['thermal']
                # Fixed settle and same warmup each time, outside measured windows.
                time.sleep(.5)
                for _ in range(2):
                    lease.ensure(10)
                    warm_request = (prepared_prediction(adapter, snapshot, prompt, previous,
                        fast_clocks=True, seed=20261021, torch=torch) if fast_clocks else {})
                    adapter.predict(snapshot, prompt, previous, delay=5, executed=5, request=warm_request)
                monitor = PowerMonitor(.02, max_gap_seconds=.1)
                monitor.start()
                try:
                    lease.ensure(8)
                    start = time.monotonic()
                    for i in range(5):
                        lease.ensure(4)
                        request = prepared_prediction(adapter, snapshot, prompt, previous,
                            fast_clocks=fast_clocks, seed=20261021+i, torch=torch)
                        monitor.sample()
                        begin = time.monotonic()
                        actions = adapter.predict(snapshot, prompt, previous, delay=5,
                                                  executed=5, request=request)
                        end = time.monotonic()
                        monitor.sample()
                        if fast_clocks and request.get('kv_recompute', {}).get('mode') != 'expert_only':
                            raise RuntimeError('Fast sample did not skip context compute')
                        lease.ensure()
                        finite = bool(np.isfinite(actions).all())
                        fallback = request.get('rtc_guidance', {}).get('variable_length_fallback')
                        if not finite or fallback:
                            raise RuntimeError('Nonfinite actions or slow fallback')
                        row['requests'].append(dict(start_monotonic=begin, stop_monotonic=end,
                            milliseconds=(end-begin)*1000, finite_actions=finite,
                            action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                            guidance=request.get('rtc_guidance'), kv_recompute=request.get('kv_recompute'),
                            energy=energy_window(monitor, begin, end)))
                        outputs.append(actions.copy())
                    stop = time.monotonic()
                    monitor.sample()
                    # Let residual request power settle, not counted in hold.
                    time.sleep(.5)
                    lease.ensure(3.5)
                    hold_start = time.monotonic()
                    time.sleep(3)
                    hold_stop = time.monotonic()
                    monitor.sample()
                    lease.ensure()
                finally:
                    monitor.stop()
                row['compute'] = energy_window(monitor, start, stop)
                row['static_hold'] = energy_window(monitor, hold_start, hold_stop)
                row['raw_power'] = [asdict(x) for x in monitor.readings]
                row['median_ms'] = statistics.median(r['milliseconds'] for r in row['requests'])
                row['mean_compute_energy_per_request_j'] = (
                    None if row['compute']['energy_j'] is None else row['compute']['energy_j']/5)
                if fast_clocks:
                    energies = [r['energy']['energy_j'] for r in row['requests']]
                    if any(e is None for e in energies):
                        raise RuntimeError('Incomplete isolated Fast energy coverage')
                    row['mean_compute_energy_per_request_j'] = statistics.mean(energies)
                    row['compute']['scope'] = 'Batch includes excluded Full preparations; NOT Fast-only energy'
                # Save before waiting out lease, so interrupted runs retain samples.
                _write_json_atomically(directory/(profile+'_measurement.json'), row)
            row['restored'] = True
            row['thermal_after_restore'] = inventory()['thermal']
            row['clock_trace'] = lease.events
            row['compute_frequency_check'] = frequency_check(lease.events, profile, start, stop)
            row['hold_frequency_check'] = frequency_check(lease.events, profile, hold_start, hold_stop)
            if (fast_clocks or visual_clocks) and (not row['compute_frequency_check']['matches'] or not row['hold_frequency_check']['matches']):
                raise RuntimeError('Fast frequency check failed')
            outputs = np.asarray(outputs)
            np.savez_compressed(directory/(profile+'_actions.npz'), actions=outputs)
            if reference_actions is None:
                reference_actions = outputs
            row['max_abs_action_difference_vs_baseline'] = float(np.max(np.abs(outputs-reference_actions)))
            if (fast_clocks or visual_clocks) and row['max_abs_action_difference_vs_baseline'] != 0:
                raise RuntimeError('Fast actions changed across clock profiles')
            result['conditions'].append(row)
            _write_json_atomically(directory/'clock_screen.json', result)
            print('CLOCK_PROFILE_DONE='+profile+' median_ms='+str(row['median_ms']), flush=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(directory/'clock_screen.json', result)
    return result
