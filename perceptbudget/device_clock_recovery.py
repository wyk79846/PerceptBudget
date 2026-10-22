"""Prepared-input recovery plus original RTC, not full live-observation service L."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import statistics
import time

from perceptbudget.device_clock_screen import Lease, ROOT, energy_window, frequency_check
from perceptbudget.device_control import inventory
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor


PLAN = (('hot_pre', 'baseline', 0.), ('full_short', 'baseline', .2),
        ('low_short', 'gpu_emc', .2), ('full_long', 'baseline', 1.),
        ('low_long', 'gpu_emc', 1.), ('hot_post', 'baseline', 0.))


class RecoveryLease(Lease):
    def __init__(self, profile, idle_seconds, directory, label):
        if profile not in ('baseline', 'gpu_emc') or idle_seconds not in (0., .2, 1.):
            raise ValueError('Recovery scenario outside whitelist')
        super().__init__(profile, directory)
        self.idle = idle_seconds
        self.path = directory/(label+'_clocks.jsonl')
        self.duration = 40

    def command(self):
        return ['sudo', '-n', '/usr/bin/python3', str(ROOT/'scripts/control_device_clocks.py'),
                'recovery', '--profile', self.profile, '--idle-seconds', str(self.idle)]

    def send(self, token):
        if token not in ('RUN', 'DONE'):
            raise ValueError('Only RUN/DONE accepted')
        self.proc.stdin.write(token+'\n')
        self.proc.stdin.flush()

    def receive(self, status, index):
        deadline = time.monotonic()+4
        while time.monotonic() < deadline:
            event = self.lines.get(timeout=max(.01, deadline-time.monotonic()))
            if event.get('status') == status:
                if event['request_index'] != index:
                    raise RuntimeError('Recovery request index mismatch')
                return event
            if event.get('status') == 'EOF':
                raise RuntimeError('Root helper exited before recovery acknowledgement')
        raise TimeoutError('Recovery status not received')

    def __exit__(self, *exc):
        # EOF makes root restore promptly if the worker fails between tokens.
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        super().__exit__(*exc)


def summarize(rows):
    return dict(requests=len(rows),
        recovery_compute_median_ms=statistics.median(r['recovery_compute_ms'] for r in rows),
        predict_median_ms=statistics.median(r['predict_ms'] for r in rows),
        up_readback_median_ms=statistics.median(r['up_readback_ms'] for r in rows),
        mean_recovery_energy_j=statistics.mean(r['recovery']['energy_j'] for r in rows),
        mean_cycle_energy_j=statistics.mean(r['cycle']['energy_j'] for r in rows),
        actual_idle_before_predict_seconds=[r['actual_idle_before_predict_seconds'] for r in rows])


def screen(adapter, snapshot, prompt, previous, torch, directory, metadata):
    import numpy as np
    directory = Path(directory)
    result = dict(metadata, status='running', conditions=[], clock_settings_changed=False,
        scope='prepared-input recovery plus Full RTC; includes root/worker coordination, '
              'excludes live observation acquisition; not full service L or closed-loop Continue',
        requested_idle_seconds=[0, .2, 1.], input_mode='saved paired observation/old actions/d5s5',
        power_boundary='INA3221 VDD_IN module input', requests_per_condition=5,
        total_requests=30, first_gpu_operator_timestamp=None)
    reference = None
    last_end = None
    path = directory/'clock_recovery.json'
    try:
        for label, profile, idle in PLAN:
            print('CLOCK_RECOVERY_START='+label, flush=True)
            row = dict(label=label, idle_profile=profile, requested_idle_seconds=idle, requests=[],
                       thermal_at_start=inventory()['thermal'])
            outputs = []
            monitor = PowerMonitor(.02, max_gap_seconds=.1)
            monitor.start()
            try:
                with RecoveryLease(profile, idle, directory, label) as lease:
                    result['clock_settings_changed'] = True
                    for index in range(5):
                        # Seed before RUN, no model work hidden in the idle phase.
                        torch.manual_seed(20261021+index)
                        lease.send('RUN')
                        event = lease.receive('SERVE', index)
                        request = {}
                        begin = time.monotonic()
                        actions = adapter.predict(snapshot, prompt, previous, delay=5,
                                                  executed=5, request=request)
                        end = time.monotonic()
                        lease.send('DONE')
                        lease.receive('REQUEST_ACK', index)
                        if not np.isfinite(actions).all() or request.get('rtc_guidance', {}).get('variable_length_fallback'):
                            raise RuntimeError('Nonfinite actions or unexpected fallback')
                        monitor.sample()
                        recovery = energy_window(monitor, event['up']['command_monotonic'], end)
                        cycle = energy_window(monitor, event['down']['command_monotonic'], end)
                        if recovery['energy_j'] is None or cycle['energy_j'] is None:
                            raise RuntimeError('Power coverage incomplete')
                        row['requests'].append(dict(index=index, timing=event,
                            inference_started_monotonic=begin, inference_ended_monotonic=end,
                            predict_ms=(end-begin)*1000,
                            recovery_compute_ms=(end-event['up']['command_monotonic'])*1000,
                            up_readback_ms=1000*(event['up']['observed_match_monotonic']-event['up']['command_monotonic']),
                            worker_dispatch_ms=1000*(begin-event['up']['observed_match_monotonic']),
                            actual_idle_before_predict_seconds=None if last_end is None else begin-last_end,
                            measured_idle_state_dwell_seconds=(event['up']['command_monotonic']-
                                                               event['down']['observed_match_monotonic']),
                            recovery=recovery, cycle=cycle,
                            action_sha256=hashlib.sha256(actions.tobytes()).hexdigest()))
                        last_end = end
                        outputs.append(actions.copy())
                        _write_json_atomically(directory/(label+'_requests.json'), row)
            finally:
                monitor.stop()
                _write_json_atomically(directory/(label+'_power.json'), [asdict(r) for r in monitor.readings])
            for request in row['requests']:
                if idle > 0:
                    idle_check = frequency_check(lease.events, profile,
                        request['timing']['down']['observed_match_monotonic'],
                        request['timing']['up']['command_monotonic'])
                    idle_check['note'] = '20ms software sampling during requested idle-state dwell'
                    if not idle_check['matches']:
                        raise RuntimeError('Idle phase did not remain at requested frequencies')
                    request['idle_frequency_check'] = idle_check
                check = frequency_check(lease.events, 'baseline',
                                        request['inference_started_monotonic'], request['inference_ended_monotonic'])
                check['note'] = '20ms software sampling during prediction, not hard real-time guarantee'
                if not check['matches']:
                    raise RuntimeError('Inference did not remain at baseline frequencies')
                request['prediction_frequency_check'] = check
            outputs = np.asarray(outputs)
            np.savez_compressed(directory/(label+'_actions.npz'), actions=outputs)
            if reference is None:
                reference = outputs
            row.update(restored=True, summary=summarize(row['requests']),
                       max_abs_action_difference_vs_hot_pre=float(np.max(np.abs(outputs-reference))),
                       thermal_after=inventory()['thermal'])
            result['conditions'].append(row)
            _write_json_atomically(path, result)
            print('CLOCK_RECOVERY_DONE='+label, flush=True)
        result.update(status='completed', summary=[dict(label=c['label'], profile=c['idle_profile'],
            requested_idle_seconds=c['requested_idle_seconds'], restored=c['restored'],
            max_abs_action_difference_vs_hot_pre=c['max_abs_action_difference_vs_hot_pre'],
            **c['summary']) for c in result['conditions']])
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(path, result)
    return result
