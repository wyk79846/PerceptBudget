"""Bound one inference request, including executor draining, across processes."""
import json
import math
import time
from pathlib import Path

from perceptbudget.experiment import _write_json_atomically


class InferenceTimeout(TimeoutError):
    def __init__(self, record):
        self.record = record
        super().__init__('Inference request %s exceeded its %.3fs timeout' % (
            record['prediction_id'], record['timeout_seconds']))


def expired_request(path, now=None):
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get('state') not in ('armed','expired','idle'):
        raise ValueError('Unknown inference watchdog state')
    if record['state'] not in ('armed', 'expired'):
        return None
    deadline = record['deadline_monotonic']
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError('Invalid inference watchdog deadline')
    if (time.monotonic() if now is None else now) >= deadline:
        return record
    return None


class RequestWatchdog:
    def __init__(self, path, timeout_seconds):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('Use a positive finite inference timeout')
        self.path, self.timeout = Path(path), timeout_seconds
        self.record = None

    def arm(self, request, episode_directory):
        if self.record is not None:
            raise RuntimeError('Cannot overlap watched inference requests')
        start = request['observation_monotonic']
        self.record = dict(state='armed', prediction_id=request['prediction_id'],
            observation_step=request['observation_step'], start_monotonic=start,
            deadline_monotonic=start+self.timeout, timeout_seconds=self.timeout,
            episode_directory=str(episode_directory))
        _write_json_atomically(self.path, self.record)

    def check(self, completed_monotonic=None):
        if self.record is None:
            return
        now = time.monotonic() if completed_monotonic is None else completed_monotonic
        if now >= self.record['deadline_monotonic']:
            self.record.update(state='expired', detected_monotonic=time.monotonic())
            _write_json_atomically(self.path, self.record)
            raise InferenceTimeout(dict(self.record))

    def complete(self, ready_monotonic):
        self.check(ready_monotonic)
        if self.record is not None:
            _write_json_atomically(self.path, dict(self.record, state='idle', ready_monotonic=ready_monotonic))
            self.record = None


def inject_request_fault(request, config, episode_directory):
    """Only the explicit watchdog experiment changes request1's return time."""
    case = config.get('m3_watchdog_case')
    if case is None or request['prediction_id'] != 1:
        return
    if case not in ('late', 'underflow', 'hang'):
        raise ValueError('Unknown watchdog fault case')
    seconds = {'late': .25, 'underflow': 1.0, 'hang': None}[case]
    marker = dict(case=case, prediction_id=1, observation_step=request['observation_step'],
        injected_monotonic=time.monotonic(), added_return_delay_seconds=seconds,
        episode_directory=str(episode_directory))
    request['watchdog_fault'] = marker
    _write_json_atomically(Path(config['m3_watchdog_path']).parent/'fault_injection.json', marker)
    if case == 'hang':
        import threading
        threading.Event().wait()
    else:
        time.sleep(seconds)


def validate_fault_suite(directory):
    """A killed hang is a passed fault check, never a successful task episode."""
    directory = Path(directory)
    outer = json.loads((directory/'outer_watchdog_timeout.json').read_text())
    marker = json.loads((directory/'fault_injection.json').read_text())
    recording = json.loads((directory/'recording.json').read_text())
    checks = dict(hang_injected=marker['case']=='hang' and marker['prediction_id']==1,
        same_request=outer['episode_directory']==marker['episode_directory']
            and outer['prediction_id']==marker['prediction_id'],
        timeout_observed=outer['detected_monotonic']>=outer['deadline_monotonic'],
        completed_two_windows=len(recording.get('episodes', []))==2,
        one_model_load=recording.get('model_loads')==1)
    abort_path=Path(marker['episode_directory'])/'watchdog_abort.json'
    abort=json.loads(abort_path.read_text()) if abort_path.exists() else {}
    checks['local_timeout_recorded']=(abort.get('error')=='InferenceTimeout'
        and abort.get('request',{}).get('prediction_id')==1)
    cases=[]
    for index, case in enumerate(('late','underflow')):
        if index>=len(recording.get('episodes', [])):
            checks[case+'_passed']=False
            continue
        row=recording['episodes'][index]
        run=Path(row['run'])
        episode=json.loads((run/'summary.json').read_text())['episode']
        requests=json.loads((run/'rtc/q5_requests.json').read_text())
        injected=next((q for q in requests if q.get('watchdog_fault',{}).get('case')==case),None)
        timing=(injected or {}).get('m3_timing',{})
        late=timing.get('arrival_lateness_seconds',0)>0
        valid=(injected is not None and row['view']=='watchdog_'+case
            and row['audit_passed'] and episode['status']=='fault_window_completed'
            and (late if case=='late' else bool(episode['underflow_steps'])))
        checks[case+'_passed']=valid
        cases.append(dict(case=case, run=str(run), passed=valid,
            underflow_steps=episode['underflow_steps'],
            arrival_lateness_seconds=timing.get('arrival_lateness_seconds')))
    clock_paths=[Path(row['run'])/'rtc/runtime_clocks.jsonl' for row in recording.get('episodes', [])]
    clock_paths.append(Path(marker['episode_directory'])/'runtime_clocks.jsonl')
    # The privileged finite lease survives worker termination to restore clocks.
    until=time.monotonic()+5.
    while time.monotonic()<until:
        if all(p.exists() and 'RESTORED' in p.read_text().splitlines() for p in clock_paths):
            break
        time.sleep(.05)
    checks['all_clocks_restored']=all(p.exists() and 'RESTORED' in p.read_text().splitlines() for p in clock_paths)
    result=dict(status='completed' if all(checks.values()) else 'invalid', checks=checks,
        cases=cases+[dict(case='hang', request=outer)], model_loads=recording.get('model_loads'),
        scope='Injected real-model control failures, not task success-rate or energy comparison')
    _write_json_atomically(directory/'watchdog_checks.json', result)
    if result['status']!='completed':
        raise RuntimeError('Watchdog checks incomplete; inspect watchdog_checks.json')
    recording.update(status='completed',scope=result['scope'],watchdog_checks=str(directory/'watchdog_checks.json'))
    _write_json_atomically(directory/'recording.json',recording)
    active=Path(marker['episode_directory']).parent/'run.json'
    if active.exists():
        metadata=json.loads(active.read_text())
        metadata.update(status='aborted_by_watchdog',fault_case='hang')
        _write_json_atomically(active,metadata)
    return result
