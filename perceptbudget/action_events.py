"""Action-command hints shared by offline review and the future M3 controller."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class EventConfig:
    gripper_threshold: float = .2
    confirm_steps: int = 2
    minimum_motion: float = .1
    slowdown_ratio: float = .5
    turn_degrees: float = 60.
    match_steps: int = 5
    slowdown_window_steps: int = 1

    def __post_init__(self):
        if not (0 < self.gripper_threshold <= 1 and self.minimum_motion > 0
                and 0 < self.slowdown_ratio < 1 and 0 < self.turn_degrees < 180):
            raise ValueError('Invalid action-event thresholds')
        if type(self.confirm_steps) is not int or self.confirm_steps < 1:
            raise ValueError('confirm_steps must be a positive integer')
        if type(self.match_steps) is not int or self.match_steps < 0:
            raise ValueError('match_steps must be a nonnegative integer')
        if type(self.slowdown_window_steps) is not int or self.slowdown_window_steps not in (1,3,5):
            raise ValueError('slowdown_window_steps must be 1, 3, or 5')
        if not all(math.isfinite(v) for v in (self.gripper_threshold,
                self.minimum_motion, self.slowdown_ratio, self.turn_degrees)):
            raise ValueError('Event thresholds must be finite')


def _actions(actions):
    result = [tuple(float(v) for v in a) for a in actions]
    if any(len(a) != 7 or not all(math.isfinite(v) for v in a) for a in result):
        raise ValueError('Events require finite decoded LIBERO 7D actions')
    return result


def detect_events(actions, config=None, *, start_offset=0):
    """Return future command events; translation is in controller command units.

    Read the full chunk to retain hysteresis before start_offset, then discard
    consumed events. These are command changes, not contact/grasp labels.
    """
    config = config or EventConfig()
    actions = _actions(actions)
    if type(start_offset) is not int or not 0 <= start_offset <= len(actions):
        raise ValueError('start_offset must be inside the chunk')
    if not actions:
        return []
    n = config.confirm_steps
    motion = [math.sqrt(sum(x*x for x in a[:3])) for a in actions]
    events = []

    def emit(kind, i, **detail):
        if i >= start_offset:
            events.append(dict(kind=kind, offset=i, confirmed_offset=i+n-1, **detail))

    def grip(value):
        return 1 if value >= config.gripper_threshold else -1 if value <= -config.gripper_threshold else 0

    state = 0
    pending, begin = 0, 0
    for i, a in enumerate(actions):
        value = grip(a[6])
        if not value or value == state:
            pending = 0
            continue
        if value != pending:
            pending, begin = value, i
        if i-begin+1 >= n:
            if state:
                emit('gripper_close' if value > 0 else 'gripper_open', begin,
                     from_command=state, to_command=value)
            state, pending = value, 0

    slowed = turned = False
    for i in range(1, len(actions)-n+1):
        prior = motion[i-1]
        reference = median(motion[max(0,i-config.slowdown_window_steps):i])
        if prior >= config.minimum_motion:
            low = all(v <= prior*config.slowdown_ratio for v in motion[i:i+n])
            confirmed = reference >= config.minimum_motion and all(
                v <= reference*config.slowdown_ratio for v in motion[i:i+n])
            # Confirm an existing onset, never introduce extra onsets or
            # postpone a rejected one into the same low-motion segment.
            if low and not slowed and confirmed:
                detail=({} if config.slowdown_window_steps==1 else
                        dict(reference_command_norm=reference,reference_window_steps=min(i,config.slowdown_window_steps)))
                emit('slowdown', i, previous_command_norm=prior,
                     next_command_norm=motion[i],**detail)
            slowed = low
        elif motion[i] >= config.minimum_motion:
            slowed = False
        direction_changed = False
        angles = []
        if prior >= config.minimum_motion and all(v >= config.minimum_motion for v in motion[i:i+n]):
            for j in range(i, i+n):
                cosine = sum(a*b for a, b in zip(actions[i-1][:3], actions[j][:3]))/(prior*motion[j])
                angles.append(math.degrees(math.acos(max(-1., min(1., cosine)))))
            direction_changed = all(angle >= config.turn_degrees for angle in angles)
        if direction_changed and not turned:
            emit('turn', i, angle_degrees=angles[0])
        turned = direction_changed
    return sorted(events, key=lambda e: (e['offset'], e['kind']))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_episode(run, config=None):
    """Read an existing episode without changing it; use only past ticks for ETA."""
    config = config or EventConfig()
    run = Path(run)
    trace_path = run/'rtc/trace.jsonl'
    controls, commits = [], []
    with trace_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('event') == 'control_step':
                controls.append(row)
            elif row.get('event') == 'prediction_committed':
                commits.append({k: row[k] for k in
                    ('prediction_id', 'observation_step', 'commit_step', 'commit_monotonic')})
    if not controls or [r['step'] for r in controls] != list(range(len(controls))):
        raise ValueError('Require one complete consecutively indexed episode')
    qs = {r['q'] for r in controls}
    if len(qs) != 1:
        raise ValueError('Mixed q values in episode')
    q = next(iter(qs))
    requests_path = run/f'rtc/q{q}_requests.json'
    requests = {r['prediction_id']: r for r in json.loads(requests_path.read_text())}
    metadata = json.loads((run/'run.json').read_text())
    cfg = metadata['config_snapshot']
    summary_path = run/'summary.json'
    episode = json.loads(summary_path.read_text()).get('episode', {}) if summary_path.exists() else {}
    if any(r['applied_arm_gain'] != 1 for r in controls):
        raise ValueError('M3 recording requires unmodified executed actions')
    initial_period = cfg.get('rtc_initial_control_period_seconds', .05)
    actual = detect_events([r['action'] for r in controls], config)
    for event in actual:
        event.update(step=event['offset'], command_monotonic=controls[event['offset']]['action_start_monotonic'])
    committed = {c['prediction_id'] for c in commits}
    forecasts, chunks = [], []
    for commit in commits:
        pid, origin, step = (commit[k] for k in ('prediction_id', 'observation_step', 'commit_step'))
        request = requests[pid]
        actions = request['actions']
        offset = max(0, step-origin)
        if offset > len(actions):
            raise ValueError('Committed chunk expired before publication')
        prior = [r for r in controls if r['action_end_monotonic'] <= commit['commit_monotonic']][-17:]
        intervals = [b['action_start_monotonic']-a['action_start_monotonic'] for a,b in zip(prior,prior[1:])]
        period = median(intervals) if intervals else initial_period
        found = detect_events(actions, config, start_offset=offset)
        chunks.append(dict(prediction_id=pid, observation_step=origin, commit_step=step,
                           consumed_prefix=offset, remaining_output_actions=len(actions)-offset,
                           control_period_estimate_seconds=period, events=len(found)))
        for event in found:
            target = origin+event['offset']
            time_estimate = commit['commit_monotonic']+(target-step)*period
            matches = [a for a in actual if a['kind']==event['kind'] and a['step']>=step
                       and abs(a['step']-target)<=config.match_steps]
            nearest = min(matches, key=lambda a: abs(a['step']-target)) if matches else None
            source = controls[target]['source'] if target < len(controls) else None
            disposition = ('episode_ended_before_target' if target >= len(controls) else
                           'target_executed_from_this_chunk' if source and source['prediction_id']==pid else
                           'target_overwritten_or_underflow')
            forecasts.append(dict(event, event_id=f'{pid}:{event["kind"]}:{event["offset"]}',
                prediction_id=pid, target_step=target, known_step=step,
                known_monotonic=commit['commit_monotonic'], estimated_event_monotonic=time_estimate,
                predicted_lead_seconds=(target-step)*period, disposition=disposition,
                observed_command_step=nearest['step'] if nearest else None,
                command_step_error=nearest['step']-target if nearest else None,
                command_time_error_seconds=nearest['command_monotonic']-time_estimate if nearest else None,
                observed_command_source=controls[nearest['step']]['source'] if nearest else None))

    # Physical evidence accompanies, but never relabels, command predictions.
    physical = []
    contact_before = None
    for row in controls:
        contacts = []
        for c in row['after'].get('contacts') or []:
            names = (c.get('geom1') or '', c.get('geom2') or '')
            if c.get('distance', 0) <= 0 and any('finger' in x for x in names):
                if any(x and not any(part in x for part in ('robot', 'gripper', 'finger')) for x in names):
                    contacts.append(list(names))
        present = bool(contacts)
        if contact_before is not None and present != contact_before:
            physical.append(dict(kind='finger_external_contact_start' if present else 'finger_external_contact_end',
                                 step=row['step'], monotonic=row['action_end_monotonic'], pairs=contacts))
        contact_before = present
    task_energy = episode.get('including_drain_power',{}).get('total_energy_mj')
    late = [r['m3_timing']['arrival_lateness_seconds'] for r in requests.values()
            if r['prediction_id'] in committed and 'arrival_lateness_seconds' in r.get('m3_timing',{})]
    return dict(status='completed', run=str(run), mode=cfg.get('m3_scheduler_mode','record_only'), q=q,
        suite=cfg.get('rtc_suite'), task_id=controls[0]['task_id'], control_steps=len(controls),
        episode_metrics=dict(outcome=episode.get('status'),task_seconds=episode.get('real_seconds'),
            total_seconds_including_drain=episode.get('including_drain_seconds'),
            total_energy_j=task_energy/1000 if task_energy is not None else None,
            underflow_steps=len(episode.get('underflow_steps',[])),
            refresh_intervals_steps=episode.get('actual_refresh_intervals_steps'),
            deadline_late_requests=sum(t>0 for t in late),maximum_arrival_lateness_seconds=max(late,default=0.)),
        event_config=asdict(config), action_representation='decoded LIBERO delta-pose controller commands; xyz command norm, not physical speed',
        summary=dict(committed_chunks=len(commits), uncommitted_requests=len(set(requests)-committed),
                     forecast_counts=dict(Counter(e['kind'] for e in forecasts)),
                     actual_command_counts=dict(Counter(e['kind'] for e in actual)),
                     disposition_counts=dict(Counter(e['disposition'] for e in forecasts)),
                     matched_command_forecasts=sum(e['observed_command_step'] is not None for e in forecasts)),
        chunks=chunks, forecasts=forecasts, actual_command_events=actual, physical_events=physical,
        matching='Nearest same command event within match_steps; repeated forecasts may match one event. Contact evidence is separate. Overwritten tails are not executed ground truth.',
        sources=[dict(path=str(p), sha256=_sha(p)) for p in (trace_path, requests_path, run/'run.json',
            *([summary_path] if summary_path.exists() else []))])
