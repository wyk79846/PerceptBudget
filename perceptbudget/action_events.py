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
    slowdown_confirm_steps: int | None = None
    slowdown_detector: str = 'drop'
    valley_radius_steps: int = 5
    valley_spacing_steps: int = 5

    def __post_init__(self):
        if not (0 < self.gripper_threshold <= 1 and self.minimum_motion > 0
                and 0 < self.slowdown_ratio < 1 and 0 < self.turn_degrees < 180):
            raise ValueError('Invalid action-event thresholds')
        if type(self.confirm_steps) is not int or self.confirm_steps < 1:
            raise ValueError('confirm_steps must be a positive integer')
        if type(self.match_steps) is not int or self.match_steps < 0:
            raise ValueError('match_steps must be a nonnegative integer')
        if self.slowdown_confirm_steps is not None and (
                type(self.slowdown_confirm_steps) is not int or self.slowdown_confirm_steps<1):
            raise ValueError('slowdown_confirm_steps must be a positive integer or None')
        if self.slowdown_detector not in ('drop','valley'):
            raise ValueError('Unknown slowdown detector')
        if any(type(x) is not int or x<1 for x in (self.valley_radius_steps,self.valley_spacing_steps)):
            raise ValueError('Valley radius and spacing must be positive integers')
        if not all(math.isfinite(v) for v in (self.gripper_threshold,
                self.minimum_motion, self.slowdown_ratio, self.turn_degrees)):
            raise ValueError('Event thresholds must be finite')


def _actions(actions):
    result = [tuple(float(v) for v in a) for a in actions]
    if any(len(a) != 7 or not all(math.isfinite(v) for v in a) for a in result):
        raise ValueError('Events require finite decoded LIBERO 7D actions')
    return result


def _phase_valleys(motion, config):
    """Engineering candidate on decoded XYZ-command amplitude, not physical speed."""
    size=len(motion)
    smooth=[median(motion[max(0,i-1):min(size,i+2)]) for i in range(size)]
    candidates=[]
    i=1
    while i<size-1:
        left=i
        while i+1<size and smooth[i+1]==smooth[left]:
            i+=1
        right=i
        if smooth[left]<smooth[left-1] and right+1<size and smooth[right]<smooth[right+1]:
            lo=max(0,left-config.valley_radius_steps)
            hi=min(size,right+config.valley_radius_steps+1)
            shoulder=min(max(smooth[lo:left]),max(smooth[right+1:hi]))
            floor=smooth[left]
            if shoulder>=config.minimum_motion and floor<=shoulder*config.slowdown_ratio:
                point=(left+right)//2
                candidates.append(dict(offset=point,confirmed_offset=min(size-1,hi),
                    detector='phase_valley',valley_command_norm=floor,shoulder_command_norm=shoulder,
                    prominence=shoulder-floor,relative_depth=(shoulder-floor)/shoulder,
                    smoothing_window_steps=3))
        i+=1
    selected=[]
    for event in sorted(candidates,key=lambda e:(-e['prominence'],e['offset'])):
        if all(abs(event['offset']-old['offset'])>=config.valley_spacing_steps for old in selected):
            selected.append(event)
    return sorted(selected,key=lambda e:e['offset'])


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

    def emit(kind, i, *, confirmation_steps=None, **detail):
        if i >= start_offset:
            count=n if confirmation_steps is None else confirmation_steps
            events.append(dict(kind=kind, offset=i, confirmed_offset=i+count-1, **detail))

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

    slowed = False
    slowdown_n=config.slowdown_confirm_steps or n
    for i in range(1, len(actions)-slowdown_n+1) if config.slowdown_detector=='drop' else ():
        prior = motion[i-1]
        if prior >= config.minimum_motion:
            low = all(v <= prior*config.slowdown_ratio for v in motion[i:i+slowdown_n])
            if low and not slowed:
                emit('slowdown', i, previous_command_norm=prior,
                     next_command_norm=motion[i],confirmation_steps=slowdown_n)
            slowed = low
        elif motion[i] >= config.minimum_motion:
            slowed = False
    if config.slowdown_detector=='valley':
        events.extend(dict(e,kind='slowdown') for e in _phase_valleys(motion,config) if e['offset']>=start_offset)
    turned = False
    for i in range(1, len(actions)-n+1):
        prior = motion[i-1]
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


def compare_gripper_windows(previous, current, commit_step, config=None, *, radius=3):
    """Compare already available forecasts at common absolute control indices.

    Decoded actions are controller commands, not predicted Cartesian poses.
    No similarity threshold or online suppression decision is made here.
    """
    config=config or EventConfig()
    if type(radius) is not int or radius<0 or type(commit_step) is not int:
        raise ValueError('Use an integer commit step and nonnegative radius')
    old_origin,new_origin=previous['observation_step'],current['observation_step']
    if any(type(x) is not int or x<0 for x in (old_origin,new_origin,commit_step)) or not old_origin<new_origin<=commit_step:
        raise ValueError('Require ordered prediction origins before the commit')
    old,new=_actions(previous['actions']),_actions(current['actions'])
    start=max(commit_step,old_origin,new_origin)
    end=min(old_origin+len(old),new_origin+len(new))
    old_events=[dict(e,target=old_origin+e['offset']) for e in detect_events(old,config)
                if e['kind'].startswith('gripper') and start<=old_origin+e['offset']<end
                and old_origin+e['confirmed_offset']<end]
    new_events=[dict(e,target=new_origin+e['offset']) for e in detect_events(new,config)
                if e['kind'].startswith('gripper') and new_origin+e['offset']>=start]

    def metrics(lo,hi):
        if lo>=hi:return None
        pairs=[(old[s-old_origin],new[s-new_origin]) for s in range(lo,hi)]
        xyz=[[b[k]-a[k] for k in range(3)] for a,b in pairs]
        rotation=[[b[k]-a[k] for k in range(3,6)] for a,b in pairs]
        norm=lambda v:math.sqrt(sum(x*x for x in v))
        angles=[];magnitude=[];cumulative=[0.,0.,0.];max_cumulative=0.
        for (a,b),delta in zip(pairs,xyz):
            an,bn=norm(a[:3]),norm(b[:3]);magnitude.append(abs(an-bn))
            if min(an,bn)>=config.minimum_motion:
                cosine=sum(a[k]*b[k] for k in range(3))/(an*bn)
                angles.append(math.degrees(math.acos(max(-1.,min(1.,cosine)))))
            cumulative=[x+y for x,y in zip(cumulative,delta)]
            max_cumulative=max(max_cumulative,norm(cumulative))
        grip=lambda v:1 if v>=config.gripper_threshold else -1 if v<=-config.gripper_threshold else 0
        return dict(start_step=lo,end_step_exclusive=hi,steps=hi-lo,
            xyz_rmse=math.sqrt(sum(x*x for v in xyz for x in v)/(3*len(pairs))),
            xyz_max_l2=max(norm(v) for v in xyz),
            rotation_rmse=math.sqrt(sum(x*x for v in rotation for x in v)/(3*len(pairs))),
            mean_xyz_magnitude_difference=sum(magnitude)/len(magnitude),
            mean_direction_degrees=sum(angles)/len(angles) if angles else None,
            direction_comparable_steps=len(angles),
            max_cumulative_xyz_command_difference=max_cumulative,
            gripper_sign_disagreement_steps=sum(grip(a[6])!=grip(b[6]) for a,b in pairs))

    result=[]
    delay=current.get('rtc_guidance',{}).get('delay_estimate_steps')
    for event in new_events:
        row=dict(kind=event['kind'],previous_prediction_id=previous['prediction_id'],
            current_prediction_id=current['prediction_id'],available_at_commit_step=commit_step,
            new_target_step=event['target'],common_future_start=start,common_future_end_exclusive=end)
        candidates=[e for e in old_events if e['kind']==event['kind']]
        if event['target']>=end or new_origin+event['confirmed_offset']>=end:
            row['status']='outside_common_future'
        elif not candidates:
            row['status']='new_event_not_in_old_forecast'
        else:
            nearest=min(candidates,key=lambda e:abs(e['target']-event['target']))
            new_count=sum(e['kind']==event['kind'] and e['target']<end
                          and new_origin+e['confirmed_offset']<end for e in new_events)
            lo=max(start,event['target']-radius);hi=min(end,event['target']+radius+1)
            row.update(status='matched' if len(candidates)==new_count==1 else 'ambiguous_match',
                old_target_step=nearest['target'],event_step_shift=event['target']-nearest['target'],
                same_kind_old_candidates=len(candidates),same_kind_new_candidates=new_count,
                window_complete=lo==event['target']-radius and hi==event['target']+radius+1,
                local_window=metrics(lo,hi),
                beyond_estimated_delay=(metrics(max(lo,new_origin+delay),hi)
                    if type(delay) is int and delay>=0 else None))
        result.append(row)
    return result


def select_open_refresh_probe(requests, commits):
    """Select one retrospective intervention, never a deployable skip rule."""
    byid={q['prediction_id']:q for q in requests}
    candidates=[]
    for old_commit,current_commit in zip(commits,commits[1:]):
        current=byid[current_commit['prediction_id']]
        following=next((q for q in requests if q['prediction_id']>current['prediction_id']),None)
        if following is None:continue
        timing=following.get('m3_timing',{})
        for event in compare_gripper_windows(byid[old_commit['prediction_id']],current,current_commit['commit_step']):
            if (event['status']=='matched' and event['kind']=='gripper_open'
                    and timing.get('reason')=='action_event'
                    and timing.get('critical_step')==event['new_target_step']
                    and 'gripper_open' in timing.get('candidate_event_kinds',[])):
                candidates.append(dict(prediction_id=current['prediction_id'],
                    observation_step=current['observation_step'],commit_step=current_commit['commit_step'],
                    target_step=event['new_target_step'],kind='gripper_open',available_features=event))
    if not candidates:
        return None
    selected=min(candidates,key=lambda x:x['available_features']['local_window']['mean_direction_degrees']
                 if x['available_features']['local_window']['mean_direction_degrees'] is not None else float('inf'))
    return dict(selected,scope='Retrospective single-event intervention selected from this resident baseline; not an online classifier')


def audit_open_probe_prefix(reference_run, candidate_run, plan):
    """Reject timing/input divergence before the planned intervention."""
    import numpy as np
    runs=[Path(reference_run),Path(candidate_run)]
    requests=[json.loads((p/'rtc/q5_requests.json').read_text()) for p in runs]
    pid=plan['prediction_id'];step=plan['commit_step']
    prefix=[[r for r in q if r['prediction_id']<=pid] for q in requests]
    checks={'same_prefix_length':len(prefix[0])==len(prefix[1])==pid+1}
    for field in ('prediction_id','observation_step','observation_raw_state','model_state','actions_raw','actions'):
        checks[field+'_equal']=len(prefix[0])==len(prefix[1]) and all(
            np.array_equal(a[field],b[field]) for a,b in zip(*prefix))
    checks['noise_equal']=len(prefix[0])==len(prefix[1]) and all(
        np.array_equal(a['rtc_solver_inputs']['noise'],b['rtc_solver_inputs']['noise']) for a,b in zip(*prefix))
    traces=[[json.loads(line) for line in (p/'rtc/trace.jsonl').open()] for p in runs]
    controls=[[e for e in trace if e['event']=='control_step' and e['step']<step] for trace in traces]
    checks['same_control_prefix']=len(controls[0])==len(controls[1])==step and all(
        a['step']==b['step'] and np.array_equal(a['action'],b['action'])
        and np.array_equal(a['before']['raw_state'],b['before']['raw_state']) for a,b in zip(*controls))
    applied=[e['diagnostic_open_probe'] for e in traces[1] if e['event']=='m3_decision' and e.get('diagnostic_open_probe')]
    checks['planned_event_applied']=bool(applied) and all(
        x['prediction_id']==pid and x['target_step']==plan['target_step'] and x['commit_step']==step
        and x['kind']=='gripper_open' for x in applied)
    return dict(passed=all(checks.values()),checks=checks,predictions_compared=len(prefix[0]),
        controls_compared=len(controls[0]),scope='Exact inputs/actions/states before one retrospective intervention; no claim of a learned or general skip policy')


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

    window_comparisons=[]
    for previous_commit,current_commit in zip(commits,commits[1:]):
        previous=dict(requests[previous_commit['prediction_id']],observation_step=previous_commit['observation_step'])
        current=dict(requests[current_commit['prediction_id']],observation_step=current_commit['observation_step'])
        window_comparisons.extend(compare_gripper_windows(previous,current,current_commit['commit_step'],config))

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
        gripper_window_comparisons=window_comparisons,
        gripper_window_scope='Post-commit diagnostics only; absolute-step aligned commands, not Cartesian positions. RTC-guided agreement does not prove future reliability or justify skipping the prediction that produced it.',
        matching='Nearest same command event within match_steps; repeated forecasts may match one event. Contact evidence is separate. Overwritten tails are not executed ground truth.',
        sources=[dict(path=str(p), sha256=_sha(p)) for p in (trace_path, requests_path, run/'run.json',
            *([summary_path] if summary_path.exists() else []))])
