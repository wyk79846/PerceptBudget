"""M3 recording and Full/Continue refresh scheduling."""
import json
import hashlib
import os
import math
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from dataclasses import asdict
from statistics import median

from perceptbudget.action_events import EventConfig, analyze_episode, detect_events, merge_nearby_slowdowns
from perceptbudget.deadline_controller import FullServiceTimes, SafetyMargin, timing_plan
from perceptbudget.experiment import _write_json_atomically


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    if settings['mode'] not in ('record_only','deadline_high','deadline_dvfs') or settings['q'] != 5:
        raise ValueError('Stage one records the unchanged q5 Full policy')
    tasks = settings['calibration_tasks']
    allowed = {'libero_spatial', 'libero_object', 'libero_goal', 'libero_10'}
    if not tasks or len({tuple(t) for t in tasks}) != len(tasks) or any(
            len(t) != 2 or t[0] not in allowed or type(t[1]) is not int or not 0 <= t[1] < 10 for t in tasks):
        raise ValueError('Use distinct suite/task pairs with task IDs 0 through 9')
    EventConfig(**settings['events'])
    return settings


def comparison_variants(config):
    if config.get('m3_formal_comparison'):
        from perceptbudget.m3_comparison import group_order
        return group_order(config.get('m3_formal_task_index',0))
    if config.get('m3_watchdog_checks'):
        return ('watchdog_late','watchdog_underflow','watchdog_hang')
    if config.get('m3_event_dedup_comparison'):
        labels=('events_and_variance__coverage_event_only',
                'events_and_variance__coverage_shared_refresh')
        return labels+('variance_only',) if 'variance_only' in (config.get('m3_trigger_variants') or ()) else labels
    if config.get('m3_static_q') is not None:
        q=config['m3_static_q']
        if q not in (5,10,20,30,40):
            raise ValueError('Use an established fixed-q baseline')
        return (f'fixed_q{q}', 'variance_only__device_dvfs')
    if config.get('m3_trigger_comparison'):
        variants=config.get('m3_trigger_variants') or ('events_only','variance_only','events_and_variance')
        repeats=config.get('m3_trigger_repeats',1)
        if (not variants or len(set(variants))!=len(variants) or
                any(v not in ('events_only','events_only_merge','variance_only','events_and_variance','events_and_variance_hold3','events_and_variance_open_probe','events_and_variance_valley','events_and_variance_latest','events_and_variance_no_turn','events_and_variance_slowdown_probe','events_and_variance_merge') for v in variants)
                or type(repeats) is not int or repeats<1):
            raise ValueError('Use distinct trigger strategies and positive repeats')
        if 'events_and_variance_hold3' in variants and (
                tuple(variants)!=('events_and_variance','events_and_variance_hold3')
                or repeats!=1 or config.get('m3_device_comparison')):
            raise ValueError('Sustained slowdown requires a single mixed/hold3 resident pair')
        if 'events_and_variance_open_probe' in variants and (
                tuple(variants)!=('events_and_variance','events_and_variance_open_probe')
                or repeats!=1 or config.get('m3_device_comparison')):
            raise ValueError('Open-event probe requires one baseline/probe resident pair')
        if 'events_and_variance_valley' in variants and (
                tuple(variants)!=('events_and_variance','events_and_variance_valley')
                or repeats!=1 or config.get('m3_device_comparison')):
            raise ValueError('Phase valleys require one baseline/candidate resident pair')
        if 'events_and_variance_latest' in variants and (
                tuple(variants) not in (('events_and_variance','events_and_variance_latest'),
                    ('variance_only','events_and_variance','events_and_variance_latest'))
                or repeats!=1 or config.get('m3_device_comparison')):
            raise ValueError('Latest-signal experiment requires the mixed pair, optionally preceded by variance_only')
        if 'events_and_variance_no_turn' in variants and (
                tuple(variants)!=('events_and_variance','events_and_variance_no_turn')
                or repeats!=1 or config.get('m3_device_comparison')):
            raise ValueError('No-turn experiment requires one baseline/candidate resident pair')
        if 'events_and_variance_slowdown_probe' in variants and (
                tuple(variants)!=('events_and_variance','events_and_variance_slowdown_probe')
                or repeats!=1 or config.get('m3_device_comparison')):
            raise ValueError('Slowdown probe requires one baseline/candidate resident pair')
        if 'events_and_variance_merge' in variants and (
                tuple(variants)!=('events_and_variance','events_and_variance_merge')
                or repeats!=1 or config.get('m3_device_comparison')):
            raise ValueError('Nearby-event merge requires one baseline/candidate resident pair')
        if 'events_only_merge' in variants and (
                tuple(variants)!=('events_only','events_only_merge')
                or repeats!=1 or config.get('m3_device_comparison')):
            raise ValueError('Event-only merge requires one baseline/candidate resident pair')
        labels=tuple(v if repeats==1 else f'{v}__repeat{r}'
                     for r in range(1,repeats+1) for v in variants)
        if (tuple(variants) in (('events_only','variance_only'),('events_only','events_and_variance')) and repeats==1
                and config.get('m3_trigger_task_index',0)%2):
            labels=labels[::-1]
        if config.get('m3_device_comparison'):
            return tuple(f'{v}__device_{device}' for v in labels for device in ('high','dvfs'))
        return labels
    if config.get('m3_dvac_horizon_comparison'):
        return ('dvac_h20','dvac_h50')
    return ('deadline_high','dvac_high')


def configure_trigger_signals(config, strategy):
    """Keep denoising-trigger ablations consistent with their candidate labels."""
    config['m3_dvac_enabled']=strategy not in ('events_only','events_only_merge')
    config['m3_action_events_enabled']=strategy!='variance_only'
    if any(s in (config.get('m3_trigger_variants') or []) for s in ('events_and_variance_merge','events_only_merge')):
        config['m3_merge_slowdown_steps']=8 if strategy in ('events_and_variance_merge','events_only_merge') else None


def configure_fixed_baseline(config, q):
    """Use the original async q counter; do not leave the deadline scheduler active."""
    if q not in (5,10,20,30,40):
        raise ValueError('Use an established fixed-q baseline')
    config.update(candidate_q=[q],m3_scheduler_mode='record_only',
        m3_record_only=True,m3_dvac_enabled=False,m3_action_events_enabled=False,
        execution_mode='m3_fixed_q_actual_async',m3_device_profile='high')
    return q


def recording_session(config, settings, directory, cleanup_gate):
    from perceptbudget.device_quality import resident_quality
    cfg = dict(config, m3_events=settings['events'])
    if settings.get('watchdog_checks'):
        cfg.update(m3_watchdog_checks=True,m3_action_events_enabled=False,
            m3_watchdog_path=str(Path(directory)/'request_watchdog.json'),m3_request_timeout_seconds=3.)
    if settings.get('formal_comparison'):
        cfg.update(m3_formal_comparison=True,m3_formal_signal=settings['formal']['trigger_strategy'],
            m3_action_events_enabled=settings['formal']['trigger_strategy']!='variance_only',
            m3_dvac_enabled=settings['formal']['trigger_strategy']!='events_only',
            m3_watchdog_path=str(Path(directory)/'request_watchdog.json'),
            m3_request_timeout_seconds=settings['formal']['request_timeout_seconds'])
        _write_json_atomically(Path(directory)/'comparison_plan.json',settings)
    if settings.get('mode','record_only') in ('deadline_high','deadline_dvfs'):
        times = FullServiceTimes.from_m2_summary(settings['timing']['source_summary'],
            probability=settings['timing']['service_quantile'])
        recovery_source=settings['timing'].get('recovery_calibration_source')
        if settings['mode']=='deadline_dvfs' and recovery_source:
            source=Path(recovery_source)
            calibrated=json.loads(source.read_text())
            if calibrated.get('status')!='completed' or not calibrated.get('samples_seconds'):
                raise ValueError('Require completed measured recovery calibration')
            for seconds in calibrated['samples_seconds']:
                times.observe('recovery',seconds)
            cfg['m3_recovery_calibration']=dict(source=str(source),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),samples=len(times.recovery_samples),
                initial_estimate_seconds=times.estimate('recovery'))
        cfg.update(m3_scheduler_mode=settings['mode'], m3_timing=settings['timing'],
                   m3_service_samples={k:list(v) for k,v in times.samples.items()},
                   m3_recovery_samples=list(times.recovery_samples))
    if settings.get('dvac_paired'):
        cfg['m3_dvac_comparison']=True
        cfg['m3_dvac_horizon_comparison']=bool(settings.get('dvac_horizon_paired'))
        cfg['m3_trigger_comparison']=bool(settings.get('trigger_comparison'))
        cfg['m3_trigger_variants']=settings.get('trigger_variants')
        cfg['m3_trigger_repeats']=settings.get('trigger_repeats',1)
        cfg['m3_device_comparison']=bool(settings.get('device_comparison'))
        cfg['m3_static_q']=settings.get('static_q')
        cfg['m3_event_dedup_comparison']=bool(settings.get('event_dedup_comparison'))
    with compiled_environments(directory, settings['calibration_tasks'], config) as cache:
        record = resident_quality(cfg, directory, 'full', cleanup_gate=cleanup_gate,
                                  task_plan=[tuple(t) for t in settings['calibration_tasks']], record_only=True)
    reports = [json.loads(Path(e['action_events']).read_text()) for e in record['episodes']]
    result = summarize(directory, reports, model_loads=record['model_loads'],
                       suffix_loads=record['suffix_loads'], episodes=record['episodes'])
    result['environment_cache'] = cache
    result['mode'] = settings.get('mode','record_only')
    result['policy_changed'] = result['mode'] != 'record_only'
    result['clocks_changed'] = result['mode'] == 'deadline_dvfs'
    if record.get('diagnostic_status')=='not_applicable':
        result.update(diagnostic_status='not_applicable',skipped_episodes=record['skipped_episodes'],
            scope='Baseline completed; no comparable target, so the intervention was not run')
    if 'paired_bootstrap_audit' in record:
        result['paired_bootstrap_audit']=record['paired_bootstrap_audit']
    if 'open_probe_prefix_audit' in record:
        result['open_probe_prefix_audit']=record['open_probe_prefix_audit']
        result['scope']='Retrospective one-event ablation, not a general online skip policy'
    if settings.get('dvac_paired'):
        result['comparison'] = list(comparison_variants(cfg))
    if settings.get('formal_comparison'):
        from perceptbudget.m3_comparison import summarize_comparison
        result['paired_bootstrap_audit']=record['paired_bootstrap_audit']
        result['formal_summary']=summarize_comparison(directory,record,screen=settings.get('comparison_screen',False))
    _write_json_atomically(Path(directory)/'summary.json', result)
    return result


class DeadlineRuntime:
    """Action hints and denoising variance select time; RTC owns alignment."""
    def __init__(self, config):
        self.settings = config['m3_timing']
        self.events_config = EventConfig(**config['m3_events'])
        self.times = FullServiceTimes(config['m3_service_samples'],
                                     probability=self.settings['service_quantile'])
        for seconds in config.get('m3_recovery_samples',[]):
            self.times.observe('recovery',seconds)
        self.margin = SafetyMargin(initial=self.settings['safety_margin_seconds'])
        self.initial_period = config['rtc_initial_control_period_seconds']
        self.horizon = config.get('H')
        self.events, self.chunks, self.plan = [], {}, None
        self.covered_events = []
        self.open_probe=config.get('m3_open_probe_plan')
        self.open_probe_applied=None
        self.event_coverage_mode=config.get('m3_event_coverage_mode','event_only')
        if self.event_coverage_mode not in ('event_only','shared_refresh'):
            raise ValueError('Unknown action-event coverage policy')
        self.mode = config['m3_scheduler_mode']
        self.dvfs_policy=config.get('m3_dvfs_policy','joint')
        if self.dvfs_policy not in ('joint','independent'):
            raise ValueError('Unknown device preparation policy')
        self.signal_policy=config.get('m3_signal_policy','earliest')
        self.latest_signal_event=None
        if self.signal_policy not in ('earliest','latest'):
            raise ValueError('Unknown signal combination policy')
        self.dvac_enabled = bool(config.get('m3_dvac_enabled'))
        self.action_events_enabled = bool(config.get('m3_action_events_enabled',True))
        self.ignore_turn = bool(config.get('m3_ignore_turn',False))
        self.merge_gap=config.get('m3_merge_slowdown_steps')
        if self.merge_gap is not None and (type(self.merge_gap) is not int or self.merge_gap<1):
            raise ValueError('Use a positive integer event gap')
        self.merged_slowdowns=[]
        self.clocks = None
        self.recovering_plan = None

    def start_clocks(self, directory):
        if self.mode == 'deadline_dvfs':
            self.clocks = RuntimeClocks(directory)
            self.clocks.__enter__()

    def stop_clocks(self):
        if self.clocks is not None:
            self.clocks.close()

    def requested(self, request):
        if request['observation_step'] and self.plan:
            request['m3_timing'] = dict(self.plan)
            request['m3_timing']['actual_request_monotonic'] = request['observation_monotonic']
            if self.recovering_plan is not None:
                request['m3_timing']['service_start_monotonic'] = self.clocks.last_change['command_monotonic']
                request['m3_timing']['history'] = 'recovery'
                request['m3_timing']['actual_low_dwell_seconds'] = self.clocks.last_change.get('low_dwell_seconds')
                self.recovering_plan = None

    def committed(self, request, commit):
        self.latest_signal_event=None
        pid, origin = request['prediction_id'], request['observation_step']
        self.chunks[pid] = dict(origin=origin, observation=request['observation_monotonic'])
        signal=request.get('dvac',{})
        prefix=signal.get('prefix_actions') if self.dvac_enabled else None
        self.chunks[pid]['dvac_step']=origin+prefix if prefix is not None else None
        self.events = [dict(e, target_step=origin+e['offset'], prediction_id=pid)
                       for e in detect_events(request['actions'], self.events_config,
                           start_offset=max(0,commit['commit_step']-origin))
                       if not (self.ignore_turn and e['kind']=='turn')] if self.action_events_enabled else []
        self.merged_slowdowns=[]
        if self.merge_gap is not None:
            self.events,self.merged_slowdowns=merge_nearby_slowdowns(self.events,self.merge_gap)
        if self.open_probe is not None and pid==self.open_probe['prediction_id']:
            plan=self.open_probe
            matches=[e for e in self.events if e['kind']==plan['kind'] and e['target_step']==plan['target_step']]
            if (self.open_probe_applied is not None or plan['kind'] not in ('gripper_open','slowdown')
                    or origin!=plan['observation_step'] or commit['commit_step']!=plan['commit_step'] or len(matches)!=1):
                raise RuntimeError('Open probe context changed before intervention; do not count as task failure')
            self.events=[e for e in self.events if e is not matches[0]]
            self.open_probe_applied={k:plan[k] for k in ('prediction_id','commit_step','target_step','kind')}
        timing = request.get('m3_timing')
        if timing:
            service_start = timing.get('service_start_monotonic',request['observation_monotonic'])
            actual = commit['commit_monotonic']-service_start
            timing['actual_service_seconds'] = actual
            timing['arrival_lateness_seconds'] = max(0.,commit['commit_monotonic']-timing['deadline'])
            critical=timing.get('critical_step')
            if self.event_coverage_mode=='event_only':
                # Frozen pre-optimization policy, retained only for paired tests.
                if (timing['reason']=='action_event' and not timing['arrival_lateness_seconds']
                        and commit['commit_step'] <= timing['critical_step']):
                    self.covered_events.extend(dict(kind=kind,target_step=timing['critical_step'])
                        for kind in timing['candidate_event_kinds'])
            elif (self.action_events_enabled and critical is not None
                    and origin <= commit['commit_step'] <= critical):
                # A variance/buffer-triggered refresh can also cover an action
                # event. Its own (earlier) deadline is not the event deadline.
                # Only match a transition still present near the old forecast;
                # changed direction or a larger shift must remain eligible.
                self.covered_events.extend(dict(kind=kind,target_step=critical)
                    for kind in timing.get('candidate_event_kinds',[])
                    if any(e['kind']==kind and abs(e['target_step']-critical)<=self.events_config.match_steps
                           for e in self.events))
            if self.settings.get('margin_updates'):
                timing['margin_update'] = self.margin.observe(predicted=timing['service_seconds'],actual=actual,
                    start_lateness=max(0.,service_start-timing['latest_start']))
            self.times.observe(timing.get('history','hot'), actual)

    def admitted(self, item, step, now):
        if item is None:
            return False
        chunk = self.chunks[item[1]]
        age_limit=self.settings['maximum_observation_age_seconds']
        return (step < chunk['origin']+self.settings['maximum_action_offset'] and
                (age_limit is None or now < chunk['observation']+age_limit))

    def decide(self, *, now, step, queue, rows, inflight, rtc_delay_steps=None):
        item = queue.get(step)
        intervals = [b['action_start_monotonic']-a['action_start_monotonic']
                     for a,b in zip(rows[-17:], rows[-16:])] if len(rows)>=17 else [
                         b['action_start_monotonic']-a['action_start_monotonic'] for a,b in zip(rows,rows[1:])]
        period = median(intervals) if intervals else self.initial_period
        end = step
        while end in queue and self.admitted(queue[end],end,now):
            end += 1
        current = self.chunks[item[1]] if item is not None else None
        tolerance = self.events_config.match_steps
        self.covered_events = [e for e in self.covered_events if e['target_step']+tolerance>=step]
        visible = [e for e in self.events if step <= e['target_step'] < end]
        candidates = [e for e in visible if not any(c['kind']==e['kind'] and
                      abs(c['target_step']-e['target_step'])<=tolerance for c in self.covered_events)]
        critical = min((e['target_step'] for e in candidates), default=None)
        critical_kinds=sorted({e['kind'] for e in candidates if e['target_step']==critical})
        if self.signal_policy=='latest':
            if self.latest_signal_event is None and critical is not None:
                self.latest_signal_event=dict(step=critical,kinds=critical_kinds)
            if self.latest_signal_event is not None:
                critical=self.latest_signal_event['step']
                critical_kinds=self.latest_signal_event['kinds']
        recovery = self.clocks is not None and (self.clocks.profile!='baseline' or self.clocks.pending is not None)
        age_limit=self.settings['maximum_observation_age_seconds']
        budget_history='recovery' if recovery and self.dvfs_policy=='joint' else 'hot'
        plan = timing_plan(now=now,step=step,buffer_end_step=end,period=period,
            critical_step=critical,
            dvac_step=current.get('dvac_step') if current else None,
            observation_deadline=current['observation']+age_limit if current and age_limit is not None else None,
            service_seconds=self.times.estimate(budget_history),margin_seconds=self.margin.seconds,
            signal_policy=self.signal_policy)
        rtc_latest_submit=None
        rtc_guard_steps=None
        rtc_latest_start=None
        if not inflight and current is not None and rtc_delay_steps is not None:
            if (type(rtc_delay_steps) is not int or self.horizon is None
                    or not 0<=rtc_delay_steps<=self.horizon):
                raise ValueError('Invalid RTC step delay or missing model horizon')
            rtc_latest_submit=current['origin']+self.horizon-rtc_delay_steps
            # A LOW->HIGH acknowledgement is consumed on a later control tick.
            # This is a submission constraint, separate from result arrival.
            rtc_guard_steps=(1 if recovery else 0)+math.ceil(self.margin.seconds/period-1e-9)
            rtc_latest_start=now+(rtc_latest_submit-step-rtc_guard_steps)*period
        self.plan = dict(asdict(plan), admitted_remaining_actions=end-step,
                         decision_monotonic=now,
                         estimated_control_period_seconds=period, critical_step=critical,
                         dvac_step=current.get('dvac_step') if current else None,
                         action_events_enabled=self.action_events_enabled, dvac_enabled=self.dvac_enabled,
                         inflight=inflight, path='low_continue_recovery' if recovery else 'high_frequency',
                         current_profile=self.clocks.profile if self.clocks else 'baseline',
                         covered_event_count=len(self.covered_events),
                         event_coverage_mode=self.event_coverage_mode,
                         signal_policy=self.signal_policy,
                         dvfs_policy=self.dvfs_policy,service_budget_history=budget_history,
                         merged_slowdowns=self.merged_slowdowns,
                         diagnostic_open_probe=self.open_probe_applied,
                         suppressed_event_count=len(visible)-len(candidates),
                         candidate_event_kinds=critical_kinds)
        self.plan.update(rtc_delay_steps=rtc_delay_steps,rtc_latest_submit_step=rtc_latest_submit,
                         rtc_submit_guard_steps=rtc_guard_steps,start_constraint='service_deadline')
        if rtc_latest_start is not None and rtc_latest_start<self.plan['latest_start']:
            self.plan.update(latest_start=rtc_latest_start,wait_seconds=max(0.,rtc_latest_start-now),
                             start_now=rtc_latest_start<=now,start_constraint='rtc_prefix')
        return self.plan

    def control_clocks(self, decision, *, inflight):
        if self.clocks is None:
            return decision
        self.clocks.poll()
        if inflight:
            if self.clocks.profile!='baseline' or self.clocks.pending is not None:
                raise RuntimeError('Frequency must stay high during Full inference')
            return decision
        if self.recovering_plan is not None:
            if self.clocks.profile=='baseline' and self.clocks.pending is None:
                self.plan = dict(self.recovering_plan,start_now=True,wait_seconds=0.,inflight=False)
                return self.plan
            return dict(decision,start_now=False,clock_wait=True)
        if self.clocks.pending is not None:
            return dict(decision,start_now=False,clock_wait=True)
        if self.clocks.profile=='gpu_emc':
            if decision['start_now']:
                self.recovering_plan = dict(decision)
                self.clocks.switch('baseline')
            return dict(decision,start_now=False,clock_wait=decision['start_now'])
        recovery_reserve=(self.times.estimate('recovery')-self.times.estimate('hot')) if self.dvfs_policy=='joint' else 0.
        low_wait = decision['latest_start']-decision['decision_monotonic']-recovery_reserve
        if not decision['start_now'] and low_wait >= self.settings['minimum_low_idle_seconds']:
            self.clocks.switch('gpu_emc')
        return decision


class RuntimeClocks:
    """Reuse the tested stdout reader; switch acknowledgements are nonblocking."""
    def __init__(self, directory):
        from perceptbudget.device_clock_screen import Lease
        class OnlineLease(Lease):
            def command(self):
                from perceptbudget.device_clock_screen import ROOT
                return ['sudo','-n','/usr/bin/python3',str(ROOT/'scripts/control_device_clocks.py'),
                        'runtime','--seconds','120']
        self.lease = OnlineLease('baseline',Path(directory))
        self.lease.duration = 120
        self.lease.path = Path(directory)/'runtime_clocks.jsonl'
        self.profile, self.pending, self.last_change = 'baseline', None, None
        self.last_low = None
        self.changes = 0

    def __enter__(self):
        self.lease.__enter__()
        return self

    def poll(self):
        import queue
        while True:
            try: event = self.lease.lines.get_nowait()
            except queue.Empty: break
            if event.get('status')=='EOF':
                raise RuntimeError('Runtime clock helper exited during task')
            if event.get('status')=='APPLIED':
                if self.pending != event['profile'] or event['request_index'] != self.changes:
                    raise RuntimeError('Runtime frequency acknowledgement mismatch')
                if event['profile']=='gpu_emc':
                    self.last_low = event['observed_match_monotonic']
                elif self.last_low is not None:
                    event['low_dwell_seconds'] = event['command_monotonic']-self.last_low
                self.profile, self.last_change, self.pending = event['profile'], event, None
                self.changes += 1
        self.lease.ensure(1.)

    def switch(self, profile):
        if profile not in ('baseline','gpu_emc') or self.pending is not None:
            raise ValueError('Use a single high/low request')
        self.lease.proc.stdin.write('HIGH\n' if profile=='baseline' else 'LOW\n')
        self.lease.proc.stdin.flush()
        self.pending = profile

    def close(self):
        import time
        until = time.monotonic()+3.
        while self.pending is not None and self.lease.proc.poll() is None:
            self.poll()
            if time.monotonic() >= until:
                self.lease.proc.stdin.close()
                self.lease.finish()
                raise TimeoutError('Clock change did not finish before task cleanup')
            time.sleep(.002)
        if self.lease.proc.stdin and not self.lease.proc.stdin.closed:
            if self.lease.proc.poll() is None:
                self.lease.proc.stdin.write('STOP\n')
                self.lease.proc.stdin.flush()
            self.lease.proc.stdin.close()
        self.lease.finish()


def clock_preflight(directory):
    """Exercise the exact new online protocol before spending time on weights."""
    import time
    clock = RuntimeClocks(directory)
    result = dict(status='running', model_loaded=False, changes=[])
    entered = False
    try:
        clock.__enter__()
        entered = True
        for profile in ('gpu_emc','baseline'):
            clock.switch(profile)
            limit = time.monotonic()+3.
            while clock.pending is not None:
                clock.poll()
                if time.monotonic() >= limit:
                    raise TimeoutError('Online clock preflight acknowledgement timed out')
                time.sleep(.002)
            result['changes'].append(dict(clock.last_change))
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            if entered:
                clock.close()
                result['restored'] = True
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            _write_json_atomically(Path(directory)/'clock_preflight.json',result)
    return result


def environment_replay(suite, tid, index):
    """Fixed controls for XML/cache parity only; never used as policy actions."""
    entry=index['cells'].get(f'{suite}_task{tid:02d}_q5_actual_history')
    if entry is None:
        if suite!='libero_10':
            raise ValueError('Missing historical preflight source for '+suite)
        actions=[[x,y,0.,0.,0.,0.,-1. if block<2 else 1.]
                 for block,(x,y) in enumerate(((.02,0.),(0.,.02),(-.02,0.),(0.,-.02)))
                 for _ in range(5)]
        return actions,'fixed20-small-motion-cache-parity'
    source=Path(entry['result']['run'])/'rtc/trace.jsonl'
    actions=[]
    with source.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('event')=='control_step':
                actions.append(row['action'])
                if len(actions)==20:break
    if len(actions)!=20:
        raise ValueError('Incomplete environment preflight controls')
    return actions,str(source)


@contextmanager
def compiled_environments(directory, tasks, config):
    """Compile and verify identical MuJoCo models before loading VLA weights.

    The cache belongs to this process/run. Every load creates fresh model/data;
    fixture handling, hard resets, camera geometry and physics remain unchanged.
    """
    import gc
    import ctypes
    import numpy as np
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import mujoco
    import robosuite.utils.binding_utils as binding
    from scripts import eval_q as base
    from scripts.complete_m1_h50 import calibrate
    from perceptbudget.device_quality import install_render_buffer_cleanup
    from perceptbudget.libero_input import raw_libero_state
    from perceptbudget.rtc_memory import memory_record
    directory = Path(directory)
    folder = directory/'environment_models'
    folder.mkdir()
    torch, _ = base._check_cuda_and_native(config)
    torch.use_deterministic_algorithms(True)
    base._patch_egl_cleanup()
    cleanup = install_render_buffer_cleanup(shared_egl=True)
    report = dict(status='preparing', compiled=0, binary_loads=0, frozen=False,
                  version=mujoco.__version__, cases=[], render_cleanup=cleanup)
    original = binding.MjSim.from_xml_string

    def factory(cls, xml):
        key = hashlib.sha256((mujoco.__version__+'\0'+xml).encode()).hexdigest()
        path = folder/(key+'.mjb')
        if path.exists():
            native = mujoco.MjModel.from_binary_path(str(path))
            report['binary_loads'] += 1
            sim = cls(native)
        elif report['frozen']:
            raise RuntimeError('Unprepared environment XML after model loading; '+key)
        else:
            sim = original(xml)
            mujoco.mj_saveModel(sim.model._model, filename=str(path))
            report['compiled'] += 1
        _write_json_atomically(directory/'environment_cache.json', report)
        return sim

    index = json.loads((Path('/home/wyk/workspace/任务产出/perceptbudget/M1')/
        '20261020T160525.143512Z-m1-h50-67126d0b/index.json').read_text())
    with patch.object(binding.MjSim, 'from_xml_string', classmethod(factory)):
        _, cls = base._libero_api()
        for suite, tid in tasks:
            task = base._load_tasks([tid], [suite])[0]
            captures = []
            replay,replay_source=environment_replay(suite,tid,index)
            for repeat in range(2):
                base._seed_process(7, np, torch)
                env = base._make_env(cls, task)
                env._pb_lock_fixtures = True
                try:
                    calibrate(env, task, np)
                    observation, _ = base._prepare_env_episode(env, task, np)
                    samples = []
                    for a in [None, *replay]:
                        if a is not None:
                            observation, _, _, _ = env.step(np.asarray(a, dtype=np.float32))
                        samples.append((env.env.sim.get_state().flatten().copy(),
                            raw_libero_state(observation).copy(),
                            np.concatenate((observation['agentview_image'], observation['robot0_eye_in_hand_image']),axis=1).copy(),
                            bool(env.check_success())))
                    captures.append(samples)
                finally:
                    env.close()
                    env = observation = None
                    gc.collect()
                    ctypes.CDLL(None).malloc_trim(0)
            physical = max(float(np.max(np.abs(a[0]-b[0]))) for a,b in zip(*captures))
            state = max(float(np.max(np.abs(a[1]-b[1]))) for a,b in zip(*captures))
            pixels = [np.abs(a[2].astype(np.int16)-b[2].astype(np.int16)) for a,b in zip(*captures)]
            case = dict(suite=suite, task_id=tid, replay_source=replay_source,
                replay_actions=replay,physical_max_abs=physical, state_max_abs=state,
                pixel_max_abs=max(int(x.max()) for x in pixels),
                max_changed_pixels=max(int(np.any(x!=0,axis=-1).sum()) for x in pixels),
                success_equal=all(a[3]==b[3] for a,b in zip(*captures)))
            case['passed'] = physical==0 and state==0 and case['success_equal'] and case['pixel_max_abs']<=1 and case['max_changed_pixels']<=4
            report['cases'].append(case)
            if not case['passed']:
                raise RuntimeError('Compiled environment replay mismatch: '+str(case))
            del captures, pixels
            print('M3_ENVIRONMENT_PREFLIGHT='+json.dumps(case), flush=True)
        report.update(status='passed', frozen=True, memory_before_weights=memory_record(torch,'compiled_environments_ready'))
        _write_json_atomically(directory/'environment_cache.json', report)
        try:
            yield report
        finally:
            _write_json_atomically(directory/'environment_cache.json', report)


def offline_review(directory, sources, settings):
    directory = Path(directory)
    reports = []
    for index, source in enumerate(sources):
        report = analyze_episode(source, EventConfig(**settings['events']))
        _write_json_atomically(directory/f'episode_{index}_events.json', report)
        reports.append(report)
    return summarize(directory, reports, model_loads=0, suffix_loads=0)


def summarize(directory, reports, *, model_loads, suffix_loads, episodes=None):
    rows = []
    for report in reports:
        rows.append(dict(run=report['run'], suite=report['suite'], task_id=report['task_id'],
            q=report['q'], control_steps=report['control_steps'],
            episode_metrics=report.get('episode_metrics',{}), **report['summary']))
    scheduled = any(r['mode'] != 'record_only' for r in reports)
    dvfs = any(r['mode'] == 'deadline_dvfs' for r in reports)
    result = dict(status='completed', stage='M3 deadline DVFS loop' if dvfs else 'M3 deadline high-frequency loop' if scheduled else 'M3 action-event recording', rows=rows,
                  model_loads=model_loads, suffix_loads=suffix_loads,
                  policy_changed=scheduled, clocks_changed=dvfs, thresholds_calibrated=False)
    if episodes is not None:
        result['episodes'] = episodes
    _write_json_atomically(Path(directory)/'summary.json', result)
    lines = ['# M3动作提示与调度记录', '',
             '当前为降频与提前恢复短测。' if dvfs else '当前为高频最晚启动短测，动作提示和缓冲决定刷新时间。' if scheduled else '当前阶段只记录和离线核对，刷新策略与频率保持原状。阈值为校准起点。', '',
             '|集合|任务|q|控制步|已提交chunk|夹爪提示|减速提示|转向提示|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        c = r['forecast_counts']
        lines.append(f'|{r["suite"]}|{r["task_id"]}|{r["q"]}|{r["control_steps"]}|{r["committed_chunks"]}|'
                     f'{c.get("gripper_open",0)+c.get("gripper_close",0)}|{c.get("slowdown",0)}|{c.get("turn",0)}|')
    lines += ['', '同一个实际事件可能被多个chunk提前提示，表中是预测提示次数，不是独立抓取次数。',
              'action_events记录每次提示的预测位置、当时可知时间、实际命令差、是否被新预测覆盖及独立接触证据。',
              '实际命令对齐与实际接触分开分析。被覆盖的预测尾段没有执行，不作为原chunk的执行真值。',
              '', f'模型加载：{model_loads}次；专家加载：{suffix_loads}次。']
    (Path(directory)/'summary.md').write_text('\n'.join(lines)+'\n')
    return result
