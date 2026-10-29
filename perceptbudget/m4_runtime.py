"""Opt-in mechanism3 controller on the established M3 event/RTC runtime."""
import json,time
from collections import deque
from perceptbudget.deadline_controller import percentile
from pathlib import Path
from perceptbudget.closed_loop import DeadlineRuntime
from perceptbudget.m4_costs import load_service_costs,load_transitions
from perceptbudget.m4_candidates import wait_model,hold_energy
from perceptbudget.m4_planner import plan


class M4Runtime(DeadlineRuntime):
    def __init__(self,config,adapter):
        super().__init__(config)
        self.adapter=adapter;self.settings4=config['m4_settings']
        self.cover_close_slowdown=bool(config.get('m4_cover_close_slowdown'))
        self.services=load_service_costs(self.settings4)
        self.waits=wait_model(json.loads(Path(self.settings4['wait_cost_source']).read_text()))
        self.transitions=load_transitions(self.settings4['directional_cost_source'])
        self.ready_histories={self.cost_key(c):deque(c.get('service_samples_seconds',[c['p95_service_seconds']]),maxlen=128) for c in self.services}
        self.calibration_ready_samples={k:list(v) for k,v in self.ready_histories.items()}
        self.end_to_end_histories={k:deque(maxlen=128) for k in self.ready_histories}
        self.poll_delays=deque(maxlen=128)
        self.active=None;self.phase=None;self.phase_end=None;self.service_begin=None
        self.consecutive_fast=0;self.quiet_since=None;self.after_state=None;self.plan_counter=0

    @staticmethod
    def cost_key(c):
        return c['mode'],c['profile'],c['from_profile'],c['history']

    def planning_services(self,period):
        # Calibration and observations end at ready; deadline and energy include
        # the control thread's harvest/commit, without changing eta.
        poll_mean=sum(self.poll_delays)/len(self.poll_delays) if self.poll_delays else period/2
        result=[]
        for c in self.services:
            if c['profile'] not in self.waits:continue
            r=dict(c)
            key=self.cost_key(c)
            # Initial ready measurements reserve one measured control period.
            # Replace this prior with whole observed prepare->commit samples;
            # never add independently measured component quantiles.
            initial=[v+period for v in self.calibration_ready_samples[key]]
            whole=(initial+list(self.end_to_end_histories[key]))[-128:]
            r['p95_service_seconds']=percentile(whole,.95)
            r['service_budget_source']='ready_plus_control_period_prior_and_observed_prepare_to_commit'
            r['mean_service_seconds']=c['mean_service_seconds']+poll_mean
            r['mean_absolute_service_energy_j']=c['mean_absolute_service_energy_j']+hold_energy(self.waits,c['profile'],0.,poll_mean)
            result.append(r)
        return result

    @staticmethod
    def trigger_key(base):
        return base.get('reason'),base.get('critical_step'),tuple(base.get('candidate_event_kinds',[]))

    def invalidate_emergency(self):
        if (self.active and self.active['selected'].get('label')=='emergency-Full-high'
                and self.active.get('trigger')!=self.trigger_key(self.base) and self.clocks.pending is None):
            self.active=None;self.phase=None;self.service_begin=None;self.phase_end=None
            return True
        return False

    def start_clocks(self,directory):
        super().start_clocks(directory)
        self.log=Path(directory)/'m4_plans.jsonl'

    def committed(self,request,commit):
        # Keep event detection, coverage accounting and RTC admission unchanged.
        # Fast/low service samples must not pollute the M3 Full/high history.
        saved={k:list(v) for k,v in self.times.samples.items()}
        saved_recovery=list(self.times.recovery_samples)
        super().committed(request,commit)
        if self.cover_close_slowdown:
            from perceptbudget.m4_event_coverage import close_slowdowns
            added=close_slowdowns(request,commit,self.events,self.covered_events,self.events_config.match_steps)
            self.covered_events.extend(added)
            request['m4_event_coverage']=added
        for k,values in saved.items():self.times.samples[k].clear();self.times.samples[k].extend(values)
        self.times.recovery_samples.clear();self.times.recovery_samples.extend(saved_recovery)
        mode=request.get('m4_service_mode','Full');profile=request.get('m4_compute_profile','baseline')
        initial=request.get('m4_entry_state','baseline')
        begin=request.get('m3_timing',{}).get('service_start_monotonic',request['observation_monotonic'])
        if request['observation_step']>0:
            elapsed=request['ready_monotonic']-begin
            if elapsed<=0:raise RuntimeError('Non-positive service timing')
            for c in self.services:
                if (c['mode'],c['profile'],c['from_profile'])==(mode,profile,initial):
                    self.ready_histories[self.cost_key(c)].append(elapsed)
                    self.end_to_end_histories[self.cost_key(c)].append(commit['commit_monotonic']-begin)
            self.poll_delays.append(max(0.,commit['commit_monotonic']-request['ready_monotonic']))
        self.consecutive_fast=self.consecutive_fast+1 if mode=='Fast' else 0
        self.quiet_since=request['ready_monotonic']
        self.after_state=self.active['selected']['next_state'] if self.active else self.clocks.profile
        self.active=None;self.phase='post' if self.after_state!=self.clocks.profile else None
        self.phase_end=None;self.service_begin=None

    def decide(self,*,now,step,queue,rows,inflight,rtc_delay_steps=None):
        base=super().decide(now=now,step=step,queue=queue,rows=rows,inflight=inflight,rtc_delay_steps=rtc_delay_steps)
        self.base=dict(base)
        if inflight:return dict(base,start_now=False,m4_phase='inflight')
        if self.phase=='post':return dict(base,start_now=False,m4_phase='post')
        self.invalidate_emergency()
        last_origin=max((c['origin'] for c in self.chunks.values()),default=0)
        if step-last_origin<self.settings.get('minimum_request_steps',5):
            return dict(base,start_now=False,m4_phase='minimum_request_interval')
        measured=self.planning_services(base['estimated_control_period_seconds'])
        if self.active is None:
            available=base['deadline']-now
            if available<=0:available=.001
            critical=(base['critical_step']-step)*base['estimated_control_period_seconds'] if base.get('critical_step') is not None else None
            # Covered events need no extra refresh, but still constrain Fast quality.
            upcoming=min((e['target_step'] for e in self.events if e['target_step']>=step),default=None)
            quality_time=(upcoming-step)*base['estimated_control_period_seconds'] if upcoming is not None else None
            full_budget=max(c['p95_service_seconds'] for c in measured if c['mode']=='Full' and c['profile']=='baseline')
            fast_permitted=quality_time is None or quality_time>full_budget+self.margin.seconds
            result=plan(self.settings4,measured,self.waits,current_profile=self.clocks.profile,
                quiet_age_seconds=max(0.,now-(self.quiet_since or now)),window_seconds=available,
                deadline_seconds=available,critical_seconds=max(0.,critical) if critical is not None else None,
                unexpected_change=bool(rows and rows[-1].get('late_hold')),consecutive_fast=self.consecutive_fast,
                transitions=self.transitions,fast_permitted=fast_permitted)
            result['quality_critical_seconds']=quality_time
            self.plan_counter+=1
            with self.log.open('a') as f:f.write(json.dumps(dict(now=now,step=step,**result))+'\n')
            if result['selected'] is None:
                # Earliest Full/high fallback; do not pretend the deadline was met.
                selected=dict(label='emergency-Full-high',service_mode='Full',compute_profile='baseline',next_state='baseline',
                    service_start_seconds=0.,prepare_start_seconds=0.,segments=[],energy_j=None,service_p95_seconds=self.times.estimate('recovery' if self.clocks.profile!='baseline' else 'hot'))
            else:selected=result['selected']
            self.active=dict(selected=selected,origin=now,initial=self.clocks.profile,
                prepare_at=min(now+selected['prepare_start_seconds'],base['latest_start']-
                    max(0.,selected['service_start_seconds']-selected['prepare_start_seconds'])),deadline=base['deadline'],trigger=self.trigger_key(base))
            self.phase='waiting'
        # A tightening RTC/event limit may advance, never postpone, the plan.
        entry_seconds=max(0.,self.active['selected']['service_start_seconds']-self.active['selected']['prepare_start_seconds'])
        self.active['prepare_at']=min(self.active['prepare_at'],base['latest_start']-entry_seconds)
        chosen=self.active['selected']
        return dict(base,start_now=self.phase=='ready',m4_phase=self.phase,m4_label=chosen['label'],
            m4_service_mode=chosen['service_mode'],m4_compute_profile=chosen['compute_profile'],
            m4_next_state=chosen['next_state'],m4_predicted_energy_j=chosen['energy_j'])

    def control_clocks(self,decision,*,inflight):
        self.clocks.poll();now=time.monotonic()
        if self.clocks.last_change:
            self.quiet_since=max(self.quiet_since or 0.,self.clocks.last_change['observed_match_monotonic'])
        if inflight:
            if self.clocks.pending is not None:raise RuntimeError('Clock change during active inference')
            return decision
        if self.invalidate_emergency():return dict(decision,start_now=False,m4_phase='emergency_replan')
        if decision.get('m4_phase')=='minimum_request_interval':return decision
        if self.phase=='post':
            # Do not enter a low state when another request is already urgent.
            target=self.after_state if self.base['latest_start']-now>=.25 else self.clocks.profile
            if target!=self.clocks.profile and self.clocks.pending is None:
                self.clocks.switch(target);self.phase_end=now+.25
            if self.base['latest_start']<=now:self.phase_end=None
            if self.clocks.pending is None and (self.phase_end is None or now>=self.phase_end):
                self.phase=None;self.phase_end=None
            return dict(decision,start_now=False,m4_phase='post')
        if self.active is None:return dict(decision,start_now=False)
        chosen=self.active['selected'];segments=chosen['segments']
        first=next((s for s in segments if s['seconds']>1e-8),None)
        if self.phase=='waiting':
            # Continue plans may switch the waiting state before deferred Full.
            if first and first['kind']=='transition' and self.clocks.pending is None and self.clocks.profile!=first['target']:
                self.clocks.switch(first['target'])
            if now<self.active['prepare_at'] or self.clocks.pending is not None:return dict(decision,start_now=False)
            target=chosen['compute_profile']
            self.entry_state=self.clocks.profile
            entry_seconds=max(0.,chosen['service_start_seconds']-chosen['prepare_start_seconds'])
            # External preparation is a separate segment. Only a frequency
            # recovery INCLUDED in service starts its timer before observation.
            included_recovery=self.clocks.profile!=target and entry_seconds==0
            self.service_begin=now if included_recovery else None
            if entry_seconds:self.entry_state=target
            if self.clocks.profile!=target:self.clocks.switch(target)
            # Preparation has an absolute planned end. A late control-thread
            # check must not restart the whole preparation interval. This also
            # serves plans that already downclocked at the start of the wait.
            self.phase_end=(min(self.active['origin']+chosen['service_start_seconds'],
                                self.active['prepare_at']+entry_seconds) if entry_seconds else now)
            self.phase='preparing'
        if self.phase=='preparing' and self.clocks.pending is None and now>=self.phase_end:
            self.adapter.select_next_mode(chosen['service_mode']);self.phase='ready'
        return dict(decision,start_now=self.phase=='ready',m4_phase=self.phase)

    def requested(self,request):
        if request['observation_step']==0:
            self.adapter.select_next_mode('Full');request.update(m4_service_mode='Full',m4_compute_profile='baseline',m4_entry_state='baseline')
            return
        super().requested(request)
        chosen=self.active['selected']
        critical=min((e['target_step'] for e in self.events if e['target_step']>=request['observation_step']),default=None)
        seconds=(critical-request['observation_step'])*self.plan['estimated_control_period_seconds'] if critical is not None else float('inf')
        measured=self.planning_services(self.plan['estimated_control_period_seconds'])
        full_budget=max(c['p95_service_seconds'] for c in measured if c['mode']=='Full' and c['profile']=='baseline')
        if chosen['service_mode']=='Fast' and seconds<=full_budget+self.margin.seconds:
            chosen['service_mode']='Full'
            chosen['service_p95_seconds']=max(c['p95_service_seconds'] for c in measured if c['mode']=='Full' and c['profile']==chosen['compute_profile'])
            self.adapter.select_next_mode('Full')
            request['m4_quality_override']='critical_event_near_at_submission'
        request.update(m4_service_mode=chosen['service_mode'],m4_compute_profile=chosen['compute_profile'],m4_entry_state=self.entry_state)
        request['m3_timing'].update(service_start_monotonic=self.service_begin if self.service_begin is not None else request['observation_monotonic'],service_seconds=chosen['service_p95_seconds'],
            m4_label=chosen['label'],m4_predicted_energy_j=chosen['energy_j'])
        self.phase='inflight'
