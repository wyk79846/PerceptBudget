"""Opt-in mechanism3 controller on the established M3 event/RTC runtime."""
import json,time
from pathlib import Path
from perceptbudget.closed_loop import DeadlineRuntime
from perceptbudget.m4_costs import load_service_costs,load_transitions
from perceptbudget.m4_candidates import wait_model
from perceptbudget.m4_planner import plan


class M4Runtime(DeadlineRuntime):
    def __init__(self,config,adapter):
        super().__init__(config)
        self.adapter=adapter;self.settings4=config['m4_settings']
        self.services=load_service_costs(self.settings4)
        self.waits=wait_model(json.loads(Path(self.settings4['wait_cost_source']).read_text()))
        self.transitions=load_transitions(self.settings4['directional_cost_source'])
        self.active=None;self.phase=None;self.phase_end=None;self.service_begin=None
        self.consecutive_fast=0;self.quiet_since=None;self.after_state=None;self.plan_counter=0

    def start_clocks(self,directory):
        super().start_clocks(directory)
        self.log=Path(directory)/'m4_plans.jsonl'

    def committed(self,request,commit):
        # Keep event detection, coverage accounting and RTC admission unchanged.
        # Fast/low service samples must not pollute the M3 Full/high history.
        saved={k:list(v) for k,v in self.times.samples.items()}
        saved_recovery=list(self.times.recovery_samples)
        super().committed(request,commit)
        for k,values in saved.items():self.times.samples[k].clear();self.times.samples[k].extend(values)
        self.times.recovery_samples.clear();self.times.recovery_samples.extend(saved_recovery)
        mode=request.get('m4_service_mode','Full');profile=request.get('m4_compute_profile','baseline')
        initial=request.get('m4_entry_state','baseline')
        begin=request.get('m3_timing',{}).get('service_start_monotonic',request['observation_monotonic'])
        actual=request['ready_monotonic']-begin
        for c in self.services:
            if (c['mode'],c['profile'],c['from_profile'])==(mode,profile,initial):
                # Never shrink measured tail budgets from a handful of online calls.
                c['p95_service_seconds']=max(c['p95_service_seconds'],actual)
        self.consecutive_fast=self.consecutive_fast+1 if mode=='Fast' else 0
        self.quiet_since=request['ready_monotonic']
        self.after_state=self.active['selected']['next_state'] if self.active else self.clocks.profile
        self.active=None;self.phase='post';self.phase_end=None;self.service_begin=None

    def decide(self,*,now,step,queue,rows,inflight,rtc_delay_steps=None):
        base=super().decide(now=now,step=step,queue=queue,rows=rows,inflight=inflight,rtc_delay_steps=rtc_delay_steps)
        self.base=dict(base)
        if inflight:return dict(base,start_now=False,m4_phase='inflight')
        if self.phase=='post':return dict(base,start_now=False,m4_phase='post')
        if self.active is None:
            available=base['deadline']-now
            if available<=0:available=.001
            critical=(base['critical_step']-step)*base['estimated_control_period_seconds'] if base.get('critical_step') is not None else None
            result=plan(self.settings4,self.services,self.waits,current_profile=self.clocks.profile,
                quiet_age_seconds=max(0.,now-(self.quiet_since or now)),window_seconds=available,
                deadline_seconds=available,critical_seconds=max(0.,critical) if critical is not None else None,
                unexpected_change=bool(rows and rows[-1].get('late_hold')),consecutive_fast=self.consecutive_fast,
                transitions=self.transitions)
            self.plan_counter+=1
            with self.log.open('a') as f:f.write(json.dumps(dict(now=now,step=step,**result))+'\n')
            if result['selected'] is None:
                # Earliest Full/high fallback; do not pretend the deadline was met.
                selected=dict(label='emergency-Full-high',service_mode='Full',compute_profile='baseline',next_state='baseline',
                    service_start_seconds=0.,prepare_start_seconds=0.,segments=[],energy_j=None,service_p95_seconds=self.times.estimate('recovery' if self.clocks.profile!='baseline' else 'hot'))
            else:selected=result['selected']
            self.active=dict(selected=selected,origin=now,initial=self.clocks.profile,
                prepare_at=min(now+selected['prepare_start_seconds'],base['latest_start']-
                    max(0.,selected['service_start_seconds']-selected['prepare_start_seconds'])),deadline=base['deadline'])
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
            self.service_begin=now+entry_seconds
            if entry_seconds:self.entry_state=target
            if self.clocks.profile!=target:self.clocks.switch(target)
            self.phase_end=now+entry_seconds
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
        critical=self.plan.get('critical_step')
        seconds=(critical-request['observation_step'])*self.plan['estimated_control_period_seconds'] if critical is not None else float('inf')
        full_budget=max(c['p95_service_seconds'] for c in self.services if c['mode']=='Full' and c['profile']=='baseline')
        if chosen['service_mode']=='Fast' and seconds<=full_budget+self.margin.seconds:
            chosen['service_mode']='Full'
            chosen['service_p95_seconds']=max(c['p95_service_seconds'] for c in self.services if c['mode']=='Full' and c['profile']==chosen['compute_profile'])
            self.adapter.select_next_mode('Full')
            request['m4_quality_override']='critical_event_near_at_submission'
        request.update(m4_service_mode=chosen['service_mode'],m4_compute_profile=chosen['compute_profile'],m4_entry_state=self.entry_state)
        request['m3_timing'].update(service_start_monotonic=self.service_begin,service_seconds=chosen['service_p95_seconds'],
            m4_label=chosen['label'],m4_predicted_energy_j=chosen['energy_j'])
        self.phase='inflight'
