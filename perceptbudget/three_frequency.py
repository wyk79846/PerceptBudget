"""Opt-in medium-first Full, high fallback,306MHz Continue; original baseline retained."""
import json
import math
import time
from pathlib import Path

from perceptbudget.closed_loop import DeadlineRuntime,RuntimeClocks
from perceptbudget.deadline_controller import percentile
from perceptbudget.m3_low_full import LowFullRuntime


def enable_profiles():
    from perceptbudget.device_clocks import PROFILES
    PROFILES['deep_idle']=(306000000,2133000000,1497600)


def transition_costs(output_root):
    root=Path(output_root)/'M4'
    medium=root/'20261102T122709.206402Z-gpu-clock-switch-6a72818c/summary.json'
    high=root/'20261102T121105.707481Z-continue-lower-combos-a777f723/preflight.json'
    m=json.loads(medium.read_text());h=json.loads(high.read_text())
    if m['status']!='completed' or not m['restored'] or m['upper_gpu_hz']!=816000000:
        raise ValueError('Missing verified low-medium transition measurements')
    # The parent run failed later on input pairing, but this root transition
    # stage completed and restored. Verify its raw receipt independently.
    log=high.parent/'idle306_switch.log'
    lines=log.read_text().splitlines()
    events=[json.loads(line) for line in lines if line.startswith('{')]
    if 'RESTORED' not in lines or sum(e.get('status')=='TRANSITION' for e in events)!=10:
        raise ValueError('High transition receipt incomplete')
    return dict(medium_seconds=m['conditions']['up']['max_ms']/1000,
        high_seconds=h['timings']['idle306']['up']['max_ms']/1000,
        high_to_medium_seconds=h['timings']['idle306']['down']['max_ms']/1000,
        medium_source=str(medium),high_source=str(high),
        scope='Measured static readback maxima; control-tick acknowledgement separately reserved')


def select_full(decision, *, medium_seconds, medium_cap, current_profile, transitions):
    """Always test medium feasibility first; energy is not a selection criterion."""
    now=decision['decision_monotonic']
    end=decision['deadline']-decision['margin_seconds']
    ack=decision['estimated_control_period_seconds']
    first=decision.get('first_refresh_cap')
    rows=[]
    # Compute budgets already include one normal commit/control tick. Reserve
    # one additional tick only when waiting for a clock acknowledgement.
    for profile,compute,cap in [('gpu_emc',medium_seconds,medium_cap),
                                ('baseline',decision['service_seconds'],decision['latest_start'])]:
        change=current_profile!=profile
        switch=0. if not change else transitions['high_to_medium_seconds'] if profile=='gpu_emc' and current_profile=='baseline' else transitions['medium_seconds' if profile=='gpu_emc' else 'high_seconds']
        # Existing recovery Full history includes the old high frequency change.
        # Do not add that frequency cost a second time.
        extra=(switch+ack) if profile=='gpu_emc' and change else 0.
        service=compute+extra
        first_cap=(first['latest_start']+(first.get('recovery_ticks',0)-int(change))*ack) if first else end
        latest=min(end-service,cap,first_cap)
        rows.append(dict(profile=profile,service_seconds=service,compute_seconds=compute,
            switch_seconds=switch,ack_reserve_seconds=ack if change else 0.,
            latest_start=latest,rtc_cap=cap,first_cap=first_cap,feasible=latest>=now-1e-9))
    selected=rows[0] if rows[0]['feasible'] else rows[1]
    return dict(policy='medium_first_v1',selected=selected,candidates=rows,
        selection_order=['gpu_emc','baseline'],
        reason='medium_meets_deadline' if selected is rows[0] else
               'high_medium_too_late' if rows[1]['feasible'] else 'high_emergency',
        deadline=decision['deadline'],energy_used_for_selection=False)


def start_point(step, now, latest_start, period, limit_seconds):
    """Round the startup point, never the duration of the required computation."""
    if not all(math.isfinite(x) for x in (now,latest_start,period,limit_seconds)) or period<=0:
        raise ValueError('Invalid startup timing')
    if limit_seconds not in (0.,.025):raise ValueError('Startup floor supports0 or25ms')
    remaining=latest_start-now
    fractional=step+remaining/period
    floor_allowed=remaining<=limit_seconds+1e-9
    return dict(decision_step=step,continuous_start_step=fractional,
        advance_seconds=max(0.,remaining),limit_seconds=limit_seconds,
        floor_allowed=floor_allowed,overdue=remaining<0,
        requested_rule='start_point_floor_within_limit')


def critical_high(formula, decision):
    """Speed up event refreshes without spending the gain on a later start."""
    if decision['reason']!='action_event':return formula
    original=formula['selected']
    high=next(row for row in formula['candidates'] if row['profile']=='baseline')
    return dict(formula,policy='critical_event_high_v1',budget_policy=formula['policy'],
        baseline_selected=original,selected=high,reason='critical_event_high',
        prepare_anchor=dict(profile=original['profile'],latest_start=original['latest_start']))


def advance_grasp_deadline(decision, *, step, lead_steps):
    """Reserve action correction ticks before closure, separately from service.

    Keep the physical critical step for RTC/wait/phase coverage. The new
    deadline only asks the refresh to finish earlier; model latency and eta
    are unchanged. No task, seed or object identity enters this rule.
    """
    if (not lead_steps or decision.get('grasp_lead') or
            decision.get('reason')!='action_event' or
            'gripper_close' not in decision.get('candidate_event_kinds',[])):
        return decision
    period=decision['estimated_control_period_seconds']
    target=decision['critical_step']-lead_steps
    deadline=min(decision['deadline'],decision['decision_monotonic']+(target-step)*period)
    latest=min(decision['latest_start'],deadline-decision['service_seconds']-decision['margin_seconds'])
    return dict(decision,deadline=deadline,latest_start=latest,
        wait_seconds=max(0.,latest-decision['decision_monotonic']),
        start_now=latest<=decision['decision_monotonic'],
        deadline_already_unreachable=decision['decision_monotonic']+decision['service_seconds']+decision['margin_seconds']>deadline,
        grasp_lead=dict(policy='gripper_close_correction_lead_v1',lead_steps=lead_steps,
            physical_critical_step=decision['critical_step'],refresh_target_step=target,
            lead_seconds=lead_steps*period,physical_deadline=decision['deadline']))


def wait_for_clock(clocks, limit_seconds):
    """Bounded frequency acknowledgement in this control tick; no env.step."""
    if limit_seconds!=.025:raise ValueError('Startup acknowledgement wait must be25ms')
    start=time.monotonic();end=start+limit_seconds
    while True:
        clocks.poll()
        now=time.monotonic()
        if clocks.pending is None or now>=end:break
        time.sleep(min(.001,end-now))
    return dict(start_monotonic=start,stop_monotonic=now,seconds=now-start,
        limit_seconds=limit_seconds,acknowledged=clocks.pending is None,
        simulation_frozen=True,queue_consumed=False)


class ThreeClocks(RuntimeClocks):
    def __init__(self,directory):
        enable_profiles()
        super().__init__(directory)
        from perceptbudget.clock_access import helper_command
        self.lease.command=lambda: helper_command('runtime3','--seconds','120')

    def switch(self,profile):
        tokens={'baseline':'HIGH','gpu_emc':'MID','deep_idle':'IDLE'}
        if profile not in tokens or self.pending is not None:
            raise ValueError('Only one outstanding three-state switch allowed')
        self.lease.proc.stdin.write(tokens[profile]+'\n')
        self.lease.proc.stdin.flush()
        self.pending=profile


class ThreeFullRuntime(LowFullRuntime):
    def __init__(self,config):
        super().__init__(config)
        self.ordinary_fast=bool(config.get('m3_ordinary_fast'))
        self.ordinary_costs=config.get('ordinary_fast_costs')
        self.ordinary_policy=config.get('m3_ordinary_fast_policy','ordinary_fast_same_full_schedule_v1')
        self.ordinary_previous='Full'
        self.ordinary_full_streak=1
        self.ordinary_full_per_fast=config.get('m3_fast_full_per_fast',1)
        self.fast_delays={}
        self.fast_own_budget=bool(config.get('m3_fast_own_budget'))
        self.fast_every_event=bool(config.get('m3_fast_every_event'))
        self.critical_high=bool(config.get('m3_critical_high'))
        self.grasp_lead_steps=config.get('m3_grasp_lead_steps',0)
        self.full_energy=bool(config.get('m3_full_energy'))
        self.energy_costs=config.get('m3_full_energy_costs')
        self.last_ready=None
        if self.full_energy and self.ordinary_fast:raise ValueError('Compare Full energy without Fast')
        self.transitions=config['m3_three_frequency_transitions']
        self.start_floor_seconds=config.get('m3_start_floor_ms',0)/1000
        if self.start_floor_seconds not in (0.,.025):raise ValueError('Startup floor supports0 or25ms')
        self.recovery_target=None
        self.safe_response_enabled=bool(config.get('safe_failure_probe') and
            config.get('safe_probe_mode','refresh')=='refresh')
        self.safe_pending=None
        self.visual_response_enabled=config.get('visual_monitor',{}).get('mode')=='refresh'
        self.visual_pending=None

    def start_clocks(self,directory):
        self.action_signal_directory=Path(directory)
        self.clocks=ThreeClocks(directory)
        self.clocks.__enter__()

    def decide(self,*,now,step,queue,rows,inflight,rtc_delay_steps=None):
        self.ordinary_step=step
        item_now=queue.get(step)
        self.ordinary_phases=self.chunks[item_now[1]].get('low_full_phases',[]) if item_now else []
        if self.fast_own_budget:
            self.clocks.poll()
            if self.clocks.pending is not None and self.recovering_plan is None:
                # A pending Continue transition must not resurrect the base
                # Full-only start time while Fast is deliberately waiting.
                return dict(self.plan,start_now=False,clock_wait=True)
        d=DeadlineRuntime.decide(self,now=now,step=step,queue=queue,rows=rows,
                                 inflight=inflight,rtc_delay_steps=rtc_delay_steps)
        high_history=d['service_budget_history']
        if inflight or self.recovering_plan is not None or self.clocks.pending is not None:
            return d
        item=queue.get(step)
        if item is None:return d
        if self.safe_pending is not None:
            from perceptbudget.safe_runtime import cap_refresh
            d=cap_refresh(d,self.safe_pending,now=now,step=step,current_profile=self.clocks.profile)
        if self.visual_pending is not None:
            from perceptbudget.visual_monitor import cap_visual_refresh
            d=cap_visual_refresh(d,self.visual_pending,now=now,step=step,current_profile=self.clocks.profile)
        d=advance_grasp_deadline(d,step=step,lead_steps=self.grasp_lead_steps)
        period=d['estimated_control_period_seconds']
        guard=math.ceil(self.margin.seconds/period-1e-9)+(self.clocks.profile!='gpu_emc')
        submit=self.chunks[item[1]]['origin']+self.horizon-self.low_delay_steps
        cap=now+(submit-step-guard)*period
        medium=max(percentile(list(v),self.settings['service_quantile']) for v in self.low_histories.values())
        formula=select_full(d,medium_seconds=medium,medium_cap=cap,
                            current_profile=self.clocks.profile,transitions=self.transitions)
        if self.fast_own_budget and d['reason']=='action_event' and not d.get('safe_refresh') and not d.get('visual_refresh'):
            from perceptbudget.ordinary_fast import choose
            from perceptbudget.fast_start_budget import select
            choices={}
            for proposal in formula['candidates']:
                profile=proposal['profile']
                trial=dict(d,compute_profile=profile,latest_start=proposal['latest_start'],
                    service_seconds=proposal['service_seconds'],
                    rtc_delay_steps=self.low_delay_steps if profile=='gpu_emc' else rtc_delay_steps)
                choices[profile]=choose(trial,step=step,previous_mode=self.ordinary_previous,
                    phases=self.ordinary_phases,events=self.events,costs=self.ordinary_costs,
                    fast_delay=self.fast_delays[profile],policy=self.ordinary_policy,
                    match_steps=self.events_config.match_steps,
                    full_per_fast=self.ordinary_full_per_fast,full_streak=self.ordinary_full_streak,
                    every_event=self.fast_every_event)
            formula=select(formula,d,choices=choices,fast_delays=self.fast_delays,
                costs=self.ordinary_costs,probability=self.settings['service_quantile'],
                prefix_caps=dict(gpu_emc=cap,baseline=now+(
                    d['rtc_latest_submit_step']-step-d['rtc_submit_guard_steps'])*period
                    if d.get('rtc_latest_submit_step') is not None else float('inf')))
        if self.critical_high:formula=critical_high(formula,d)
        if d.get('critical_freshness'):
            # An extra observation uses its own chosen HIGH mode's budget;
            # the normal policy retains its original medium prepare anchor.
            formula.pop('prepare_anchor',None)
            formula['supplement_prepare_rule']='selected_mode_service_budget'
        chosen=formula['selected']
        anchor=formula.get('prepare_anchor')
        latest=min(chosen['latest_start'],anchor['latest_start']) if anchor else chosen['latest_start']
        # Preserve LowFullRuntime's separate medium measurement history and RTC
        # queue; it uses the old low_full_formula selected.history field.
        age=max(0.,now-(self.clocks.last_low or now))+max(0.,latest-now)
        history='hot' if age<.2 else 'short' if age<1. else 'long'
        medium_chosen=chosen['profile']=='gpu_emc'
        due=latest-now<period-1e-9
        startup=None
        if self.start_floor_seconds:
            startup=start_point(step,now,latest,period,self.start_floor_seconds)
            # When a clock change is needed, start its preparation in the
            # existing window. Only submit this tick after a timely real ack.
            # At an already-ready frequency, bound the early submission itself.
            if self.clocks.profile==chosen['profile']:due=startup['floor_allowed']
        if d.get('first_refresh_cap'):
            d['first_refresh_cap']=dict(d['first_refresh_cap'],latest_start=chosen['first_cap'],
                recovery_ticks=int(self.clocks.profile!=chosen['profile']))
        expected_start=d['deadline']-chosen['service_seconds']-self.margin.seconds
        constraint='service_deadline'
        chosen_cap=cap if medium_chosen else now+(d['rtc_latest_submit_step']-step-d['rtc_submit_guard_steps'])*period if d.get('rtc_latest_submit_step') is not None else float('inf')
        if chosen_cap<expected_start:expected_start=chosen_cap;constraint='rtc_prefix'
        first=d.get('first_refresh_cap',{}).get('latest_start',float('inf'))
        if first<expected_start:constraint='first_refresh'
        if anchor:
            d['critical_high_prepare_anchor']=anchor
            if anchor['latest_start']<chosen['latest_start']-1e-9:constraint='critical_event_fixed_prepare'
        d.update(three_frequency_formula=formula,start_constraint=constraint,
            low_full_formula=dict(selected=dict(chosen,history=history),policy='medium_first_v1'),
            compute_profile=chosen['profile'],service_budget_history='low' if medium_chosen else d['service_budget_history'],
            service_seconds=chosen['service_seconds'],
            latest_start=latest,wait_seconds=0. if due else max(0.,latest-now),
            start_now=due,path='three_frequency',
            rtc_delay_steps=self.low_delay_steps if medium_chosen else rtc_delay_steps,
            rtc_latest_submit_step=submit if medium_chosen else d['rtc_latest_submit_step'],
            rtc_submit_guard_steps=guard if medium_chosen else d['rtc_submit_guard_steps'],
            deadline_already_unreachable=now+chosen['service_seconds']+self.margin.seconds>d['deadline'])
        if startup is not None:d['start_point_rounding']=startup
        budget_policy=formula.get('budget_policy',formula['policy'])
        if budget_policy=='critical_fast_own_budget_v1':
            d['fast_start_budget']=dict(policy=budget_policy,mode=chosen['mode'],
                source='Fast mode/profile observed delay plus commit tick and measured prior',
                original_full=next(r for r in formula['full_candidates'] if r['profile']==chosen['profile']))
        if self.full_energy and due:
            from perceptbudget.full_energy import select,POLICY
            anchor=dict(profile=chosen['profile'],latest_start=d['latest_start'],
                        start_constraint=d['start_constraint'])
            energy=select(formula,d,self.energy_costs,
                quiet_age=max(0.,now-(self.last_ready if self.last_ready is not None else now)),
                minimum_idle=self.settings['minimum_low_idle_seconds'])
            new=next(r for r in formula['candidates'] if r['profile']==energy['profile'])
            medium_selected=new['profile']=='gpu_emc'
            delay=self.low_delay_steps if medium_selected else rtc_delay_steps
            new_guard=math.ceil(self.margin.seconds/period-1e-9)+(self.clocks.profile!=new['profile'])
            new_submit=self.chunks[item[1]]['origin']+self.horizon-delay
            latest=min(anchor['latest_start'],new['latest_start'])
            formula=dict(formula,policy=POLICY,baseline_selected=chosen,selected=new,
                reason=energy['reason'],energy_used_for_selection=energy['energy_used_for_selection'],energy=energy)
            d.update(three_frequency_formula=formula,energy_prepare_anchor=anchor,
                compute_profile=new['profile'],service_seconds=new['service_seconds'],
                service_budget_history='low' if medium_selected else high_history,
                latest_start=latest,start_now=True,wait_seconds=0.,
                rtc_delay_steps=delay,rtc_latest_submit_step=new_submit,rtc_submit_guard_steps=new_guard,
                deadline_already_unreachable=now+new['service_seconds']+self.margin.seconds>d['deadline'])
            if d.get('first_refresh_cap'):
                d['first_refresh_cap']=dict(d['first_refresh_cap'],latest_start=new['first_cap'],
                    recovery_ticks=int(self.clocks.profile!=new['profile']))
            own=d['deadline']-new['service_seconds']-self.margin.seconds
            prefix_cap=now+(new_submit-step-new_guard)*period
            constraint='rtc_prefix' if prefix_cap<own else 'service_deadline'
            if d.get('first_refresh_cap',{}).get('latest_start',math.inf)<min(own,prefix_cap):constraint='first_refresh'
            d['start_constraint']='full_energy_fixed_prepare' if anchor['latest_start']<new['latest_start']-1e-9 else constraint
            if startup is not None:d['start_point_rounding']=start_point(step,now,latest,period,self.start_floor_seconds)
        if self.ordinary_fast:
            from perceptbudget.ordinary_fast import choose
            d['m3_fast_formula']=choose(d,step=step,previous_mode=self.ordinary_previous,
                phases=self.chunks[item[1]].get('low_full_phases',[]),events=self.events,
                costs=self.ordinary_costs,fast_delay=self.fast_delays[chosen['profile']],
                policy=self.ordinary_policy,match_steps=self.events_config.match_steps,
                full_per_fast=self.ordinary_full_per_fast,full_streak=self.ordinary_full_streak,
                every_event=self.fast_every_event)
        if d.get('critical_freshness'):
            proof=d['critical_freshness']
            self._skip_critical_freshness=True
            try:
                baseline=self.decide(now=now,step=step,queue=queue,rows=rows,
                    inflight=inflight,rtc_delay_steps=rtc_delay_steps)
            finally:
                self._skip_critical_freshness=False
            event_time=now+(proof['target_step']-step)*period
            baseline_start=max(now,baseline['latest_start'])
            capture=step+max(0.,baseline_start-now)/period
            already_timely=(proof['target_step']-capture<=self.critical_freshness_age
                and baseline_start+baseline['service_seconds']+self.margin.seconds<=event_time+1e-9)
            feasible=(d['reason']=='action_event' and d['latest_start']>=now-1e-9
                and not d['deadline_already_unreachable'])
            estimated_wait=max(0.,now+d['service_seconds']+self.margin.seconds-d['deadline'])
            wait_budget=self.critical_freshness_wait_steps*period
            prefix_ready=(d.get('rtc_latest_submit_step') is None or
                step+d['rtc_submit_guard_steps']<=d['rtc_latest_submit_step'])
            short_wait=(self.critical_freshness_wait_steps>0
                and d['reason']=='action_event' and step<proof['target_step']
                and prefix_ready and 0.<estimated_wait<=wait_budget+1e-9)
            feasible=feasible or short_wait
            if already_timely or not feasible:
                baseline['critical_freshness_skipped']=dict(proof,
                    reason='existing_refresh_timely' if already_timely else 'selected_service_too_late',
                    selected_mode=d.get('m3_fast_formula',{}).get('mode','Full'),
                    selected_service_seconds=d['service_seconds'],
                    estimated_wait_seconds=estimated_wait,maximum_estimated_wait_seconds=wait_budget,
                    rtc_prefix_available=prefix_ready)
                self.plan=baseline
                return baseline
            proof.update(selected_mode=d.get('m3_fast_formula',{}).get('mode','Full'),
                selected_service_seconds=d['service_seconds'],latest_start=d['latest_start'],
                baseline_latest_start=baseline['latest_start'],baseline_service_seconds=baseline['service_seconds'])
            if short_wait:
                proof.update(maximum_wait_steps=self.critical_freshness_wait_steps,
                    maximum_estimated_wait_seconds=wait_budget,
                    estimated_wait_seconds=estimated_wait,
                    wait_policy='critical_freshness_short_wait_v1')
            if self.critical_freshness_correction_steps and proof['kind']=='gripper_close':
                pause=max(step,proof['target_step']-self.critical_freshness_correction_steps)
                proof.update(pause_policy='pre_grasp_correction_v1',pause_step=pause,
                    correction_steps=self.critical_freshness_correction_steps,
                    estimated_pause_seconds=max(0.,now+d['service_seconds']+self.margin.seconds
                        -(now+(pause-step)*period)))
        self.plan=d
        return d

    def committed(self,request,commit):
        # Fast may change future action events, but cannot shorten Full budgets.
        fast=request.get('m3_fast_choice',{}).get('mode')=='Fast'
        if fast:
            samples={k:list(v) for k,v in self.times.samples.items()}
            recovery=list(self.times.recovery_samples)
            low={k:list(v) for k,v in self.low_histories.items()}
        super().committed(request,commit)
        if fast:
            for k,v in samples.items():self.times.samples[k].clear();self.times.samples[k].extend(v)
            self.times.recovery_samples.clear();self.times.recovery_samples.extend(recovery)
            for k,v in low.items():self.low_histories[k].clear();self.low_histories[k].extend(v)
        self.ordinary_previous=request.get('m3_fast_choice',{}).get('mode','Full')
        self.ordinary_full_streak=(0 if fast else 1 if request['prediction_id']==0 else self.ordinary_full_streak+1)
        self.last_ready=request.get('ready_monotonic',commit['commit_monotonic'])
        if self.safe_response_enabled and request.get('safe_failure_probe',{}).get('new_alarm'):
            self.safe_pending=dict(prediction_id=request['prediction_id'],
                commit_step=commit['commit_step'],request_step=commit['commit_step']+1,
                score=request['safe_failure_probe']['cumulative_score'],
                threshold=request['safe_failure_probe']['threshold'])

    def control_clocks(self,decision,*,inflight):
        decision=self._control_full_clocks(decision,inflight=inflight)
        if self.ordinary_fast and not inflight and decision.get('start_now'):
            # Preparation can consume a control step. Recheck only the mode;
            # never move the retained Full start/deadline/frequency plan.
            from perceptbudget.ordinary_fast import choose
            decision['m3_fast_formula']=choose(decision,step=self.ordinary_step,
                previous_mode=self.ordinary_previous,phases=self.ordinary_phases,events=self.events,
                costs=self.ordinary_costs,fast_delay=self.fast_delays[decision['compute_profile']],
                policy=self.ordinary_policy,match_steps=self.events_config.match_steps,
                full_per_fast=self.ordinary_full_per_fast,full_streak=self.ordinary_full_streak,
                every_event=self.fast_every_event)
            self.plan=decision
        return decision

    def _control_full_clocks(self,decision,*,inflight):
        self.clocks.poll()
        if inflight:
            if self.clocks.pending is not None or self.clocks.profile!=self.active_profile or self.clocks.profile=='deep_idle':
                raise RuntimeError('Full frequency changed while inference active')
            return decision
        if self.recovering_plan is not None:
            if self.clocks.pending is None and self.clocks.profile==self.recovery_target:
                self.plan=dict(self.recovering_plan,start_now=True,wait_seconds=0.)
                return self.plan
            return dict(decision,start_now=False,clock_wait=True)
        if self.clocks.pending is not None:
            return dict(decision,start_now=False,clock_wait=True)
        if decision['start_now']:
            target=decision.get('compute_profile','baseline')
            if self.clocks.profile!=target:
                self.recovering_plan=dict(decision)
                self.recovery_target=target
                self.clocks.switch(target)
                if self.start_floor_seconds and decision.get('start_point_rounding',{}).get('floor_allowed'):
                    ack=wait_for_clock(self.clocks,self.start_floor_seconds)
                    self.recovering_plan['startup_ack_wait']=ack
                    if ack['acknowledged']:
                        if self.clocks.profile!=target:raise RuntimeError('Wrong frequency acknowledged')
                        self.plan=dict(self.recovering_plan,start_now=True,wait_seconds=0.)
                        return self.plan
                return dict(decision,start_now=False,clock_wait=True)
            return decision
        wait=decision['latest_start']-decision['decision_monotonic']
        threshold=self.settings['minimum_low_idle_seconds']
        # A newly admitted short gap must not introduce an unbudgeted clock-ACK
        # control tick. That would advance Full and can change MID/HIGH choice.
        # Existing >=200ms behavior is retained; shorter gaps reuse an already
        # planned compute-frequency change. History bins stay at0.2/1s.
        short=threshold<.2 and wait<.2
        budgeted_change=decision.get('compute_profile') not in (None,self.clocks.profile)
        can_lower=not short or budgeted_change
        decision['continue_idle_gate']=dict(predicted_gap_seconds=wait,minimum_gap_seconds=threshold,
            current_profile=self.clocks.profile,short_gap=short,
            planned_compute_profile=decision.get('compute_profile'),
            existing_clock_change=budgeted_change,
            reason='existing_clock_budget' if short and can_lower else 'extra_ack_tick_not_budgeted' if short else 'original_idle_rule',
            downclock=self.clocks.profile!='deep_idle' and wait>=threshold and can_lower)
        if decision['continue_idle_gate']['downclock']:
            self.clocks.switch('deep_idle')
        return decision

    def arm_visual_alarm(self, sample):
        if self.visual_response_enabled and sample.get('alarm') and self.visual_pending is None:
            self.visual_pending=dict(sample_id=sample['sample_id'],sample_step=sample['step'],
                request_step=sample['step'],observed_monotonic=sample['acquisition']['stop_monotonic'],
                reason=sample['reason'])

    def requested(self,request):
        if self.clocks.pending is not None or self.clocks.profile=='deep_idle':
            raise RuntimeError('Cannot infer in deep Continue or before clock acknowledgement')
        DeadlineRuntime.requested(self,request)
        if self.safe_pending is not None:
            request['safe_failure_response']=dict(self.safe_pending)
            self.safe_pending=None
        if self.visual_pending is not None:
            request['visual_monitor_response']=dict(self.visual_pending)
            self.visual_pending=None
        self.active_profile=self.clocks.profile
        request['m4_compute_profile']=self.active_profile
        request['m3_full_frequency']=self.active_profile
        request['m3_three_frequency_policy']=('full_energy_fixed_prepare_v1' if self.full_energy else
            self.plan.get('three_frequency_formula',{}).get('policy','medium_first_v1') if self.plan else 'medium_first_v1')
        if self.start_floor_seconds:request['m3_start_floor_ms']=self.start_floor_seconds*1000
        if request['observation_step'] and self.active_profile=='gpu_emc':
            request['m3_timing']['low_full_measurement_history']=self.plan['low_full_formula']['selected']['history']


def preflight(directory):
    from perceptbudget.experiment import _write_json_atomically
    clock=ThreeClocks(directory)
    result=dict(status='running',changes=[],model_loaded=False)
    try:
        with_clock=False
        clock.__enter__();with_clock=True
        for profile in ('deep_idle','gpu_emc','deep_idle','baseline'):
            clock.switch(profile)
            limit=time.monotonic()+3.
            while clock.pending is not None:
                clock.poll()
                if time.monotonic()>limit:raise TimeoutError('Three-state preflight timeout')
                time.sleep(.002)
            result['changes'].append(dict(clock.last_change))
        result['status']='completed'
    finally:
        if with_clock:
            clock.close();result['restored']=True
        _write_json_atomically(Path(directory)/'three_frequency_preflight.json',result)
    return result
