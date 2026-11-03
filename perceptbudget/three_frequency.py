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
        from perceptbudget.device_clock_screen import ROOT
        self.lease.command=lambda: ['sudo','-n','/usr/bin/python3',
            str(ROOT/'scripts/control_device_clocks.py'),'runtime3','--seconds','120']

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
        self.transitions=config['m3_three_frequency_transitions']
        self.start_floor_seconds=config.get('m3_start_floor_ms',0)/1000
        if self.start_floor_seconds not in (0.,.025):raise ValueError('Startup floor supports0 or25ms')
        self.recovery_target=None

    def start_clocks(self,directory):
        self.clocks=ThreeClocks(directory)
        self.clocks.__enter__()

    def decide(self,*,now,step,queue,rows,inflight,rtc_delay_steps=None):
        d=DeadlineRuntime.decide(self,now=now,step=step,queue=queue,rows=rows,
                                 inflight=inflight,rtc_delay_steps=rtc_delay_steps)
        if inflight or self.recovering_plan is not None or self.clocks.pending is not None:
            return d
        item=queue.get(step)
        if item is None:return d
        period=d['estimated_control_period_seconds']
        guard=math.ceil(self.margin.seconds/period-1e-9)+(self.clocks.profile!='gpu_emc')
        submit=self.chunks[item[1]]['origin']+self.horizon-self.low_delay_steps
        cap=now+(submit-step-guard)*period
        medium=max(percentile(list(v),self.settings['service_quantile']) for v in self.low_histories.values())
        formula=select_full(d,medium_seconds=medium,medium_cap=cap,
                            current_profile=self.clocks.profile,transitions=self.transitions)
        chosen=formula['selected']
        # Preserve LowFullRuntime's separate medium measurement history and RTC
        # queue; it uses the old low_full_formula selected.history field.
        age=max(0.,now-(self.clocks.last_low or now))+max(0.,chosen['latest_start']-now)
        history='hot' if age<.2 else 'short' if age<1. else 'long'
        medium_chosen=chosen['profile']=='gpu_emc'
        due=chosen['latest_start']-now<period-1e-9
        startup=None
        if self.start_floor_seconds:
            startup=start_point(step,now,chosen['latest_start'],period,self.start_floor_seconds)
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
        d.update(three_frequency_formula=formula,start_constraint=constraint,
            low_full_formula=dict(selected=dict(chosen,history=history),policy='medium_first_v1'),
            compute_profile=chosen['profile'],service_budget_history='low' if medium_chosen else d['service_budget_history'],
            service_seconds=chosen['service_seconds'],
            latest_start=chosen['latest_start'],wait_seconds=0. if due else max(0.,chosen['latest_start']-now),
            start_now=due,path='three_frequency',
            rtc_delay_steps=self.low_delay_steps if medium_chosen else rtc_delay_steps,
            rtc_latest_submit_step=submit if medium_chosen else d['rtc_latest_submit_step'],
            rtc_submit_guard_steps=guard if medium_chosen else d['rtc_submit_guard_steps'],
            deadline_already_unreachable=now+chosen['service_seconds']+self.margin.seconds>d['deadline'])
        if startup is not None:d['start_point_rounding']=startup
        self.plan=d
        return d

    def control_clocks(self,decision,*,inflight):
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
        if self.clocks.profile!='deep_idle' and wait>=self.settings['minimum_low_idle_seconds']:
            self.clocks.switch('deep_idle')
        return decision

    def requested(self,request):
        if self.clocks.pending is not None or self.clocks.profile=='deep_idle':
            raise RuntimeError('Cannot infer in deep Continue or before clock acknowledgement')
        DeadlineRuntime.requested(self,request)
        self.active_profile=self.clocks.profile
        request['m4_compute_profile']=self.active_profile
        request['m3_full_frequency']=self.active_profile
        request['m3_three_frequency_policy']='medium_first_v1'
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
