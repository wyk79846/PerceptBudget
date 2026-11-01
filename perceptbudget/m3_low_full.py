"""Two Full frequencies on M3; Continue and key-event logic stay in M3."""
import math
from collections import deque
from statistics import median

from perceptbudget.closed_loop import DeadlineRuntime
from perceptbudget.deadline_controller import percentile
from perceptbudget.m4_candidates import hold_energy


def load_costs(settings):
    import json
    from pathlib import Path
    from perceptbudget.m4_costs import load_service_costs
    from perceptbudget.m4_candidates import wait_model
    cells=[c for c in load_service_costs(settings) if c['mode']=='Full'
           and c['profile'] in ('baseline','gpu_emc')]
    return dict(cells=cells, waits=wait_model(json.loads(Path(settings['wait_cost_source']).read_text())),
                scope='Absolute measured entry-to-ready energy; recovery included, no extra up charge')


def initial_low_delay(costs, control_period):
    """Initialize RTC like HIGH: measured inference time / this environment's period."""
    if not math.isfinite(control_period) or control_period<=0:
        raise ValueError('Invalid calibrated control period')
    samples=[v for c in costs['cells'] if c['profile']=='gpu_emc'
             for v in c['service_samples_seconds']]
    if not samples or any(not math.isfinite(v) or v<=0 for v in samples):
        raise ValueError('Missing measured low Full durations')
    return math.ceil(median(samples)/control_period)


def choose(decision, *, costs, low_budget, quiet_age, ordinary, rtc_low_cap):
    """Use the whole remaining window: LOW waits less; the event/buffer deadline stays fixed."""
    now=decision['decision_monotonic'];end=decision['deadline']-decision['margin_seconds']
    rows=[]
    for profile in ('baseline','gpu_emc'):
        q=decision['service_seconds'] if profile=='baseline' else low_budget
        cap=decision['latest_start'] if profile=='baseline' else rtc_low_cap
        latest=min(end-q,cap)
        before=max(0.,latest-now)
        entry_age=quiet_age+before
        history='hot' if entry_age<.2 else 'short' if entry_age<1. else 'long'
        cell=next(c for c in costs['cells'] if c['profile']==profile
                  and c['from_profile']=='gpu_emc' and c['history']==('short' if history=='hot' and profile=='baseline' else history))
        mean=cell['mean_service_seconds']
        tail=max(0.,end-(now+before+mean))
        wait_j=hold_energy(costs['waits'],'gpu_emc',quiet_age,before)
        tail_j=hold_energy(costs['waits'],profile,0.,tail)
        energy=wait_j+cell['mean_absolute_service_energy_j']+tail_j
        rows.append(dict(profile=profile,history=cell['history'],service_seconds=q,
            mean_seconds=mean,latest_start=latest,feasible=latest>=now-1e-9,
            quality_allowed=profile=='baseline' or ordinary,
            wait_before_seconds=before,wait_before_energy_j=wait_j,
            service_energy_j=cell['mean_absolute_service_energy_j'],
            tail_seconds=tail,tail_energy_j=tail_j,energy_j=energy,
            source=cell['source']))
    low=rows[1]
    selected=low if low['feasible'] and low['quality_allowed'] else rows[0]
    return dict(policy='shared_deadline_v4',selected=selected,candidates=rows,quiet_age_seconds=quiet_age,
        ordinary=ordinary,window_end=end,rtc_low_cap=rtc_low_cap,
        reason='low_meets_deadline' if selected is low else 'high_key_phase' if not ordinary else 'high_low_too_late',
        energy_used_for_selection=False,
        energy_scope='LOW entry through common deadline-minus-margin; service includes any recovery; Continue separate')


class LowFullRuntime(DeadlineRuntime):
    def __init__(self,config):
        super().__init__(config)
        if self.fast_selection or config.get('m4_enabled'):
            raise ValueError('Low Full experiment requires M3 without Fast')
        self.costs=config['m3_low_full_costs']
        self.low_histories={h:deque((v+self.initial_period for v in next(c for c in self.costs['cells']
            if c['profile']=='gpu_emc' and c['history']==h)['service_samples_seconds']),maxlen=128)
            for h in ('hot','short','long')}
        self.active_profile='baseline'
        self.pending_low=False
        self.low_delay_steps=config.get('m3_low_full_initial_delay',
            initial_low_delay(self.costs,self.initial_period))

    def committed(self,request,commit):
        low=request.get('m4_compute_profile')=='gpu_emc'
        samples={k:list(v) for k,v in self.times.samples.items()} if low else None
        recovery=list(self.times.recovery_samples) if low else None
        super().committed(request,commit)
        from perceptbudget.key_phases import absolute_intervals
        self.chunks[request['prediction_id']]['low_full_phases']=absolute_intervals(
            request['actions'],request['observation_step'],self.events_config)
        if low:
            for k,v in samples.items():self.times.samples[k].clear();self.times.samples[k].extend(v)
            self.times.recovery_samples.clear();self.times.recovery_samples.extend(recovery)
            history=request['m3_timing']['low_full_formula']['selected']['history']
            self.low_histories[history].append(commit['commit_monotonic']-request['observation_monotonic'])

    def decide(self,*,now,step,queue,rows,inflight,rtc_delay_steps=None):
        d=super().decide(now=now,step=step,queue=queue,rows=rows,inflight=inflight,rtc_delay_steps=rtc_delay_steps)
        self.pending_low=False
        if inflight or self.recovering_plan is not None:return d
        # First refresh stays high. LOW admission only from already-low Continue;
        # the old M3 handles HIGH->LOW waiting and measured HIGH recovery.
        item=queue.get(step)
        if (not item or item[1]==0 or not self.clocks or self.clocks.profile!='gpu_emc'
                or self.clocks.pending is not None):return d
        low_q=max(percentile(list(v),self.settings['service_quantile']) for v in self.low_histories.values())
        phases=self.chunks[item[1]].get('low_full_phases',[])
        upcoming=[e['target_step'] for e in self.events if e['target_step']>=step]
        ordinary=(d['reason']!='action_event' and
            not any(p['start_step']<=step<p['stop_step'] for p in phases) and
            not any((target-step)*d['estimated_control_period_seconds']<=low_q+self.margin.seconds for target in upcoming))
        period=d['estimated_control_period_seconds']
        guard=math.ceil(self.margin.seconds/period-1e-9)
        latest_submit=self.chunks[item[1]]['origin']+self.horizon-self.low_delay_steps
        cap=now+(latest_submit-step-guard)*period if latest_submit is not None else float('inf')
        age=max(0.,now-(self.clocks.last_low or now))
        formula=choose(d,costs=self.costs,low_budget=low_q,quiet_age=age,ordinary=ordinary,rtc_low_cap=cap)
        d['low_full_formula']=formula
        if formula['selected']['profile']=='gpu_emc':
            selected=formula['selected'];self.pending_low=True
            d.update(service_seconds=low_q,latest_start=selected['latest_start'],
                wait_seconds=max(0.,selected['latest_start']-now),start_now=selected['latest_start']<=now,
                path='low_full_compute',service_budget_history='low',rtc_submit_guard_steps=guard,
                rtc_delay_steps=self.low_delay_steps,rtc_latest_submit_step=latest_submit,
                start_constraint='rtc_prefix' if cap<d['deadline']-low_q-self.margin.seconds else 'service_deadline',
                deadline_already_unreachable=now+low_q+self.margin.seconds>d['deadline'])
        self.plan=d
        return d

    def control_clocks(self,decision,*,inflight):
        if inflight and self.active_profile=='gpu_emc':
            self.clocks.poll()
            if self.clocks.profile!='gpu_emc' or self.clocks.pending is not None:
                raise RuntimeError('Frequency changed during low Full inference')
            return decision
        if self.pending_low:
            self.clocks.poll()
            if self.clocks.profile!='gpu_emc' or self.clocks.pending is not None:
                raise RuntimeError('Low Full requires an acknowledged LOW state')
            self.plan=decision
            return decision
        return super().control_clocks(decision,inflight=inflight)

    def requested(self,request):
        super().requested(request)
        self.active_profile=self.clocks.profile if self.clocks else 'baseline'
        request['m4_compute_profile']=self.active_profile
        request['m3_full_frequency']=self.active_profile
        if request['observation_step'] and self.active_profile=='gpu_emc':
            if not self.pending_low:raise RuntimeError('Low Full request without admitted phase/deadline plan')
            request['m3_timing']['low_full_measurement_history']=self.plan['low_full_formula']['selected']['history']
