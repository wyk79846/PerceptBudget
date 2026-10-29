"""Estimate active-control quiet power from existing real-async runs."""
import json
from pathlib import Path
from perceptbudget.device_clocks import PROFILES
from perceptbudget.power import PowerReading,_integrate_report


def overlaps(a,b,c,d):
    return a<d and c<b


def extract(run,guard_seconds=.3):
    run=Path(run);episode=json.loads((run/'summary.json').read_text())['episode']
    raw=episode['power']['raw_readings']
    readings=[PowerReading(x['timestamp_monotonic'],x['power_mw'],x.get('error')) for x in raw]
    requests=json.loads((run/'rtc/q5_requests.json').read_text())
    busy=[]
    quiet_origins=[]
    for q in requests:
        t=q.get('m3_timing') or {}
        start=t.get('service_start_monotonic') or q['observation_monotonic']
        busy.append((start,q['ready_monotonic']+guard_seconds))
        quiet_origins.append(q['ready_monotonic'])
    clocks=[]
    for line in (run/'rtc/runtime_clocks.jsonl').read_text().splitlines():
        try:clocks.append(json.loads(line))
        except ValueError:pass
    switches=[c for c in clocks if c.get('status')=='APPLIED']
    quiet_origins.extend(c['observed_match_monotonic'] for c in switches)
    blocked=[(c['command_monotonic'],c['observed_match_monotonic']+guard_seconds) for c in switches]
    samples=[c for c in clocks if c.get('status')=='SAMPLE']
    controls=[json.loads(line) for line in (run/'rtc/trace.jsonl').read_text().splitlines()
              if '"control_step"' in line]
    windows=[]
    for left,right in zip(controls,controls[1:]):
        a,b=left['tick_monotonic'],right['tick_monotonic']
        if b<=a or left.get('inference_active') or right.get('inference_active'):continue
        if left.get('late_hold') or right.get('late_hold'):continue
        if any(overlaps(a,b,c,d) for c,d in busy+blocked):continue
        near=[s for s in samples if a-.03<=s['monotonic']<=b+.03]
        if not near or near[0]['monotonic']>a+.03 or near[-1]['monotonic']<b-.03:continue
        matches=[]
        for profile in ('baseline','gpu_emc'):
            gpu,emc,cpu=PROFILES[profile]
            target=(gpu,emc,cpu,cpu)
            keys=('class/devfreq/17000000.gpu/cur_freq','kernel/debug/bpmp/debug/clk/emc/rate',
                  'devices/system/cpu/cpufreq/policy0/scaling_cur_freq','devices/system/cpu/cpufreq/policy4/scaling_cur_freq')
            if all(tuple(s['actual'].get(k) for k in keys)==target for s in near):matches.append(profile)
        if len(matches)!=1:continue
        power=_integrate_report(readings,a,b,.2)
        if power.total_energy_mj is None or power.missing_count:continue
        origin=max((t for t in quiet_origins if t<=a),default=a)
        windows.append(dict(profile=matches[0],start=a,stop=b,seconds=b-a,
            quiet_age_seconds=a-origin,energy_j=power.total_energy_mj/1000))
    transitions=[dict(profile=c['profile'],seconds=c['observed_match_monotonic']-c['command_monotonic'],
        energy_status='unresolved_at_100ms_power_poll',command_monotonic=c['command_monotonic']) for c in switches]
    return dict(run=str(run),guard_seconds=guard_seconds,windows=windows,transitions=transitions)
