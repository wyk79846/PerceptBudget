"""Partition measured module energy into service and Continue, exactly once."""
import json
from pathlib import Path
from perceptbudget.power import PowerReading, _integrate_report


def build_ledger(episode, requests, *, max_gap_seconds=.3, forbid_hot_fast=False):
    start=episode['including_drain_start_monotonic']
    stop=episode['including_drain_stop_monotonic']
    readings=[PowerReading(r['timestamp_monotonic'],r['power_mw'],r.get('error'))
              for r in episode['including_drain_power']['raw_readings']]
    commits={r['prediction_id']:r['commit_monotonic'] for r in episode['commits']}
    services=[]
    for request in requests:
        timing=request.get('m3_timing',{})
        recovery=timing.get('service_start_monotonic') is not None
        a=timing['service_start_monotonic'] if recovery else request['observation_monotonic']
        b=commits.get(request['prediction_id'],request.get('ready_monotonic'))
        if b is None:raise ValueError('Request has no completion timestamp')
        mode=request.get('m3_fast_choice',{}).get('mode','Full')
        if forbid_hot_fast and mode=='Fast' and not recovery:
            raise ValueError('Hot Fast violates selected policy')
        a=max(start,a);b=min(stop,b)
        if b<a:raise ValueError('Invalid service interval')
        label=('Recovery ' if recovery else 'Hot ')+mode
        if request.get('m3_full_frequency')=='gpu_emc':label='Low Full'
        services.append((a,b,label,request['prediction_id']))
    windows=[];cursor=start
    for a,b,label,pid in sorted(services):
        if a<cursor-1e-8:raise ValueError('Overlapping services would double count energy')
        if a>cursor:windows.append((cursor,a,'Continue',None))
        windows.append((a,b,label,pid));cursor=b
    if cursor<stop:windows.append((cursor,stop,'Continue',None))
    terminal=episode.get('terminal_low')
    if terminal:
        boundary=terminal['start_monotonic']
        separated=[]
        for a,b,label,pid in windows:
            if b<=boundary:
                separated.append((a,b,label,pid))
            elif a>=boundary:
                separated.append((a,b,'Terminal LOW tail',None))
            else:
                separated.extend(((a,boundary,label,pid),(boundary,b,'Terminal LOW tail',None)))
        windows=separated
    rows=[];totals={}
    for a,b,label,pid in windows:
        report=_integrate_report(readings,a,b,max_gap_seconds)
        row=dict(activity=label,prediction_id=pid,start_monotonic=a,stop_monotonic=b,
                 seconds=b-a,covered_energy_j=(report.energy_mj or 0.)/1000,
                 total_energy_j=None if report.total_energy_mj is None else report.total_energy_mj/1000,
                 coverage_fraction=report.coverage_fraction)
        rows.append(row)
        total=totals.setdefault(label,dict(seconds=0.,covered_energy_j=0.,complete=True))
        total['seconds']+=b-a;total['covered_energy_j']+=row['covered_energy_j']
        total['complete'] &= b==a or row['total_energy_j'] is not None
    whole=_integrate_report(readings,start,stop,max_gap_seconds)
    energy=sum(r['covered_energy_j'] for r in rows)
    if abs(energy-(whole.energy_mj or 0.)/1000)>1e-6:
        raise ValueError('Partition energy does not equal whole interval')
    return dict(scope='Measured module input energy by time activity, including concurrent simulation/control; not GPU-only',
                service_scope='Recovery command (or hot observation) through action commit; recovery counted once',
                continue_scope='Other time excluding separately labelled terminal LOW tail; includes clock lowering and post-inference waiting',
                totals=totals,windows=rows,seconds=stop-start,covered_energy_j=energy,
                total_energy_j=None if whole.total_energy_mj is None else whole.total_energy_mj/1000,
                partition_verified=True)


def write_run_ledger(run, *, forbid_hot_fast=False):
    run=Path(run)
    episode=json.loads((run/'summary.json').read_text())['episode']
    requests=json.loads((run/'rtc/q5_requests.json').read_text())
    ledger=build_ledger(episode,requests,forbid_hot_fast=forbid_hot_fast)
    (run/'service_continue_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    return dict(path=str(run/'service_continue_ledger.json'),totals=ledger['totals'],partition_verified=True)
