"""Recover absolute board energy from immutable M2 power traces."""
import json
from pathlib import Path
from perceptbudget.power import PowerReading,_integrate_report


def integrate(readings,start,stop):
    report=_integrate_report(readings,start,stop,.2)
    if report.total_energy_mj is None or report.missing_count:
        raise ValueError('Incomplete power coverage')
    return report.total_energy_mj/1000


def recover(catalog):
    cells=[]
    for cell in catalog['cells']:
        if cell['mode']=='Continue':continue
        profile,history=cell['profile'],cell['history']
        services=[];batches=[];holds=[];evidence=[]
        for name in cell['raw_sources']:
            source=Path(name);data=json.loads(source.read_text())
            power=source.parent/f'full_service_{profile}_{history}_power.json'
            readings=[PowerReading(**r) for r in json.loads(power.read_text())]
            requests=[q for q in data['requests'] if q['profile']==profile and q['state_history']==history]
            for q in requests:
                start=q['timing']['up']['command_monotonic'];stop=q['measured']['ready_monotonic']
                if abs((stop-start)-q['service_ms']/1000)>.002:
                    raise ValueError('Service boundary disagrees with recorded latency')
                services.append(integrate(readings,start,stop))
            condition=next(c for c in data['conditions'] if c['profile']==profile and c['state_history']==history)
            window=condition['batch_energy'];batch_j=integrate(readings,window['start_monotonic'],window['stop_monotonic'])
            batches.append(dict(energy_j=batch_j,duration_seconds=window['duration_seconds'],requests=len(requests),
                scope='Complete repeated batch, includes dwell and logging; not pure service'))
            holdpath=source.parent/f'hold_{profile}_{history}_power.json'
            hold=[PowerReading(**r) for r in json.loads(holdpath.read_text())]
            window=condition['matched_hold']['energy'];energy=integrate(hold,window['start_monotonic'],window['stop_monotonic'])
            holds.append(dict(energy_j=energy,duration_seconds=window['duration_seconds'],mean_power_w=energy/window['duration_seconds'],
                scope='Matched hold replay includes scheduled state changes; not steady Continue power'))
            evidence.append(dict(source=str(source),power=str(power),hold_power=str(holdpath)))
        if len(services)!=cell['samples']:raise ValueError('Unexpected service count')
        cells.append(dict(mode=cell['mode'],profile=profile,history=history,samples=len(services),
            mean_absolute_service_energy_j=sum(services)/len(services),
            energy_scope='Mean board energy over preparation-start to ready windows; no hold subtraction; no event energy tails claimed',
            batches=batches,matched_hold_replays=holds,evidence=evidence))
    return dict(status='completed',cells=cells,runtime_ready=False,
        missing_measurements=['Steady model-resident Continue power with active control at each selected frequency',
            'Directional state-transition energy on matched planning windows; existing fast transitions may be below sensor resolution'],
        next_step='Audit existing idle/transition traces for compatible control workload; supplement only missing conditions')


def recover_control_cycles(path):
    """Reuse actual-control matched cycles; do not infer sub-resolution switches."""
    path=Path(path);data=json.loads(path.read_text());rows=[]
    if data['status']!='completed' or data.get('preflight_only'):
        raise ValueError('Require completed control measurements')
    for row in data['rows']:
        tag=str(row['idle_seconds']).replace('.','p')
        power=path.parent/f'{tag}_{row["profile"]}_power.json'
        readings=[PowerReading(**r) for r in json.loads(power.read_text())]
        windows={}
        for name in ('matched_cycle','early_cycle','through_handoff'):
            w=row[name];energy=integrate(readings,w['start_monotonic'],w['stop_monotonic'])
            windows[name]=dict(energy_j=energy,duration_seconds=w['duration_seconds'])
        rows.append(dict(idle_seconds=row['idle_seconds'],profile=row['profile'],index=row['index'],
            windows=windows,power_path=str(power),underflows=row['measured']['underflows']))
    return dict(source=str(path),rows=rows,
        scope='Absolute energy in real-control common windows; includes waiting, switching, service and tail; no component subtraction')
