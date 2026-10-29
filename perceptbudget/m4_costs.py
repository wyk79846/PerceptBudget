"""Absolute service costs: latest repeated hot measurements override old means."""
import json
from pathlib import Path
from perceptbudget.deadline_controller import percentile
from perceptbudget.m4_catalog import inventory
from perceptbudget.device_clocks import PROFILES


def load_service_costs(config):
    source=Path(config['absolute_service_source']);old=json.loads(source.read_text())
    if old['status']!='completed':raise ValueError('Incomplete absolute energy source')
    latency={(r['mode'],r['profile'],r['history']):r for r in inventory(config)['cells'] if r['mode']!='Continue'}
    cells={(r['mode'],r['profile'],r['history']):dict(r,source=str(source)) for r in old['cells']}
    for key,cell in cells.items():
        measured=latency[key]['service_samples_seconds']
        raw=json.loads(Path(cell['evidence'][0]['source']).read_text())
        request=next(q for q in raw['requests'] if q['profile']==cell['profile'] and q['state_history']==cell['history'])
        actual=request['timing']['down']['actual']
        initial=next((name for name,frequencies in PROFILES.items()
            if name in ('baseline','balanced','gpu_emc') and frequencies[0]==actual['class/devfreq/17000000.gpu/cur_freq']
            and frequencies[1]==actual['kernel/debug/bpmp/debug/clk/emc/rate']),None)
        if initial is None:raise ValueError('Unknown measured service entry state')
        cell['from_profile']=initial
        cell.update(mean_service_seconds=sum(measured)/len(measured),p95_service_seconds=percentile(measured,.95))
    source=Path(config['preferred_hot_energy_source']);review=json.loads(source.read_text())
    if review['status']!='reviewed':raise ValueError('Unreviewed repeated energy measurements')
    for mode in ('Full','Fast'):
        rows=[r for r in review['rows'] if r['mode']==mode]
        if len(rows)!=3:raise ValueError('Require three repeated batches')
        samples=[];energies=[];sources=[]
        for row in rows:
            path=Path(row['source']);data=json.loads(path.read_text());sources.append(str(path))
            if data['status']!='completed' or not data.get('restored'):raise ValueError('Incomplete repeat')
            for q in data['requests']:
                if q['profile']!='baseline' or q['state_history']!='hot':raise ValueError('Unexpected repeat geometry')
                samples.append(q['service_ms']/1000);energies.append(q['energy']['energy_j'])
        if len(samples)!=60:raise ValueError('Require60 services per mode')
        cells[mode,'baseline','hot']=dict(mode=mode,profile='baseline',history='hot',samples=len(samples),
            from_profile='baseline',mean_absolute_service_energy_j=sum(energies)/len(energies),
            mean_service_seconds=sum(samples)/len(samples),p95_service_seconds=percentile(samples,.95),
            sources=sources,source=str(source),energy_scope='Latest three-batch absolute service mean')
    return list(cells.values())
