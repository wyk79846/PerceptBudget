"""Absolute service costs: latest repeated hot measurements override old means."""
import json
from pathlib import Path
from perceptbudget.deadline_controller import percentile
from perceptbudget.m4_catalog import inventory


def load_service_costs(config):
    source=Path(config['absolute_service_source']);old=json.loads(source.read_text())
    if old['status']!='completed':raise ValueError('Incomplete absolute energy source')
    latency={(r['mode'],r['profile'],r['history']):r for r in inventory(config)['cells'] if r['mode']!='Continue'}
    cells={(r['mode'],r['profile'],r['history']):dict(r,source=str(source)) for r in old['cells']}
    for key,cell in cells.items():
        measured=latency[key]['service_samples_seconds']
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
            mean_absolute_service_energy_j=sum(energies)/len(energies),
            mean_service_seconds=sum(samples)/len(samples),p95_service_seconds=percentile(samples,.95),
            sources=sources,source=str(source),energy_scope='Latest three-batch absolute service mean')
    return list(cells.values())
