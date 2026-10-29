"""Read-only M2 inventory for M4; preserve energy units and source provenance."""
import hashlib
import json
from pathlib import Path
from perceptbudget.deadline_controller import percentile
from perceptbudget.device_clocks import PROFILES


def inventory(config):
    path=Path(config['cost_summary'])
    summary=json.loads(path.read_text())
    if summary['status']!='completed' or len(summary['conditions'])!=27:
        raise ValueError('Require completed27-cell M2 table')
    cache={}
    def load(p):
        p=str(p)
        if p not in cache:
            obj=json.loads(Path(p).read_text())
            if obj.get('status')!='completed':raise ValueError('Incomplete M2 source '+p)
            cache[p]=obj
        return cache[p]
    cells=[]
    for row in summary['conditions']:
        cell=dict(row)
        cell['frequencies']=dict(zip(('gpu_hz','emc_hz','cpu_khz'),PROFILES[row['profile']]))
        if row['mode']!='Continue':
            key='full' if row['mode']=='Full' else 'fast'
            sources=([b[key] for b in summary['selected_summary_sources']]
                if row['profile']=='baseline' else [row['source_path']])
            values=[q['service_ms']/1000 for source in sources for q in load(source)['requests']
                if q['profile']==row['profile'] and q['state_history']==row['history']]
            if len(values)!=row['samples']:raise ValueError('Raw service sample count mismatch')
            cell.update(service_samples_seconds=values,p95_service_seconds=percentile(values,.95),
                raw_sources=sources,energy_use='Increment above matched hold; add matching hold over SAME window, not another refresh cost')
        else:
            cell.update(p95_service_seconds=None,
                energy_use='Five-action transient window only; not steady watts or neural service energy')
        cells.append(cell)
    return dict(status='inventory_completed',source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        settings=config,cells=cells,source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in cache},
        remaining_work=['Align hold-power and transition costs on a common planning window',
            'Measure missing directional frequency transitions before enabling their runtime candidates',
            'Wire per-request visual selection and verify paired closed-loop quality'],
        runtime_ready=False)
