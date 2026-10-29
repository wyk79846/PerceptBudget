import json
from pathlib import Path
import pytest
from perceptbudget.m4_catalog import inventory


def test_catalog_preserves_energy_semantics_and_reconstructs_tail_from_raw_samples(tmp_path):
    requests=[dict(profile='baseline',state_history=h,service_ms=v) for h in ('hot','short','long') for v in (100,200)]
    source=tmp_path/'requests.json';source.write_text(json.dumps(dict(status='completed',requests=requests)))
    rows=[]
    for mode in ('Full','Fast','Continue'):
        for profile in ('baseline','balanced','gpu_emc'):
            for h in ('hot','short','long'):
                rows.append(dict(mode=mode,profile=profile,history=h,samples=2,source_path=str(source),
                    mean_incremental_refresh_energy_j=None if mode=='Continue' else 5,
                    mean_transient_control_energy_j=3 if mode=='Continue' else None))
    all_requests=[dict(q,profile=p) for p in ('baseline','balanced','gpu_emc') for q in requests]
    source.write_text(json.dumps(dict(status='completed',requests=all_requests)))
    summary=tmp_path/'summary.json';summary.write_text(json.dumps(dict(status='completed',conditions=rows,
        selected_summary_sources=[dict(full=str(source),fast=str(source))])))
    result=inventory({'cost_summary':str(summary)})
    assert len(result['cells'])==27 and not result['runtime_ready']
    assert all(c['p95_service_seconds']==.2 for c in result['cells'] if c['mode']!='Continue')
    assert all(c['p95_service_seconds'] is None for c in result['cells'] if c['mode']=='Continue')
    assert all(c['mean_incremental_refresh_energy_j'] is None for c in result['cells'] if c['mode']=='Continue')
    summary.write_text(json.dumps(dict(status='completed',conditions=rows[:-1])))
    with pytest.raises(ValueError,match='27-cell'):inventory({'cost_summary':str(summary)})
