"""Initial eta calibration: identical event policy, online P95 and joint DVFS."""
import copy

MARGINS_MS=(50,25,0)


def scan_settings(settings, *, long_run=False, round_index=1):
    if round_index not in (1,2):
        raise ValueError("Use independent round1 or2")
    result=copy.deepcopy(settings)
    result.update(mode='deadline_dvfs',dvac_paired=True,trigger_comparison=True,
        margin_scan=True,trigger_variants=['events_only'],trigger_repeats=3,
        calibration_tasks=[[s,1] for s in ('libero_spatial','libero_object','libero_goal')])
    result['margin_candidates_ms']=[50,25] if long_run else list(MARGINS_MS)
    result.update(margin_long_run=long_run,margin_round=round_index)
    if long_run:
        tasks=[[s,t] for t in (1,3,6) for s in ('libero_spatial','libero_object','libero_goal')]
        result.update(trigger_repeats=2,calibration_tasks=tasks if round_index==1 else tasks[::-1])
    result['timing'].update(maximum_action_offset=50,maximum_observation_age_seconds=None,
        service_quantile=.95,margin_updates=False)
    return result


def configure_margin(config, repeat):
    if repeat not in (1,2,3):
        raise ValueError('Unknown margin candidate')
    milliseconds=MARGINS_MS[repeat-1]
    config['m3_timing']=dict(config['m3_timing'],safety_margin_seconds=milliseconds/1000.)
    config.update(m3_margin_ms=milliseconds,m3_dvfs_policy='joint',
        m3_event_coverage_mode='event_only',m3_signal_policy='earliest',
        m3_merge_slowdown_steps=None,m3_ignore_turn=False)
    return milliseconds


def summarize_scan(directory, record):
    import json
    from pathlib import Path
    from perceptbudget.experiment import _write_json_atomically
    from perceptbudget.m3_comparison import audit_group_clocks
    rows=[]
    for episode in record['episodes']:
        run=Path(episode['run'])
        config=json.loads((run/'run.json').read_text())['config_snapshot']
        requests=json.loads((run/'rtc/q5_requests.json').read_text())
        timings=[r['m3_timing'] for r in requests if r.get('m3_timing')]
        energy=json.loads((run/'summary.json').read_text())['episode']['including_drain_power']
        audit=audit_group_clocks(run,'D_joint')
        if not audit['passed']:
            raise ValueError('Margin scan clock/power audit failed: '+str(run))
        rows.append(dict(suite=episode['suite'],task_id=episode['task_id'],
            margin_ms=config['m3_margin_ms'],outcome=episode['outcome'],steps=episode['steps'],
            task_seconds=episode['wall_seconds'],energy_j=energy['total_energy_mj']/1000.,
            underflow_steps=episode['underflow_steps'],refreshes=len(requests),
            deadline_misses=sum(t.get('arrival_lateness_seconds',0)>0 for t in timings),
            maximum_lateness_seconds=max((t.get('arrival_lateness_seconds',0) for t in timings),default=0),
            service_samples=len(timings),
            service_budget_exceedances=sum(t.get('actual_service_seconds',0)>t['service_seconds'] for t in timings),
            already_unreachable_misses=sum(t.get('arrival_lateness_seconds',0)>0 and t.get('deadline_already_unreachable',False) for t in timings),
            clock_power_audit_passed=True,run=str(run)))
    long_run=bool(record.get('margin_long_run'))
    result=dict(status='completed',service_quantile=.95,rows=rows,
        long_run=long_run,independent_round=record.get('margin_round',1),
        paired_bootstrap_audit=record['paired_bootstrap_audit'],
        scope='Expanded paired eta validation; report observed miss rates, not hard real-time guarantees' if long_run else 'Nine-episode initial eta screen, not long-run epsilon/eta calibration')
    _write_json_atomically(Path(directory)/'margin_summary.json',result)
    return result
