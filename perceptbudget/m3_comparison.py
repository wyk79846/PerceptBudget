"""Frozen M3 four-group plan, group configuration and task-level reporting."""
import copy
import csv
import hashlib
import json
from pathlib import Path

from perceptbudget.experiment import _write_json_atomically


GROUPS=('A_fixed_q5','B_high','C_independent','D_joint')


def comparison_settings(settings, *, screen=False):
    result=copy.deepcopy(settings)
    formal=result['formal']
    if formal['trigger_strategy'] not in ('events_only','variance_only','events_and_variance'):
        raise ValueError('Unknown formal trigger strategy')
    if formal['request_timeout_seconds']!=3. or formal['cleanup_grace_seconds']!=3.:
        raise ValueError('Formal watchdog settings must match the validated3s request and3s cleanup')
    proof=Path(formal['watchdog_reference'])
    evidence=json.loads(proof.read_text())
    if evidence.get('status')!='completed' or not evidence.get('checks') or not all(evidence['checks'].values()):
        raise ValueError('Require passed real-model watchdog checks')
    if any(not row.get('resumed_execution_step') for row in evidence['cases'][:2]):
        raise ValueError('Watchdog evidence must include recovered action execution')
    tasks=([['libero_spatial',0]] if screen else
        [[suite,tid] for tid in formal['task_ids'] for suite in formal['suites']])
    result.update(mode='deadline_dvfs',dvac_paired=True,formal_comparison=True,
        comparison_screen=screen,calibration_tasks=tasks)
    result['timing'].update(maximum_action_offset=50,maximum_observation_age_seconds=None,
        safety_margin_seconds=.05,margin_updates=False,minimum_low_idle_seconds=.2)
    result['formal']['watchdog_reference_sha256']=hashlib.sha256(proof.read_bytes()).hexdigest()
    return result


def group_order(task_index):
    if type(task_index) is not int or task_index<0:
        raise ValueError('Use a nonnegative task index')
    offset=task_index%len(GROUPS)
    return GROUPS[offset:]+GROUPS[:offset]


def configure_group(config, group):
    if group not in GROUPS:
        raise ValueError('Unknown formal comparison group')
    from perceptbudget.closed_loop import configure_fixed_baseline
    strategy=config['m3_formal_signal']
    if strategy not in ('events_only','variance_only','events_and_variance'):
        raise ValueError('Unknown formal trigger strategy')
    active=group!='A_fixed_q5'
    config.update(m3_formal_group=group,m3_dvac_enabled=active and strategy!='events_only',
        m3_action_events_enabled=active and strategy!='variance_only',m3_merge_slowdown_steps=None,m3_ignore_turn=False,
        m3_signal_policy='earliest',m3_event_coverage_mode='event_only',
        m3_dvfs_policy='independent' if group=='C_independent' else 'joint',
        m3_open_probe_plan=None,m3_watchdog_case=None,
        m3_hold_high_clocks=group in ('A_fixed_q5','B_high'))
    if group=='A_fixed_q5':
        return configure_fixed_baseline(config,5)
    config.update(m3_scheduler_mode='deadline_high' if group=='B_high' else 'deadline_dvfs',
        execution_mode='m3_formal_actual_async',m3_device_profile='high' if group=='B_high' else 'dvfs')
    return 5


def audit_group_clocks(run, group):
    from perceptbudget.device_clock_screen import frequency_check
    run=Path(run)
    lines=(run/'rtc/runtime_clocks.jsonl').read_text().splitlines()
    events=[]
    for line in lines:
        try:events.append(json.loads(line))
        except ValueError:pass
    requests=json.loads((run/'rtc/q5_requests.json').read_text())
    power=json.loads((run/'summary.json').read_text())['episode']['including_drain_power']
    power_complete=(power.get('total_energy_mj') is not None and power.get('coverage_fraction',0)>=1.-1e-9
        and power.get('missing_count',0)==0)
    windows=[frequency_check(events,'baseline',q['observation_monotonic'],q['ready_monotonic'])
        for q in requests]
    no_low=(not any(e.get('status')=='APPLIED' and e.get('profile')=='gpu_emc' for e in events)
        if group in ('A_fixed_q5','B_high') else True)
    return dict(passed=bool(windows) and all(w['matches'] for w in windows) and no_low and 'RESTORED' in lines and power_complete,
        full_high_windows=windows,high_group_never_lowered=no_low,restored='RESTORED' in lines,
        task_power_complete=power_complete)


def summarize_comparison(directory, record, *, screen):
    directory=Path(directory)
    bootstrap=record['paired_bootstrap_audit']
    eligibility={}
    for case in bootstrap['cases']:
        key=(case['suite'],case['task_id'])
        eligibility[key]=eligibility.get(key,True) and case['passed']
    rows=[]
    for episode in record['episodes']:
        run=Path(episode['run']);data=json.loads((run/'summary.json').read_text())['episode']
        requests=json.loads((run/'rtc/q5_requests.json').read_text())
        rows.append(dict(suite=episode['suite'],task_id=episode['task_id'],group=episode['view'],
            outcome=episode['outcome'],steps=episode['steps'],task_seconds=episode['wall_seconds'],
            including_drain_seconds=data['including_drain_seconds'],
            total_energy_j=data['including_drain_power']['total_energy_mj']/1000,
            power_coverage=data['including_drain_power']['coverage_fraction'],
            requests=len(requests),underflow_steps=len(data['underflow_steps']),
            exact_initial_pair=eligibility.get((episode['suite'],episode['task_id']),False),run=str(run)))
    groups=[]
    for group in GROUPS:
        cases=[r for r in rows if r['group']==group]
        if not cases:continue
        successes=[r for r in cases if r['outcome']=='success']
        energy=sum(r['total_energy_j'] for r in cases)
        groups.append(dict(group=group,attempts=len(cases),successes=len(successes),
            success_rate=len(successes)/len(cases),mean_attempt_energy_j=energy/len(cases),
            energy_per_success_j=energy/len(successes) if successes else None,
            mean_success_task_seconds=sum(r['task_seconds'] for r in successes)/len(successes) if successes else None,
            total_underflow_steps=sum(r['underflow_steps'] for r in cases)))
    pairs=[]
    for suite,tid in sorted(eligibility):
        cases={r['group']:r for r in rows if r['suite']==suite and r['task_id']==tid}
        if set(cases)!=set(GROUPS):raise ValueError('Incomplete formal task')
        for before,after in (('A_fixed_q5','B_high'),('B_high','C_independent'),('C_independent','D_joint'),('A_fixed_q5','D_joint')):
            a,b=cases[before],cases[after]
            pairs.append(dict(suite=suite,task_id=tid,before=before,after=after,
                exact_initial_pair=eligibility[(suite,tid)],both_success=a['outcome']==b['outcome']=='success',
                attempt_energy_difference_j=b['total_energy_j']-a['total_energy_j'],
                task_time_difference_seconds=b['task_seconds']-a['task_seconds']))
    strategy=record.get('formal_trigger_strategy','variance_only')
    result=dict(status='completed',screen=screen,trigger_strategy=strategy,groups=groups,rows=rows,pairs=pairs,
        exact_pair_tasks=sum(eligibility.values()),total_tasks=len(eligibility),
        scope='Same-task attempts; total energy includes failed attempts and drain. Do not attribute inexact-initial pairs or failed-task energy differences to efficiency gains.')
    _write_json_atomically(directory/'comparison_summary.json',result)
    with (directory/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# M3四组'+('入口短测' if screen else '正式比较'),'',f'B/C/D统一刷新信号：{strategy}；A固定q5。','','|组别|成功/尝试|平均尝试能量J|每成功任务能量J（含失败）|成功任务平均秒|欠载步|','|---|---:|---:|---:|---:|---:|']
    for g in groups:
        energy='—' if g['energy_per_success_j'] is None else '%.3f'%g['energy_per_success_j']
        seconds='—' if g['mean_success_task_seconds'] is None else '%.3f'%g['mean_success_task_seconds']
        lines.append(f"|{g['group']}|{g['successes']}/{g['attempts']}|{g['mean_attempt_energy_j']:.3f}|{energy}|{seconds}|{g['total_underflow_steps']}|")
    lines+=['',f"初始输入和首次预测严格一致的任务：{sum(eligibility.values())}/{len(eligibility)}。",'',result['scope']]
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def resume_plan(source, settings, config):
    """Audit immutable completed episodes; retain failures and run only missing cells."""
    source=Path(source).resolve()
    if settings.get('comparison_screen'):
        raise ValueError('Resume requires the full comparison')
    if json.loads((source/'comparison_plan.json').read_text())!=settings:
        raise ValueError('Resume comparison plan changed')
    original=json.loads((source/'run.json').read_text())['config_snapshot']
    if {k:v for k,v in original.items() if k not in ('m3_settings','m3_resume_plan')}!=config:
        raise ValueError('Resume model or execution configuration changed')
    record=json.loads((source/'recording.json').read_text())
    from scripts.audit_rtc_system import audit
    allowed={(s,t,g) for s,t in settings['calibration_tasks'] for g in GROUPS}
    seen=set()
    episodes=record['episodes']
    for row in episodes:
        key=(row['suite'],row['task_id'],row['view'])
        if key not in allowed or key in seen:
            raise ValueError('Duplicate or unexpected resume episode')
        if not audit(Path(row['run']))['passed'] or not audit_group_clocks(row['run'],row['view'])['passed']:
            raise ValueError('Completed episode failed resume audit: '+row['run'])
        seen.add(key)
    remaining=[task for task in settings['calibration_tasks']
        if any((task[0],task[1],g) not in seen for g in GROUPS)]
    if not remaining:
        raise ValueError('Comparison already complete')
    return dict(source=str(source),episodes=copy.deepcopy(episodes),remaining_tasks=remaining,
        remaining_episodes=len(allowed)-len(seen),
        previous_model_loads=record.get('previous_model_loads',0)+record['model_loads'])
