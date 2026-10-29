"""Fixed paired tasks and total-task reporting for the M4 acceptance run."""
import csv
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
TASK_IDS = (0, 3, 6)
GROUPS = ('q30', 'joint')


def visual_prefixes(full_lengths, fast_valid, fast_capacity):
    """Task-local tokenizer bounds, not historical task0 prompt lengths."""
    if (not full_lengths or len(full_lengths)>17 or
            full_lengths!=list(range(min(full_lengths),max(full_lengths)+1)) or
            not 512<=min(full_lengths)<=max(full_lengths)<=712 or
            type(fast_valid) is not int or not 320<fast_valid<=320+fast_capacity):
        raise ValueError('Invalid task-local Full/Fast prefix envelope')
    return full_lengths+[fast_valid], dict(kind='task_tokenizer_state_bounds_and_current_stateless_text',
        full=full_lengths,fast=[fast_valid],fast_mode='pool2_agent',fast_state_enabled=False,
        fast_prompt_capacity=fast_capacity,source='current task h50_state_prefix_lengths; no historical task restriction',
        unseen_prefix='abort before expert; no slow fallback or runtime capture')


def render_parity(case, total_pixels):
    """Permit sparse one-level RGB raster differences, never physics drift."""
    budget=max(4,int(total_pixels*0.0001))
    return dict(passed=case['physical_max_abs']==0 and case['state_max_abs']==0
                and case['success_equal'] and case['pixel_max_abs']<=1
                and case['max_changed_pixels']<=budget,
                total_pixels=total_pixels,changed_pixel_limit=budget,
                changed_fraction=case['max_changed_pixels']/total_pixels,
                rule='exact physics/state/success; RGB delta<=1; changed pixels<=0.01% (minimum4)')


@contextmanager
def precompiled_environments(directory, tasks, config):
    """Read validated on-disk models after the preparation process has exited."""
    import os
    from unittest.mock import patch
    from perceptbudget.experiment import _write_json_atomically
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import mujoco
    import robosuite.utils.binding_utils as binding
    directory=Path(directory)
    report=json.loads((directory/'environment_cache.json').read_text())
    expected={tuple(t) for t in tasks}
    if (report.get('status')!='passed' or not report.get('frozen')
            or report.get('version')!=mujoco.__version__
            or {(c['suite'],c['task_id']) for c in report['cases']}!=expected
            or not all(c['passed'] for c in report['cases'])):
        raise ValueError('Require completed matching environment preflight before model loading')
    report['preparation_process_exited']=True
    report['resident_binary_loads']=0

    def factory(cls, xml):
        key=hashlib.sha256((mujoco.__version__+'\0'+xml).encode()).hexdigest()
        path=directory/'environment_models'/(key+'.mjb')
        if not path.is_file():
            raise RuntimeError('Unprepared environment XML; no compilation with resident model: '+key)
        native=mujoco.MjModel.from_binary_path(str(path))
        report['resident_binary_loads']+=1
        return cls(native)

    with patch.object(binding.MjSim,'from_xml_string',classmethod(factory)):
        try:
            yield report
        finally:
            _write_json_atomically(directory/'environment_cache.json',report)


def manifest(config):
    from perceptbudget.baseline import FROZEN
    fixed = {k: config[k] for k in FROZEN}
    fixed.update(seed=7, init=0, tasks=[[s, t] for s in SUITES for t in TASK_IDS],
                 control_hz=20, task_limits='existing suite-specific limits',
                 checkpoint_path=config['checkpoint_path'], norm_stats_path=config['norm_stats_path'])
    fixed['fingerprint'] = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()
    return fixed


def configure_baseline(config):
    from perceptbudget.closed_loop import configure_fixed_baseline
    configure_fixed_baseline(config, 30)
    config.update(m4_enabled=False, m3_hold_high_clocks=True,
                  m2_visual_fast_enabled=False, m2_split_delay=False)
    return 30


def resume_joint(source, config):
    """Retain all audited completed outcomes and run only missing tasks."""
    import copy
    from scripts.audit_rtc_system import audit
    from perceptbudget.m4_planner import POLICY_REVISION
    source=Path(source).resolve()
    original=json.loads((source/'run.json').read_text())['config_snapshot']
    if original.get('m4_policy_revision','legacy-v1')!=config.get('m4_policy_revision',POLICY_REVISION):
        raise ValueError('Joint scheduling version changed; start a new joint group and reuse only q30')
    fixed=manifest(config)
    if (original.get('m4_acceptance_group')!='joint' or original.get('m4_acceptance_manifest')!=fixed
            or original.get('m4_fast_prompt_capacity')!=20
            or original.get('m4_cover_close_slowdown')!=config.get('m4_cover_close_slowdown',False)
            or original.get('m4_settings')!=config.get('m4_settings')):
        raise ValueError('Joint resume requires matching fixed manifest and scheduling configuration')
    record=json.loads((source/'recording.json').read_text())
    seen=set();episodes=record['episodes'];expected={tuple(t) for t in fixed['tasks']}
    for row in episodes:
        key=(row['suite'],row['task_id'])
        if key not in expected or key in seen or row['view']!='full':
            raise ValueError('Unexpected or duplicate completed task in resume record')
        run=Path(row['run'])
        result=json.loads((run/'summary.json').read_text())
        if result.get('status')!='completed' or not audit(run)['passed'] or not clock_audit(run)['passed']:
            raise ValueError('Completed joint episode failed resume audit: '+str(run))
        seen.add(key)
    remaining=[t for t in fixed['tasks'] if tuple(t) not in seen]
    if not remaining:raise ValueError('Joint group is already complete')
    return dict(source=str(source),episodes=copy.deepcopy(episodes),remaining_tasks=remaining,
                previous_model_loads=record.get('previous_model_loads',0)+record['model_loads'])


def calibrated_start(env, task, prepared, np):
    """Use the untouched calibrated reset, without constructing another simulator."""
    if (prepared['suite']!=task['suite'] or prepared['task_id']!=task['task_id']
            or not np.array_equal(prepared['physics'],env.env.sim.get_state().flatten())):
        raise ValueError('Calibrated environment changed before episode start')
    hz=float(env.env.control_freq)
    if hz!=20:raise ValueError('Calibrated start requires unchanged20Hz environment')
    return prepared['observation'],hz


def request_path(run):
    run = Path(run)
    cfg = json.loads((run/'run.json').read_text())['config_snapshot']
    return run/'rtc'/f"q{cfg['candidate_q'][0]}_requests.json"


def clock_audit(run):
    from perceptbudget.device_clock_screen import frequency_check
    run = Path(run)
    lines = (run/'rtc/runtime_clocks.jsonl').read_text().splitlines()
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    requests = json.loads(request_path(run).read_text())
    checks = [frequency_check(events, q.get('m4_compute_profile', 'baseline'),
                             q['observation_monotonic'], q['ready_monotonic']) for q in requests]
    power = json.loads((run/'summary.json').read_text())['episode']['including_drain_power']
    return dict(passed=bool(checks) and all(c['matches'] for c in checks)
                and 'RESTORED' in lines and power['coverage_fraction'] >= 1-1e-9
                and power.get('missing_count', 0) == 0,
                compute_windows=checks, restored='RESTORED' in lines)


def summarize(directory, sources):
    """Keep failures; compare energy savings only for paired successful tasks."""
    import numpy as np
    from perceptbudget.experiment import _write_json_atomically
    directory = Path(directory)
    rows, cases = [], {}
    for group, source in sources.items():
        record = json.loads((Path(source)/'summary.json').read_text())
        if record['status'] != 'completed':
            raise ValueError('Incomplete group: '+group)
        for row in record['episodes']:
            run = Path(row['run'])
            data = json.loads((run/'summary.json').read_text())['episode']
            requests = json.loads(request_path(run).read_text())
            item = dict(suite=row['suite'], task_id=row['task_id'], group=group,
                        outcome=row['outcome'], steps=row['steps'],
                        task_seconds=data['real_seconds'], including_drain_seconds=data['including_drain_seconds'],
                        total_energy_j=data['including_drain_power']['total_energy_mj']/1000,
                        underflow_steps=len(data['underflow_steps']), requests=len(requests), run=str(run))
            rows.append(item)
            cases.setdefault((row['suite'], row['task_id']), {})[group] = (row, requests[0], item)
    pairs = []
    for (suite, tid), groups in cases.items():
        if set(groups) != set(GROUPS):
            raise ValueError('Missing paired result')
        a, b = groups['q30'], groups['joint']
        inputs_equal = (a[0]['initial']['initial_hashes'] == b[0]['initial']['initial_hashes']
                        and all(np.array_equal(a[1][key], b[1][key])
                                for key in ('observation_raw_state', 'model_state'))
                        and np.array_equal(a[1]['rtc_solver_inputs']['noise'], b[1]['rtc_solver_inputs']['noise']))
        exact = inputs_equal and np.array_equal(a[1]['actions_raw'], b[1]['actions_raw'])
        both = a[2]['outcome'] == b[2]['outcome'] == 'success'
        pairs.append(dict(suite=suite, task_id=tid, initial_inputs_equal=bool(inputs_equal),
                          initial_actions_exact=bool(exact), initial_raw_max_abs=float(np.max(np.abs(
                              np.asarray(a[1]['actions_raw'])-np.asarray(b[1]['actions_raw'])))),
                          both_success=both, energy_saving_percent=(100*(1-b[2]['total_energy_j']/a[2]['total_energy_j'])
                              if both and inputs_equal else None),
                          task_time_difference_seconds=b[2]['task_seconds']-a[2]['task_seconds']))
    result = dict(status='completed', rows=rows, pairs=pairs, sources={k:str(v) for k,v in sources.items()},
                  scope='Fixed seed/init; inexact bootstrap actions explicitly reported; failed attempts retained.')
    _write_json_atomically(directory/'comparison.json', result)
    with (directory/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# M4固定任务验收', '', '|集合|任务|组别|结果|步数|任务时间s|整任务能量J|断供步数|',
             '|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['suite']}|{r['task_id']}|{r['group']}|{r['outcome']}|{r['steps']}|{r['task_seconds']:.3f}|{r['total_energy_j']:.3f}|{r['underflow_steps']}|")
    lines += ['', '能量包含结束时在途推理收尾；任务时间与含收尾时间均保存在CSV。', '',
              '|集合|组别|成功/尝试|平均尝试能量J|成功任务平均时间s|', '|---|---|---:|---:|---:|']
    for suite in (*SUITES, 'all'):
        for group in GROUPS:
            subset=[r for r in rows if r['group']==group and (suite=='all' or r['suite']==suite)]
            if not subset: continue
            success=[r for r in subset if r['outcome']=='success']
            seconds=f"{sum(r['task_seconds'] for r in success)/len(success):.3f}" if success else '—'
            lines.append(f"|{suite}|{group}|{len(success)}/{len(subset)}|{sum(r['total_energy_j'] for r in subset)/len(subset):.3f}|{seconds}|")
    lines += ['', f"初始输入相同：{sum(p['initial_inputs_equal'] for p in pairs)}/{len(pairs)}；首次动作精确相同：{sum(p['initial_actions_exact'] for p in pairs)}/{len(pairs)}。", '', result['scope']]
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result
