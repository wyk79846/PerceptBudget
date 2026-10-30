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


def copy_precompiled_environment_cache(source, destination, tasks, config):
    """Reuse an already passed, task-identical MuJoCo binary preflight."""
    import os
    import shutil

    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination:
        raise ValueError('Environment cache source and destination must differ')
    metadata=json.loads((source/'run.json').read_text())
    saved=metadata.get('config_snapshot',{})
    for key in ('m4_acceptance_group','m4_acceptance_manifest'):
        if saved.get(key)!=config.get(key):
            raise ValueError('Environment cache configuration mismatch: '+key)
    report=json.loads((source/'environment_cache.json').read_text())
    expected={tuple(t) for t in tasks}
    cases=report.get('cases',[])
    actual={(c.get('suite'),c.get('task_id')) for c in cases}
    if (report.get('status')!='passed' or not report.get('frozen') or actual!=expected
            or len(cases)!=len(expected) or not all(c.get('passed') for c in cases)):
        raise ValueError('Require a passed environment cache for exactly these tasks')
    src_models=source/'environment_models'
    model_files=list(src_models.glob('*.mjb'))
    if not model_files or len(model_files)>len(expected) or any(p.stat().st_size==0 for p in model_files):
        raise ValueError('Environment cache has missing or invalid binary models')
    if (destination/'environment_cache.json').exists() or (destination/'environment_models').exists():
        raise FileExistsError('Destination already contains an environment cache')
    counts={'hardlinks':0,'copies':0}
    def link_or_copy(src,dst):
        try:
            os.link(src,dst);counts['hardlinks']+=1
        except OSError:
            shutil.copy2(src,dst);counts['copies']+=1
    shutil.copytree(src_models,destination/'environment_models',copy_function=link_or_copy)
    shutil.copy2(source/'environment_cache.json',destination/'environment_cache.json')
    return dict(source=str(source),models=len(model_files),**counts)


def configure_baseline(config):
    from perceptbudget.closed_loop import configure_fixed_baseline
    configure_fixed_baseline(config, 30)
    config.update(m4_enabled=False, m3_fast_enabled=False, m3_hold_high_clocks=True,
                  m2_visual_fast_enabled=False, m2_split_delay=False)
    return 30


def configure_m3(config, settings):
    """Frozen M3 D logic, using the25ms margin selected in its final report."""
    from perceptbudget.m3_comparison import configure_group
    config['m3_formal_signal']='events_only'
    configure_group(config,'D_joint')
    config.update(m4_enabled=False,m3_fast_enabled=False,m3_fast_key_phases=False,m4_policy_revision='not_applicable_original_m3',
                  m2_visual_fast_enabled=False,m3_dvac_enabled=False)
    config.pop('m3_fast_cost_model',None)
    config.pop('m3_fast_policy',None)
    config.pop('m4_settings',None)
    config.pop('m4_fast_prompt_capacity',None)
    settings['mode']='deadline_dvfs'
    settings['timing'].update(maximum_action_offset=50,maximum_observation_age_seconds=None,
        safety_margin_seconds=.025,margin_updates=False)


def configure_m3_fast(config, settings):
    """Only visual selection is added to the original M3 timing/controller."""
    configure_m3(config,settings)
    config.update(m3_fast_enabled=True,m3_fast_selective416=False,m2_fast_no_state=True,m4_fast_prompt_capacity=20,
        m3_fast_policy='m3_full_fast_execution_window_v3')
    from perceptbudget.m3_fast_choice import load_cost_model
    config['m3_fast_cost_model']=load_cost_model(json.loads((Path(__file__).resolve().parents[1]/'m4.json').read_text()))


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


def summarize(directory, sources, *, task_plan=None):
    """Keep failures; compare energy savings only for paired successful tasks."""
    import numpy as np
    from perceptbudget.experiment import _write_json_atomically
    directory = Path(directory)
    if set(sources) not in ({'q30','joint'},{'q30','m3'},{'q30','m3_fast'}):
        raise ValueError('Require q30 and one complete candidate group')
    candidate=next(g for g in sources if g!='q30')
    comparison_groups=('q30',candidate)
    expected={tuple(t) for t in task_plan} if task_plan is not None else None
    if expected is not None and (not expected or len(expected)!=len(task_plan)):
        raise ValueError('Require distinct comparison tasks')
    rows, cases = [], {}
    for group, source in sources.items():
        record = json.loads((Path(source)/'summary.json').read_text())
        if record['status'] != 'completed':
            raise ValueError('Incomplete group: '+group)
        if group==candidate and expected is not None and (
                {(r['suite'],r['task_id']) for r in record['episodes']}!=expected
                or len(record['episodes'])!=len(expected)):
            raise ValueError('Candidate task list does not match the explicit comparison plan')
        for row in record['episodes']:
            if expected is not None and (row['suite'],row['task_id']) not in expected:continue
            run = Path(row['run'])
            data = json.loads((run/'summary.json').read_text())['episode']
            requests = json.loads(request_path(run).read_text())
            item = dict(suite=row['suite'], task_id=row['task_id'], group=group,
                        outcome=row['outcome'], steps=row['steps'],
                        task_seconds=data['real_seconds'], including_drain_seconds=data['including_drain_seconds'],
                        total_energy_j=data['including_drain_power']['total_energy_mj']/1000,
                        underflow_steps=len(data['underflow_steps']), requests=len(requests),
                        full_requests=sum(r.get('context_refresh',{}).get('mode','full')=='full' for r in requests),
                        fast_requests=sum(r.get('context_refresh',{}).get('mode','full')!='full' for r in requests),run=str(run))
            rows.append(item)
            cases.setdefault((row['suite'], row['task_id']), {})[group] = (row, requests[0], item)
    pairs = []
    for (suite, tid), groups in cases.items():
        if set(groups) != set(comparison_groups):
            raise ValueError('Missing paired result')
        a, b = groups['q30'], groups[candidate]
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
    lines = ['# '+('M3同任务对照' if candidate in ('m3','m3_fast') else 'M4固定任务验收'), '', '|集合|任务|组别|结果|步数|任务时间s|整任务能量J|断供步数|',
             '|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['suite']}|{r['task_id']}|{r['group']}|{r['outcome']}|{r['steps']}|{r['task_seconds']:.3f}|{r['total_energy_j']:.3f}|{r['underflow_steps']}|")
    lines += ['', '能量包含结束时在途推理收尾；任务时间与含收尾时间均保存在CSV。', '',
              '|集合|组别|成功/尝试|平均尝试能量J|成功任务平均时间s|', '|---|---|---:|---:|---:|']
    for suite in (*SUITES, 'all'):
        for group in comparison_groups:
            subset=[r for r in rows if r['group']==group and (suite=='all' or r['suite']==suite)]
            if not subset: continue
            success=[r for r in subset if r['outcome']=='success']
            seconds=f"{sum(r['task_seconds'] for r in success)/len(success):.3f}" if success else '—'
            lines.append(f"|{suite}|{group}|{len(success)}/{len(subset)}|{sum(r['total_energy_j'] for r in subset)/len(subset):.3f}|{seconds}|")
    lines += ['', f"初始输入相同：{sum(p['initial_inputs_equal'] for p in pairs)}/{len(pairs)}；首次动作精确相同：{sum(p['initial_actions_exact'] for p in pairs)}/{len(pairs)}。", '', result['scope']]
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def summarize_m3_increment(directory, reference, candidate_result):
    """Per-task comparison with the existing M3 first-measurement ledger."""
    import numpy as np
    directory,reference=Path(directory),Path(reference)
    saved=json.loads(reference.read_text())
    baseline={(r['suite'],r['task_id']):r for r in saved['rows'] if r['group']=='m3'}
    rows=[]
    for r in candidate_result['rows']:
        if r['group']!='m3_fast':continue
        a=baseline[(r['suite'],r['task_id'])]
        old_cfg=json.loads((Path(a['run'])/'run.json').read_text())['config_snapshot']
        new_cfg=json.loads((Path(r['run'])/'run.json').read_text())['config_snapshot']
        old_episode=json.loads((Path(a['run'])/'summary.json').read_text())['episode']
        new_episode=json.loads((Path(r['run'])/'summary.json').read_text())['episode']
        old_first=json.loads(request_path(a['run']).read_text())[0]
        new_first=json.loads(request_path(r['run']).read_text())[0]
        same_inputs=(all(old_episode[k]==new_episode[k] for k in ('initial_sim_state_sha256','initial_frames_sha256'))
            and all(np.array_equal(old_first[k],new_first[k]) for k in ('observation_raw_state','model_state'))
            and np.array_equal(old_first['rtc_solver_inputs']['noise'],new_first['rtc_solver_inputs']['noise']))
        same_timing=old_cfg['m3_timing']==new_cfg['m3_timing']
        rows.append(dict(suite=r['suite'],task_id=r['task_id'],m3_outcome=a['outcome'],fast_outcome=r['outcome'],
            m3_seconds=a['task_seconds'],fast_seconds=r['task_seconds'],m3_j=a['total_energy_j'],fast_j=r['total_energy_j'],
            full_requests=r['full_requests'],fast_requests=r['fast_requests'],same_timing=same_timing,same_initial_inputs=same_inputs,
            bootstrap_raw_max_abs=float(np.max(np.abs(np.asarray(old_first['actions_raw'])-np.asarray(new_first['actions_raw'])))),
            m3_run=a['run'],candidate_run=r['run']))
    result=dict(reference=str(reference),selection='M3 first measurements retained; repeated success does not replace failures',rows=rows)
    from perceptbudget.experiment import _write_json_atomically
    _write_json_atomically(directory/'m3_increment.json',result)
    lines=['# M3 + Fast单项改动对照','','沿用M3首次结果，不挑选成功复测。历史M3跨批次；初始输入和数值计算可能存在差异，不能把每个差异归因于Fast。',
        '', '|任务|M3|M3+Fast|时间s M3/新增|能量J M3/新增|Full/Fast次数|时限配置/初始输入相同|',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"|{r['suite']} {r['task_id']}|{r['m3_outcome']}|{r['fast_outcome']}|{r['m3_seconds']:.3f}/{r['fast_seconds']:.3f}|{r['m3_j']:.3f}/{r['fast_j']:.3f}|{r['full_requests']}/{r['fast_requests']}|{r['same_timing']}/{r['same_initial_inputs']}|")
    (directory/'m3_increment.md').write_text('\n'.join(lines)+'\n')
    return result
