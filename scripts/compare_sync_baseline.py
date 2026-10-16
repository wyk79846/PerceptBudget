"""Offline sync-q5 versus final async comparison; CPU only, limited evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perceptbudget.experiment import create_run
from scripts.analyze_async_q import (
    QS, EVENT_RULES, bootstrap_mean, command_events, compare_rows, paired_success,
)

FIXED = ('H', 'N', 'state_enabled', 'precision', 'model_id', 'flashrt_revision')
RECORDED = ('checkpoint_path', 'model_path', 'norm_stats_path',
            'flashrt_loading_patch_sha256', 'flashrt_tokenizer_patch_sha256')
LIMITS = [
    '仅 task/seed/init/robot-state matched；同步缺少首帧和 sim 指纹，未严格验证全场景初态（固定柜体等不由机器人state确定）。',
    '同步缺少完整预测chunk、物体/接触及帧证据，不能计算同步预测误差或用同步基线替代异步预测分析。',
    '跨运行、同步暂停仿真与异步执行器及task/session生命周期不同；真实耗时差不能仅归因q，也不是节能结论。',
    '同控制步共同前缀是描述性差异；路程归一化仅为形状/相位诊断，不是预测真值误差，无容差判定。',
    '命令事件不是物理抓取真值；不同轨迹的事件位置/counts不代表预测无误差。',
    '外扰未测，阶段未全（同步阶段unknown；异步接触仅proxy），有效缓冲不等于B_safe，M1未全完成。',
    '30任务各一次；配对区间反映任务样本，不涵盖任务内重复随机性。差值均为async减sync q5。',
    '成功率CI仅基于task匹配证据，未严证相同场景；同步未存checkpoint SHA，仅核对当时记录的路径/config及patch SHA，不以现在的文件hash冒充历史权重指纹。',
    '同步无scene，阶段不可双方对齐，baseline阶段为unknown；task20开抽屉、25推盘、27开炉灶不套用搬运/放置阶段。',
]


def read_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_value(key, ep, summary, cfg):
    values = [d[key] for d in (ep, summary, cfg, cfg.get('screen', {})) if key in d]
    if not values or values[0] is None or any(v != values[0] for v in values):
        raise ValueError('missing or inconsistent '+key)
    return values[0]


def load_source(directory, sync):
    root = Path(directory).expanduser().resolve(strict=True)
    run, summary = read_json(root/'run.json'), read_json(root/'summary.json')
    cfg = run['config_snapshot'] if sync else summary['config']
    expected_mode = 'synchronous_quality' if sync else 'asynchronous_q5_paired'
    for obj in (cfg, summary):
        if obj.get('execution_mode', expected_mode) != expected_mode:
            raise ValueError('execution_mode mismatch')
    episodes = {}
    for ep in summary['episodes']:
        if sync and ep['q'] != 5:
            continue
        task, q = ep['task_id'], ep['q']
        if type(task) is not int or task not in range(30) or type(q) is not int or q not in QS:
            raise ValueError('unexpected task/q')
        key = (task, q)
        if key in episodes:
            raise ValueError('duplicate task/q: '+str(key))
        if ep['status'] not in ('success', 'timeout', 'failed', 'failure'):
            raise ValueError('incomplete episode: '+str(key))
        if ep.get('execution_mode', expected_mode) != expected_mode:
            raise ValueError('episode execution_mode mismatch')
        identity = {k: metadata_value(k, {}, {}, cfg) for k in FIXED}
        identity.update({k: cfg.get(k) for k in RECORDED})
        identity.update({k: metadata_value(k, ep, summary, cfg) for k in ('seed', 'init_state_index')})
        episodes[key] = dict(ep=ep, identity=identity, rows=[], sync=sync)
    paths = {}
    for key, item in episodes.items():
        path = root/'trace.jsonl' if sync else root/item['ep']['artifact_dir']/'trace.jsonl'
        paths.setdefault(path.resolve(strict=True), set()).add(key)
    hashes = {}
    for path, keys in paths.items():
        hashes[str(path)] = sha256(path)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if record.get('event') != 'control_step':
                    continue
                key = (record.get('task_id'), record.get('q'))
                if key not in keys:
                    if not sync:
                        raise ValueError('unexpected trace task/q: '+str(key))
                    continue
                row = {'step': record['control_step'], 'action': record['executed_action'],
                       'before': {'raw_state': record['raw_state']}} if sync else {
                           k: record[k] for k in ('step', 'action', 'before', 'after') if k in record}
                rows = episodes[key]['rows']
                if type(row['step']) is not int or row['step'] != len(rows):
                    raise ValueError('duplicate/missing/noncontinuous control step: '+str(key))
                for value, size in ((row['action'], 7), (row['before']['raw_state'], 8)):
                    if len(value) != size or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
                        raise ValueError('invalid action/raw_state: '+str(key))
                rows.append(row)
    for key, item in episodes.items():
        ep, rows = item['ep'], item['rows']
        if not rows or len(rows) != ep['control_steps' if sync else 'steps']:
            raise ValueError('missing control steps: '+str(key))
    provenance = dict(source_run=str(root), sync=sync, config=cfg,
                      run_sha256=sha256(root/'run.json'), summary_sha256=sha256(root/'summary.json'),
                      trace_sha256=hashes, status=summary.get('status'),
                      sources=summary.get('sources', []),
                      missing_evidence=['full_predictions', 'initial_frames', 'initial_sim_fingerprint',
                                        'object_states', 'contacts', 'historical_checkpoint_sha256'] if sync else [],
                      model_lifetime=cfg.get('model_lifetime', 'unrecorded; see original sources'))
    return episodes, provenance


def event_positions(rows):
    events = command_events(rows)
    return {kind: {'count': len(steps), 'steps': steps} for kind in
            ('gripper_switch', 'deceleration', 'turn')
            for steps in [[e['step'] for e in events if e['type'] == kind]]}


def aggregate(pairs):
    mutual = [p for p in pairs if p['sync_success'] and p['async_success']]
    metrics = {}
    for name in ('translation_command_l2', 'rotation_command_l2', 'eef_xyz_distance_m',
                 'gripper_sign_disagreement_fraction', 'path_progress_distance_m',
                 'baseline_coverage', 'candidate_coverage'):
        values = [p['trajectory'][name] for p in pairs]
        metrics[name] = bootstrap_mean([v['mean'] if isinstance(v, dict) else v
                                        for v in values if v is not None])
    return dict(n=len(pairs), sync_successes=sum(p['sync_success'] for p in pairs),
                async_successes=sum(p['async_success'] for p in pairs),
                sync_success_rate=sum(p['sync_success'] for p in pairs)/len(pairs),
                async_success_rate=sum(p['async_success'] for p in pairs)/len(pairs),
                async_underflow_steps=sum(p['async_underflow_steps'] for p in pairs),
                success_difference=paired_success([p['sync_success'] for p in pairs],
                                                  [p['async_success'] for p in pairs]),
                common_success_time_difference={field: bootstrap_mean([p[field] for p in mutual])
                                                for field in ('real_seconds_difference', 'simulated_seconds_difference')},
                trajectory_metrics=metrics)


def compare(sync_runs, async_run, output_root=None):
    baseline, sources = {}, []
    for directory in sync_runs:
        episodes, source = load_source(directory, True)
        if baseline.keys() & episodes.keys():
            raise ValueError('duplicate sync task')
        baseline.update(episodes)
        sources.append(source)
    candidates, source = load_source(async_run, False)
    sources.append(source)
    if set(baseline) != {(t, 5) for t in range(30)} or set(candidates) != {(t, q) for t in range(30) for q in QS}:
        raise ValueError('missing task/q: require sync 30 and async 150 episodes')
    byq = {}
    for q in QS:
        pairs = []
        for task in range(30):
            a, b = baseline[task, 5], candidates[task, q]
            ea, eb = a['ep'], b['ep']
            if a['identity'] != b['identity']:
                raise ValueError('cfg mismatch: '+str((task, q)))
            for key in ('suite', 'task_name'):
                if not ea.get(key) or ea[key] != eb.get(key):
                    raise ValueError(key+' mismatch: '+str((task, q)))
            delta = max(abs(x-y) for x, y in zip(a['rows'][0]['before']['raw_state'], b['rows'][0]['before']['raw_state']))
            if delta > 1e-8:
                raise ValueError('initial robot state mismatch: '+str((task, q)))
            pairs.append(dict(task_id=task, suite=ea['suite'], task_name=ea['task_name'],
                              identity=a['identity'], initial_robot_state_max_abs_delta=delta,
                              evidence='task/seed/init/robot-state matched',
                              sync_success=ea['status']=='success', async_success=eb['status']=='success',
                              async_underflow_steps=len(eb.get('underflow_steps', [])),
                              real_seconds_difference=eb['real_seconds']-ea['real_task_seconds'],
                              simulated_seconds_difference=eb['simulated_seconds']-ea['simulated_control_seconds'],
                              trajectory=compare_rows(a['rows'], b['rows']),
                              command_events={'sync_q5': event_positions(a['rows']),
                                              'async': event_positions(b['rows'])}))
        byq[str(q)] = dict(**aggregate(pairs), tasks=pairs,
                          suites={suite: aggregate([p for p in pairs if p['suite']==suite])
                                  for suite in sorted({p['suite'] for p in pairs})})
    analysis_path = Path(async_run).resolve()/'analysis.json'
    reference = {'available': analysis_path.exists(), 'path': str(analysis_path),
                 'note': '原async预测/缓冲分析仍以其async q5为基线；本报告无sync预测分析。'}
    if analysis_path.exists():
        reference['sha256'] = sha256(analysis_path)
        reference['byq'] = read_json(analysis_path).get('q', {})
    result = dict(status='partial', comparison_baseline='sync_q5',
                  evidence='task/seed/init/robot-state matched',
                  full_scene_initialization_verified=False, pairing_complete=True,
                  sources=sources, limits=LIMITS, event_rules=EVENT_RULES, byq=byq,
                  async_analysis_reference=reference)
    cfg = dict(sources[-1]['config'])
    if output_root is not None:
        cfg['output_root'] = str(Path(output_root).expanduser().resolve())
    output = create_run(cfg, 'M1', 'sync-baseline-comparison', repo_root=ROOT)
    for name, value in [('analysis.json', result)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'analysis.md').write_text(render(result), encoding='utf-8')
    run = read_json(output/'run.json')
    run.update(status='partial', analysis_status='partial', pairing_complete=True,
               comparison_baseline='sync_q5',
               full_scene_initialization_verified=False)
    (output/'run.json').write_text(json.dumps(run, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return output


def render(result):
    lines = ['# 同步q5对最终异步q', '', '状态：partial。150对已配齐；初态仅机器人state证据，全场景未验证，M1未全完成。', '']
    lines.extend('- '+limit for limit in result['limits'])
    drops = {q: 100*(row['sync_success_rate']-row['async_success_rate'])
             for q, row in result['byq'].items()}
    target = [q for q, drop in drops.items() if drop <= 5+1e-10]
    ceiling = [q for q, drop in drops.items() if drop <= 10+1e-10]
    lines += ['', '相对sync q5，async '+ '、'.join(f'q{q}点估计下降{drop:.1f}pp' for q, drop in drops.items())+'。',
              '点估计下降≤5pp目标：'+('、'.join('q'+q for q in target) or '无人满足')+'；点估计下降≤10pp上限：'+('、'.join('q'+q for q in ceiling) or '无人满足')+'。仅为点估计，不能凭30任务CI声称非劣。',
              '异步断供：'+ '、'.join(f"q{q}={row['async_underflow_steps']}步" for q, row in result['byq'].items())+'；质量与供给需同时评估。']
    lines += ['', '成功率差：async−sync q5；paired95CI采用gain/loss精确区间Bonferroni组合。连续量逐任务bootstrap。', '',
              '|q / suite|配对数|sync成功率|async成功率|成功率差 / paired95CI|共同成功真实时间差(s) / 95CI|',
              '|---|---:|---|---|---|---|']
    for q, group in result['byq'].items():
        for suite, row in [('all', group), *group['suites'].items()]:
            success = row['success_difference']
            time = row['common_success_time_difference']['real_seconds_difference']
            lines.append(f"|{q} / {suite}|{row['n']}|{row['sync_successes']}/{row['n']} ({row['sync_success_rate']:.3f})|{row['async_successes']}/{row['n']} ({row['async_success_rate']:.3f})|{success['difference']:.4f} / {success['ci95']}|n={time['n']}, {time['mean']} / {time['ci95']}|")
    lines += ['', '同控制步共同前缀轨迹（均值 / 任务bootstrap95CI）；路程仅共同前缀形状诊断。', '',
              '|q|EEF距离(m)|路程归一化距离(m)|sync coverage|async coverage|',
              '|---|---|---|---|---|']
    for q, row in result['byq'].items():
        cells = [q]
        for key in ('eef_xyz_distance_m', 'path_progress_distance_m', 'baseline_coverage', 'candidate_coverage'):
            metric = row['trajectory_metrics'][key]
            cells.append(f"{metric['mean']} / {metric['ci95']}")
        lines.append('|'+ '|'.join(cells)+'|')
    lines += ['', '逐任务轨迹、coverage及夹爪/减速/转向命令事件的分类counts和位置lists见analysis.json。事件使用既有command_events规则；仅描述执行命令，未声称预测真值或无误差。',
              '', '预测/缓冲仅引用原async analysis的byq，保留路径与SHA；无同步预测数据，不替换其原异步基线。来源run/summary/trace SHA、配置与原异步来源链见JSON。外扰未测、阶段未全。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--async-run', type=Path, required=True)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    print(compare(args.sync_runs, args.async_run, args.output_root))


if __name__ == '__main__':
    main()
