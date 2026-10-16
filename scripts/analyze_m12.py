#!/usr/bin/env python3
"""M1.2：仅离线复用异步正式日志，写独立分析 run。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run
from scripts.analyze_opportunity import _episode, _aggregate, _number

DEFAULT_RUN = Path('/home/wyk/workspace/任务产出/perceptbudget/M1/20261016T082622.503351Z-async-final-edc1a3bc')
NOTES = [
    '注意：q5/8为旧task逐任务加载，q10/12/15为新session复用；同taskgroup字段统计，耗时/能耗差不能仅归因于q。',
    'INA3221 VDD_IN为模块输入（含渲染/控制），不是独立GPU功率。idle为同窗口无推理间隙，设备仍锁频。',
    'd=unionbusy/T，r=P_idle/P_on，S=(1-d)*(1-r)仅为同窗口简化机会估算，不是已实现节能。',
    '约100ms采样对短interval的分割仅为近似；短on/off标定由主agent另测，仅纳入参考。',
    'q15存在断供，不能强行视作可行点；本分析不作质量验收或B_safe判断。',
    '每成功能量=全部尝试窗口总能量/成功数，包含失败成本；不含窗口外加载能量。',
]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def episode(ep, root):
    warnings = []
    artifact = ep.get('artifact_dir')
    request_path = (root / artifact / f"q{ep.get('q')}_requests.json") if artifact else None
    requests = None
    if request_path and request_path.is_file():
        requests = load(request_path)
    else:
        warnings.append('requests缺失')
    start, stop, real = (_number(ep.get(k)) for k in
                         ('start_monotonic', 'stop_monotonic', 'real_seconds'))
    valid = start is not None and stop is not None and stop > start
    if not valid:
        warnings.append('正式时间窗口缺失/无效')
    if real is None:
        warnings.append('real_seconds缺失')
    elif valid and abs(real - (stop-start)) > 1e-5:
        warnings.append('real_seconds与窗口不一致；采用start/stop窗口')
    converted = dict(ep, completed_at_monotonic=stop,
                     real_task_seconds=stop-start if valid else None,
                     inference=dict(requests=requests))
    result = _episode(converted)
    if not result['inference_timeline_complete']:
        warnings.append('推理时间线不完整；d及分区功率不可用')
        result['busy_seconds_known'] = None
        result['busy_power'] = None
    if result['status'] != 'complete':
        warnings.append('时间线或功率覆盖不完整；缺失指标保留null')
    busy = _number(ep.get('inference_busy_seconds'))
    if busy is None:
        warnings.append('inference_busy_seconds缺失；由requests并集重算')
    result.update(suite=ep.get('suite'), taskgroup=ep.get('taskgroup', ep.get('suite')),
                  suite_task_id=ep.get('suite_task_id'), raw_record=ep,
                  requests_reference=reference(request_path) if request_path and request_path.is_file() else None,
                  recorded_inference_busy_seconds=busy,
                  busy_union_minus_recorded_seconds=result['busy_seconds_known']-busy
                  if result['busy_seconds_known'] is not None and busy is not None else None,
                  underflow_steps=len(ep['underflow_steps']) if isinstance(ep.get('underflow_steps'), list) else None,
                  warnings=warnings)
    # Integrate the full window independently so missing requests do not erase known energy.
    whole = _episode(dict(converted, inference={'requests': []}))
    result['energy_j'] = (whole['idle_power'] or {}).get('total_energy_j')
    return result


def aggregate(q, episodes):
    result = _aggregate(q, episodes, len(episodes))
    statuses = [e['source_status'] for e in episodes]
    successes = sum(s == 'success' for s in statuses) if all(s is not None for s in statuses) else None
    energies = [e['energy_j'] for e in episodes]
    energy = sum(energies) if all(v is not None for v in energies) else None
    underflows = [e['underflow_steps'] for e in episodes]
    result.update(successes=successes, energy_j=energy,
                  energy_per_success_j=energy/successes if energy is not None and successes else None,
                  underflow_steps=sum(underflows) if all(v is not None for v in underflows) else None,
                  feasibility='未判定；有断供不得作为可行点' if any(v for v in underflows) else '未判定')
    return result


def markdown(result):
    fmt = lambda v: '—' if v is None else f'{v:.3f}'
    pct = lambda v: '—' if v is None else f'{v*100:.1f}%'
    lines = ['# M1.2 异步功率机会分析', '', *NOTES, '',
             '|q / taskgroup|成功/次数|断供步|P_on W|P_idle W|d|r|same-run S|S_longwindow|能量 J|每成功 J|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['by_q']:
        for label, item in [(str(row['q']), row), *[(f"{row['q']} / {s}", v) for s, v in row['by_suite'].items()]]:
            lines.append(f"|{label}|{item['successes']}/{item['episodes']}|{item['underflow_steps']}|"
                         f"{fmt(item['P_on_w'])}|{fmt(item['P_idle_w'])}|{pct(item['observed_d'])}|"
                         f"{pct(item['r'])}|{pct(item['S'])}|{pct(item.get('S_longwindow'))}|{fmt(item['energy_j'])}|{fmt(item['energy_per_success_j'])}|")
    lines += ['', '逐task原始记录、窗口、区间及覆盖率见analysis.json；阶段/buffer沿用references中的既有async分析。',
              'S_longwindow=(1-d)*(1-r_calib)，仅独立固定scene长窗口估计；无有效标定时留空。',
              '', '警告：', *result['warnings']]
    return '\n'.join(lines)+'\n'


def analyze(async_run=DEFAULT_RUN, output_root=None, calibration_run=None):
    root = Path(async_run).expanduser().resolve(strict=True)
    summary_path = root/'summary.json'
    summary = load(summary_path)
    if summary.get('execution_mode') != 'asynchronous_q5_paired':
        raise ValueError('仅接受正式asynchronous_q5_paired输入')
    episodes, seen = [], set()
    for ep in summary['episodes']:
        if ep.get('execution_mode', summary['execution_mode']) != 'asynchronous_q5_paired' or ep.get('disturbed'):
            raise ValueError('不能混入同步或扰动回合')
        key = (ep.get('taskgroup', ep.get('suite')), ep.get('task_id'), ep.get('q'))
        if key in seen:
            raise ValueError('重复taskgroup/task/q')
        seen.add(key)
        episodes.append(episode(ep, root))
    by_q = []
    for q in sorted({e['q'] for e in episodes}):
        group = [e for e in episodes if e['q'] == q]
        row = aggregate(q, group)
        row['by_suite'] = {str(s): aggregate(q, [e for e in group if e['taskgroup'] == s])
                           for s in dict.fromkeys(e['taskgroup'] for e in group)}
        by_q.append(row)
    refs = {'summary': reference(summary_path), 'async_analysis': None}
    if (root/'analysis.json').is_file():
        refs['async_analysis'] = reference(root/'analysis.json')
        refs['async_analysis']['fields'] = ['episodes[].buffer', 'episodes[].stage_proxy_steps', 'q[].by_suite']
    calibration = None
    if calibration_run:
        path = Path(calibration_run).expanduser().resolve()/'summary.json'
        calibration = dict(reference=reference(path), summary=load(path), interpretation='独立短标定参考；不替换异步分区功率')
        ratio = _number(calibration['summary'].get('r')) if calibration['summary'].get('status') == 'completed' else None
        for row in by_q:
            for item in [row, *row['by_suite'].values()]:
                d = item['observed_d']
                item['S_longwindow'] = (1-d)*(1-ratio) if d is not None and ratio is not None else None
    source_files = ('scripts/analyze_m12.py', 'scripts/analyze_opportunity.py', 'perceptbudget/power.py')
    result = dict(status='complete' if episodes and all(e['status'] == 'complete' for e in episodes) else 'partial',
                  source_sha256={name: reference(ROOT/name)['sha256'] for name in source_files},
                  references=refs, sources=summary.get('sources'), episodes=episodes, by_q=by_q,
                  calibration=calibration, limitations=NOTES,
                  warnings=list(summary.get('warnings', []))+[
                      f"task{e['task_id']}/q{e['q']}: {w}" for e in episodes for w in e['warnings']])
    config = dict(summary['config'])
    if output_root is not None:
        config['output_root'] = str(Path(output_root).expanduser().resolve())
    output = create_run(config, 'M1', 'm12-power')
    for name in source_files:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    (output/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'analysis.md').write_text(markdown(result), encoding='utf-8')
    metadata = load(output/'run.json')
    metadata.update(status='completed', analysis_status=result['status'],
                    source_async_run=str(root), observed_episodes=len(episodes))
    (output/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--async-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--calibration-run', type=Path)
    args = parser.parse_args(argv)
    print(analyze(args.async_run, args.output_root, args.calibration_run))


if __name__ == '__main__':
    main()
