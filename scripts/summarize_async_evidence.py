"""Offline event-position and buffer evidence; never infer an unmeasured B_safe."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_async_q import (DT, H, QS, buffer_statistics, command_events,
                                     describe, stage_proxy)

KINDS = ('gripper_switch', 'deceleration', 'turn')


def forecast_positions(requests, commits=None):
    """Offsets are zero-based positions in a full raw forecast, not actual contact."""
    positions, incomplete = [], 0
    committed = None if commits is None else {c['prediction_id'] for c in commits}
    for request in requests:
        actions = request.get('actions', [])
        if len(actions) != H:
            incomplete += 1
            continue
        rows = [{'step': i, 'action': action} for i, action in enumerate(actions)]
        for event in command_events(rows):
            positions.append({
                'prediction_id': request['prediction_id'], 'type': event['type'],
                'observation_step': request['observation_step'],
                'is_bootstrap': request['observation_step'] == 0,
                'is_committed': None if committed is None else request['prediction_id'] in committed,
                'forecast_offset': event['step'],
                'lead_nominal_seconds': event['step'] * DT,
                'absolute_forecast_step': request['observation_step'] + event['step']})
    return positions, incomplete


def validate_commit_coordinates(commit):
    """Progress indices differ from elapsed time, but both must fit the same H."""
    origin, start = commit['observation_step'], commit['commit_step']
    elapsed = commit['control_steps_until_commit']
    remaining = commit['new_prediction_remaining_actions']
    alignment = commit['action_alignment']
    low, high = alignment['retained_prediction_offset_interval']
    if (elapsed != start - origin or elapsed < 0 or
            remaining != max(0, H - elapsed) or
            commit['new_prediction_remaining_interval_steps'] != [start, start + remaining] or
            alignment['scheduled_interval_steps'] != [start, start + remaining] or
            (remaining > 0 and start + remaining != origin + H) or
            not 0 <= low <= high <= H or high - low != remaining or
            low != alignment['selected_prediction_offset'] or
            alignment['discarded_prefix_actions'] != low or
            alignment['discarded_tail_actions'] != H - high or
            alignment['discarded_prefix_actions'] + remaining + alignment['discarded_tail_actions'] != H):
        raise ValueError('Inconsistent commit interval or retained prediction coordinates')


def group_evidence(episodes):
    commits = [c for ep in episodes for c in ep['steady_commits']]
    stages = sorted({c['commit_step_post_action_stage_proxy'] for c in commits})
    events = [e for ep in episodes for e in ep['forecast_event_positions']]
    return {
        'episodes': len(episodes),
        'steady_commits': len(commits),
        'bootstrap_remaining_nominal_seconds': describe([
            c['new_valid_seconds'] for ep in episodes for c in ep.get('bootstrap_commits', [])]),
        'stage_proxy_control_steps': {
            stage: sum(ep['stage_proxy_control_steps'].get(stage, 0) for ep in episodes)
            for stage in sorted({s for ep in episodes for s in ep['stage_proxy_control_steps']})},
        'underflow_steps_by_stage_proxy': {
            stage: sum(ep.get('underflow_steps_by_stage_proxy', {}).get(stage, 0) for ep in episodes)
            for stage in sorted({s for ep in episodes for s in ep.get('underflow_steps_by_stage_proxy', {})})},
        'new_remaining_nominal_seconds': describe([c['new_valid_seconds'] for c in commits]),
        'old_queue_remaining_nominal_seconds': describe([c['old_queue_seconds'] for c in commits]),
        'by_commit_step_post_action_stage_proxy': {
            stage: {
                'commits': sum(c['commit_step_post_action_stage_proxy'] == stage for c in commits),
                'tasks': len({ep['task_id'] for ep in episodes
                              if any(c['commit_step_post_action_stage_proxy'] == stage for c in ep['steady_commits'])}),
                'new_remaining_nominal_seconds': describe([
                    c['new_valid_seconds'] for c in commits if c['commit_step_post_action_stage_proxy'] == stage]),
                'old_queue_remaining_nominal_seconds': describe([
                    c['old_queue_seconds'] for c in commits if c['commit_step_post_action_stage_proxy'] == stage])}
            for stage in stages},
        'raw_forecast_event_offsets': {
            kind: describe([e['forecast_offset'] for e in events if e['type'] == kind])
            for kind in KINDS},
        'B_safe_seconds': None}


def analyze(run_dir):
    root = Path(run_dir).resolve(strict=True)
    summary_path, analysis_path = root / 'summary.json', root / 'analysis.json'
    summary, previous = [json.loads(p.read_text()) for p in (summary_path, analysis_path)]
    if summary.get('config', {}).get('H') != H:
        raise ValueError('This analysis requires the fixed H=20 protocol')
    if summary.get('config', {}).get('execution_mode') != 'asynchronous_q5_paired':
        raise ValueError('Expected asynchronous source')
    episodes, seen = [], set()
    for ep in summary['episodes']:
        key = (ep['task_id'], ep['q'])
        if key in seen:
            raise ValueError('Duplicate task/q')
        seen.add(key)
        artifact = root / ep['artifact_dir']
        rows = {}
        with (artifact / 'trace.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('event') == 'control_step' and row.get('q') == ep['q']:
                    if row['step'] in rows:
                        raise ValueError('Duplicate control step')
                    rows[row['step']] = row
        if sorted(rows) != list(range(ep['steps'])):
            raise ValueError('Missing control steps')
        requests = json.loads((artifact / f"q{ep['q']}_requests.json").read_text())
        positions, incomplete = forecast_positions(requests, ep['commits'])
        for commit in ep['commits']:
            validate_commit_coordinates(commit)
        buffer = buffer_statistics(ep['commits'])
        if any(c['h20_consistent'] is not True for c in buffer['records']):
            raise ValueError('Remaining actions inconsistent with H - elapsed steps')
        steady = []
        for c in buffer['records']:
            if c['observation_step'] == 0:
                continue
            c = dict(c)
            row = rows.get(c['commit_step'])
            if row is None and c['commit_step'] != ep['steps']:
                raise ValueError('Commit outside observed execution interval')
            c['commit_step_post_action_stage_proxy'] = stage_proxy(row) if row else 'terminal_boundary_no_action'
            steady.append(c)
        counts, underflow = {}, {}
        for row in rows.values():
            stage = stage_proxy(row)
            counts[stage] = counts.get(stage, 0) + 1
            if row['step'] in ep.get('underflow_steps', []):
                underflow[stage] = underflow.get(stage, 0) + 1
        episodes.append({
            'task_id': ep['task_id'], 'suite': ep['suite'], 'q': ep['q'],
            'status': ep['status'], 'artifact_dir': str(artifact),
            'stage_proxy_control_steps': counts, 'steady_commits': steady,
            'bootstrap_commits': [c for c in buffer['records'] if c['observation_step'] == 0],
            'underflow_steps_by_stage_proxy': underflow,
            'forecast_event_positions': positions, 'incomplete_forecasts': incomplete,
            'B_safe_seconds': None})
    by_q = {}
    for q in QS:
        eps = [ep for ep in episodes if ep['q'] == q]
        by_q[str(q)] = group_evidence(eps)
        by_q[str(q)]['by_suite'] = {
            suite: group_evidence([ep for ep in eps if ep['suite'] == suite])
            for suite in sorted({ep['suite'] for ep in eps})}
        # Existing matching compares forecasts to later executed commands only.
        by_q[str(q)]['raw_prediction_errors_by_type'] = previous['q'][str(q)]['raw_events_by_type']
        by_q[str(q)]['actionable_prediction_errors_by_type'] = previous['q'][str(q)]['actionable_events_by_type']
    return {
        'status': 'partial_evidence_only', 'source_run': str(root),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (summary_path, analysis_path)},
        'q': by_q, 'episodes': episodes,
        'limitations': [
            'No new GPU episodes. Raw forecast errors reference later commands, not synchronous trajectory truth.',
            'Applied-command deceleration/turn events can be created by pending arm gain changes, not model prediction alone.',
            'Forecast offset zero is the first action; no event before the forecast is available to detect a boundary switch.',
            'Buffer distributions pool steady commits; per-task records remain available; bootstrap excluded from buffers only and summarized separately.',
            'Raw event positions include bootstrap and uncommitted forecasts, individually flagged; raw absolute step is not the progress-aligned scheduled step.',
            'Stage proxy uses contacts/motion AFTER the action at commit_step; not exact pre-commit stage or proof that this prediction executed.',
            'Finger contact may be with fixture/support, not a confirmed grasp or transport.',
            'Stable transport and fine placement are unvalidated; external disturbance was not tested.',
            'Remaining nominal time uses control steps times 0.05 s, not measured wall-clock coverage.',
            'No quality threshold or independently validated tail interval; B_safe remains null.']}


def render(result):
    lines = ['# 关键事件与剩余区间：证据补充', '',
             '只复用已完成的异步150次。同步q5作为质量基线；本表事件误差仍指预测与随后实际执行命令的差异，不是同步轨迹真值误差。', '',
             '## 推理后剩余区间', '',
             '按稳态提交池化，排除首次阻塞推理；长回合权重更大，无任务级置信区间。每格为最小 / 中位 / 最大，单位秒。均按控制步×0.05换算，不是墙钟覆盖，也不是质量验证后的安全缓冲。首轮分列JSON。', '',
             '|q|任务数/提交数|新预测剩余区间|旧队列余量|', '|---|---|---|---|']
    def span(stat):
        return '—' if not stat['n'] else ' / '.join(f"{stat[k]:.3f}" for k in ('min', 'median', 'max'))
    for q, data in result['q'].items():
        lines.append(f"|{q}|{data['episodes']}/{data['steady_commits']}|{span(data['new_remaining_nominal_seconds'])}|{span(data['old_queue_remaining_nominal_seconds'])}|")
    lines += ['', '## 原始预测内命令事件位置', '',
              '原始完整预测内的0起始偏移：0表示第一个动作；包含首轮及未提交预测（JSON单列标记）。记录全部事件，不只首次事件；跨预测窗口可能重复覆盖同一实际事件。这不是progress衔接后实际调度位置。', '',
              '|q|夹爪切换偏移|减速偏移|转向偏移|', '|---|---|---|---|']
    for q, data in result['q'].items():
        def offsets(kind):
            d = data['raw_forecast_event_offsets'][kind]
            return '—' if not d['n'] else f"{d['min']:.0f} / {d['median']:.1f} / {d['max']:.0f}（{d['n']}次）"
        lines.append('|' + '|'.join([q] + [offsets(kind) for kind in KINDS]) + '|')
    lines += ['', '## 原始预测与随后执行命令的事件误差', '',
              '误差＝实际事件步−预测事件步；仅匹配同类且相差不超过2步的事件，未匹配项另列，边界及终止尾部按原规则删失。减速/转向可能受等待期gain0.5切换影响，不能孤立归因模型。提交后版本另保存在JSON，存在更强的自我一致性偏差。', '',
              '|q|事件|匹配数|平均误差(步)|平均绝对误差(步)|未匹配预测/实际命令|',
              '|---|---|---:|---:|---:|---|']
    for q, data in result['q'].items():
        for kind, metric in data.get('raw_prediction_errors_by_type', {}).items():
            def mean(name):
                value = metric[name]['mean']
                return '—' if value is None else f'{value:.3f}'
            lines.append(f"|{q}|{kind}|{metric['matched']}|{mean('error_steps')}|{mean('absolute_error_steps')}|{metric['unmatched_predicted']}/{metric['unmatched_actual']}|")
    lines += ['', '## 阶段证据与缺口', '',
              'JSON按任务集合、夹爪接触/非接触运动等proxy分项列出缓冲；标签取commit_step动作执行后的状态，既非提交前精确阶段，也不证明该次预测已执行。这些标签不是已经确认的搬运或精细放置阶段。', '',
              '- 接触：有geom接触证据，但可能接触柜体或支撑物，不能等同于抓稳。',
              '- 平稳搬运、精细放置：尚未可靠标注，不能以低速或成功前最后几步冒充。',
              '- 外部扰动：未测。',
              '- 剩余动作的质量允许区间：尚无预定偏差阈值和独立验证，B_safe保持空值。', '',
              '逐任务、逐提交、分集合与接触proxy统计，及原始/提交后两套预测误差，见 async_evidence.json。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--async-run', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ('async_evidence.json', 'async_evidence.md')):
        parser.error('Evidence files already exist; refusing overwrite')
    result = analyze(args.async_run)
    (args.output_dir / 'async_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'async_evidence.md').write_text(render(result))
    print(args.output_dir / 'async_evidence.md')


if __name__ == '__main__':
    main()
