"""Offline two-task pilot; no model imports or changes to evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def distance(a, b):
    return math.dist(a, b)


def path_length(points):
    return sum(distance(a, b) for a, b in zip(points, points[1:]))


def switches(records):
    # Sign changes in executed commands, not physical contact events.
    result = []
    previous = None
    for row in records:
        command = row['executed_action'][6]
        sign = 1 if command > 0 else -1 if command < 0 else previous
        if previous is not None and sign != previous:
            result.append({'step': row['control_step'], 'from_sign': previous,
                           'to_sign': sign})
        previous = sign
    return result


def resample(points, count=101):
    # Normalized accumulated path length: descriptive shape comparison only.
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + distance(a, b))
    if not cumulative[-1]:
        return [points[0]] * count
    result = []
    index = 0
    for i in range(count):
        target = cumulative[-1] * i / (count - 1)
        while index + 1 < len(points) - 1 and cumulative[index + 1] < target:
            index += 1
        span = cumulative[index + 1] - cumulative[index]
        alpha = (target - cumulative[index]) / span if span else 0.0
        result.append([a + alpha * (b - a)
                       for a, b in zip(points[index], points[index + 1])])
    return result


def analyze(trace, task_ids, qs):
    records, episodes, fields, events = {}, {}, set(), {}
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line)
            event = row.get('event')
            events[event] = events.get(event, 0) + 1
            if row.get('task_id') not in task_ids or row.get('q') not in qs:
                continue
            key = (row['task_id'], row['q'])
            if event == 'episode_start':
                episodes[key] = row
            if event == 'control_step':
                fields.update(row)
                records.setdefault(key, []).append(row)
    output = []
    for task_id in task_ids:
        baseline = records[(task_id, 5)]
        baseline_points = [r['raw_state'][:3] for r in baseline]
        baseline_shape = resample(baseline_points)
        for q in qs:
            rows = records[(task_id, q)]
            assert [r['control_step'] for r in rows] == list(range(len(rows)))
            assert all(not r['action_underflow'] for r in rows)
            points = [r['raw_state'][:3] for r in rows]
            deviations = [distance(a, b) for a, b in
                          zip(baseline_shape, resample(points))]
            output.append({
                'task_id': task_id, 'q': q,
                'task_name': episodes.get((task_id, q), {}).get('task_name'),
                'success': bool(rows[-1]['env_success']),
                'steps': len(rows),
                'observed_path_m': path_length(points),
                'path_vs_q5_ratio': path_length(points) / path_length(baseline_points),
                'normalized_path_shape_mean_distance_m': sum(deviations) / len(deviations),
                'normalized_path_shape_max_distance_m': max(deviations),
                'executed_gripper_command_switches': switches(rows),
            })
    return {'source_trace': str(trace),
            'source_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
            'control_step_fields': sorted(fields), 'event_counts': events,
            'limitations': [
                'raw_state is pre-action; final post-action position is missing',
                'normalized path-length alignment is descriptive, not phase alignment or ground truth error',
                'timeout paths and successful paths may cover different operations',
                'full predicted chunks, object poses, contact labels and videos are not logged',
                'executed command switches are not predictions or confirmed grasp/contact',
                'prediction error and operation-stage analysis require additional recording',
            ], 'episodes': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, [0, 3], [5, 8, 15])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 同步质量两任务离线试分析', '',
             '任务0：各q均成功；任务3：q=5/8成功、q=15超时。每任务比较q=5/8/15，共6条已有轨迹，不新增GPU运行。', '',
             '|任务|q|成功|步数|已记录路径m|相对q5路径长度|路径形状平均距离cm|夹爪命令切换步|',
             '|---|---|---|---|---|---|---|---|']
    for row in result['episodes']:
        steps = ', '.join(str(e['step']) for e in row['executed_gripper_command_switches']) or '无'
        lines.append(f"|{row['task_id']}|{row['q']}|{row['success']}|{row['steps']}|{row['observed_path_m']:.4f}|{row['path_vs_q5_ratio']:.3f}|{row['normalized_path_shape_mean_distance_m']*100:.2f}|{steps}|")
    lines += ['', '## 口径与缺失数据', '',
              '- 路径长度来自每步动作前的末端位置，缺少最后一步执行后的位置；单位为米。',
              '- 路径形状按累计路径长度归一化后取101点比较，只是描述性差异，不是接触阶段对齐、标准轨迹误差或失败原因。超时与成功轨迹可能覆盖不同操作，尤其不能直接归因。',
              '- 夹爪列只记录已执行命令的正负切换，步号从0开始，不代表实际抓住或接触。',
              '- 现有日志保留原始8维state、执行动作和推理时间，但没有完整20步预测、物体位姿、接触标记或视频。无法由这些日志计算未来关键动作预测误差或可靠阶段划分。',
              '- 下一步仅需对这两个任务补录完整预测、动作前后state、物体/接触信息或视频；保持原同步配置。先验证夹爪切换预测，减速/转向规则另行验证。', '',
              '原始trace：`' + str(args.trace) + '`',
              'SHA256：`' + result['source_sha256'] + '`', '']
    (args.output / '分析.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result['episodes'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
