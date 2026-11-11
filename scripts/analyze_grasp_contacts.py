"""Offline Object-suite grasp/contact review, using every old-development case."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.analyze_gripper_timing import INDEX, save, quantile, width
from perceptbudget.action_events import detect_events, EventConfig


def review(row):
    run = Path(row['run'])
    episode = json.loads((run / 'summary.json').read_text())['episode']
    name = episode['task_name']
    match = re.fullmatch('pick_up_the_(.+)_and_place_it_in_the_basket', name)
    if not match:
        raise ValueError('Cannot identify the offline target from Object task text: ' + name)
    stem = match[1] + '_1'
    key = stem + '_pos'
    controls = []
    for line in (run / 'rtc/trace.jsonl').open():
        saved = json.loads(line)
        if saved.get('event') != 'control_step':
            continue
        before, after = saved['before'], saved['after']
        pairs = []
        fingers = set()
        for c in after.get('contacts') or []:
            geoms = [c.get('geom1') or '', c.get('geom2') or '']
            if (c.get('distance', 0) <= 0 and any(n.startswith(stem + '_') for n in geoms)
                    and any('finger' in n for n in geoms)):
                pairs.append(geoms)
                fingers.update(re.findall(r'finger\d+', ' '.join(geoms)))
        controls.append(dict(step=saved['step'], action=saved['action'], source=saved['source'],
            before=saved['before']['raw_state'], after=saved['after']['raw_state'],
            target_before=before['observations'][key], target_after=after['observations'][key],
            basket_before=before['observations']['basket_1_pos'],
            contact=bool(pairs), fingers=len(fingers), pairs=pairs))
    events = [e for e in detect_events([r['action'] for r in controls], EventConfig())
              if e['kind'].startswith('gripper')]
    requests = json.loads((run / 'rtc/q5_requests.json').read_text())
    by_id = {q['prediction_id']: q for q in requests}
    open_width = max(width(r['before']) for r in controls)
    closes = []
    for event in events:
        if event['kind'] != 'gripper_close':
            continue
        step = event['offset']
        end = next((e['offset'] for e in events if e['kind'] == 'gripper_open' and e['offset'] > step), len(controls))
        current = controls[step]
        before = controls[max(0, step - 10):step]
        segment = controls[step:end]
        baseline_z = current['target_before'][2]
        peak = max(segment, key=lambda r: r['target_after'][2])
        later = [r for r in segment if r['step'] >= peak['step']]
        relative_drop = max((peak['target_after'][2] - r['target_after'][2])
            - (peak['after'][2] - r['after'][2]) for r in later)
        pid = current['source']['prediction_id']
        source_step = by_id[pid]['observation_step']
        drift = math.dist(current['target_before'][:2], controls[source_step]['target_before'][:2])
        pre_drift = max((math.dist(current['target_before'][:2], r['target_before'][:2]) for r in before), default=0.)
        contact_steps = [r['step'] for r in before if r['contact']]
        simultaneous = [r['step'] for r in segment if r['fingers'] >= 2]
        release = controls[end] if end < len(controls) else None
        closes.append(dict(seed=row['seed'], task_id=row['task_id'], outcome=row['outcome'],
            close_number=len(closes) + 1, close_step=step, open_step=end if release else None,
            target=stem, observation_step=source_step, observation_age_steps=step - source_step,
            source_mode=by_id[pid].get('context_refresh', {}).get('mode'),
            preclose_target_drift_since_observation_mm=drift * 1000,
            preclose_target_drift_last10_mm=pre_drift * 1000,
            preclose_finger_contact_steps=contact_steps,
            preclose_closed_commands=sum(r['action'][6] >= .2 for r in before),
            eef_target_xy_mm=math.dist(current['before'][:2], current['target_before'][:2]) * 1000,
            eef_above_target_mm=(current['before'][2] - baseline_z) * 1000,
            target_lift_peak_mm=(peak['target_after'][2] - baseline_z) * 1000,
            eef_lift_peak_mm=max(r['after'][2] - current['before'][2] for r in segment) * 1000,
            closed_relative_downward_mm=relative_drop * 1000,
            intermediate_open_commands=sum(r['action'][6] <= -.2 for r in segment),
            both_fingers_contact_steps=simultaneous,
            min_width_fraction=min(width(r['after']) for r in segment) / open_width if open_width else None,
            release_target_basket_xy_mm=math.dist(release['target_before'][:2], release['basket_before'][:2]) * 1000 if release else None,
            release_target_lift_mm=(release['target_before'][2] - baseline_z) * 1000 if release else None,
            run=str(run)))
    return dict(seed=row['seed'], task_id=row['task_id'], outcome=row['outcome'], task_name=name,
        target=stem, steps=len(controls), closes=closes, run=str(run),
        episode_max_target_lift_mm=max(r['target_after'][2] - controls[0]['target_before'][2] for r in controls) * 1000)


def aggregate(episodes):
    result = {}
    for success, label in ((True, 'success'), (False, 'failure')):
        selected = [e for e in episodes if (e['outcome'] == 'success') == success]
        first = [e['closes'][0] for e in selected if e['closes']]
        result[label] = dict(episodes=len(selected), first_closes=len(first),
            observation_age_median=quantile([r['observation_age_steps'] for r in first], .5),
            preclose_observation_target_drift_mm_median=quantile([r['preclose_target_drift_since_observation_mm'] for r in first], .5),
            preclose_target_contact=sum(bool(r['preclose_finger_contact_steps']) for r in first),
            preclose_contact_and_target_drift10mm=sum(bool(r['preclose_finger_contact_steps'])
                and r['preclose_target_drift_last10_mm'] >= 10 for r in first),
            first_close_target_lift_mm_median=quantile([r['target_lift_peak_mm'] for r in first], .5),
            first_close_lift_at_least20mm=sum(r['target_lift_peak_mm'] >= 20 for r in first),
            first_close_both_fingers_contact=sum(bool(r['both_fingers_contact_steps']) for r in first),
            first_close_lift_then_relative_drop20mm=sum(r['target_lift_peak_mm'] >= 20
                and r['closed_relative_downward_mm'] >= 20 for r in first),
            no20mm_lift_whole_episode=sum(e['episode_max_target_lift_mm'] < 20 for e in selected),
            multiple_closures=sum(len(e['closes']) > 1 for e in selected))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [r for r in json.loads(INDEX.read_text())['rows'] if r['suite'] == 'libero_object']
    root = args.output or INDEX.parent.parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-object-grasp-contact-old50')
    root.mkdir(parents=True, exist_ok=True)
    save('/tmp/perceptbudget-grasp-contact-current.json', dict(run=str(root)))
    save(root / 'plan.json', dict(index=str(INDEX), selected_suite='libero_object', episodes=len(rows),
        scope='Every Object episode from the old200 cohort; task text used only for offline target identification; '
              '10mm/20mm are descriptive reporting cuts, not online decision thresholds', model_loads=0))
    episodes = []
    print('run_dir=' + str(root), flush=True)
    for i, row in enumerate(rows, 1):
        episodes.append(review(row))
        save(root / 'progress.json', dict(completed=i, total=len(rows)))
        if i % 10 == 0:
            print(f'PROGRESS={i}/{len(rows)}', flush=True)
    groups = aggregate(episodes)
    result = dict(status='completed', completed=len(episodes), total=len(rows), groups=groups,
        episodes=episodes, scope='Offline physical evidence only. Object task identity is not a runtime input. '
        'Contact, lift, or center distance alone does not prove a stable grasp or its failure cause.')
    save(root / 'results.json', result)
    lines = ['# Object旧50回合抓取与接触分析', '', '覆盖旧开发集全部50回合：41成功、9超时。仅使用已有轨迹，没有模型推理或仿真重跑。', '',
        '|首次闭爪指标|成功回合|失败回合|', '|---|---:|---:|']
    for title, key in [('回合数', 'episodes'), ('观测年龄中位步数', 'observation_age_median'),
        ('从观测到闭爪目标平移中位mm', 'preclose_observation_target_drift_mm_median'),
        ('闭爪前10步有手指目标接触', 'preclose_target_contact'),
        ('闭爪前接触且目标移动至少10mm', 'preclose_contact_and_target_drift10mm'),
        ('首次闭爪后目标最高抬升中位mm', 'first_close_target_lift_mm_median'),
        ('首次闭爪后抬升至少20mm', 'first_close_lift_at_least20mm'),
        ('首次闭爪段双指同时接触目标', 'first_close_both_fingers_contact'),
        ('抬升后闭爪期间相对手臂下落至少20mm', 'first_close_lift_then_relative_drop20mm'),
        ('整个回合目标未抬升20mm', 'no20mm_lift_whole_episode'), ('多次闭爪回合', 'multiple_closures')]:
        lines.append(f'|{title}|{groups["success"][key]}|{groups["failure"][key]}|')
    lines.extend(['', '10mm/20mm仅用于描述物理轨迹，不是新调度参数；接触、抬升或相对下落不能单独确认抓稳或唯一失败原因。', '',
        '闭爪段在下一次开爪指令前结束；闭爪期间的相对下落扣除手臂自身下降，避免把正常下降和开爪放置直接当作滑落。', '',
        '|失败seed/task|首次闭爪步|观测年龄|闭爪前接触|目标平移mm|首次抬升mm|抬起后相对下落mm|闭爪次数|',
        '|---|---:|---:|---|---:|---:|---:|---:|'])
    for episode in episodes:
        if episode['outcome'] == 'success' or not episode['closes']:
            continue
        r = episode['closes'][0]
        drop = f'{r["closed_relative_downward_mm"]:.2f}' if r['target_lift_peak_mm'] >= 20 else '--'
        lines.append(f'|{episode["seed"]}/{episode["task_id"]}|{r["close_step"]}|{r["observation_age_steps"]}|'
            f'{bool(r["preclose_finger_contact_steps"])}|{r["preclose_target_drift_since_observation_mm"]:.2f}|'
            f'{r["target_lift_peak_mm"]:.2f}|{drop}|{len(episode["closes"])}|')
    lines.extend(['', '物体没有先抬起20mm时，不把“手臂抬起但物体留在桌上”记作滑落，表中以--表示。', '',
        '## 结果判断', '',
        '9个失败回合中，7个全程目标抬升不足20mm；另外2个曾抬起，再在下一次开爪之前回落。'
        '逐步核对这2个样本，掉落期间没有中间开爪指令：seed67/task2在61步达到峰值，63步已下降超过20mm，111步才开爪；'
        'seed109/task5在76步达到峰值，78步已下降超过20mm，123步才开爪。两例下降时机械臂仍向上，指令仍为闭爪。', '',
        '失败侧8/9在闭爪前10步已有手指与目标接触，但没有一例在这10步内目标平移达到10mm。'
        '成功与失败从动作源观测到闭爪的目标平移中位均约5mm；本批证据不支持把“大幅推走目标”作为主要解释，细小位置变化仍可能影响夹取。', '',
        '同任务对照也说明观测年龄不能单独解释失败：牛奶盒成功2回合的观测年龄为17、18步，失败3回合为16、17、16步。'
        '其闭爪时末端高于目标中心的中位数为成功约45mm、失败约63mm，值得核查尚未到位就闭爪；'
        '末端坐标点、手指姿态和物体中心不同，不能把这个高度直接变成统一线上阈值。', '',
        '后续参数标定仍使用全200回合的通用动作与state信号，重点核对抓取前接近阶段的减速/转向判断及抓取后的执行反馈；'
        '这些离线目标位姿与任务名称不进入线上调度，不修改最终v4。', '',
        '[已保存图像的成功/失败对比](grasp_examples.png) · [两例闭爪中掉落的逐步证据](closed_drop_evidence.json)', ''])
    (root / 'results.md').write_text('\n'.join(lines) + '\n')
    print('SUMMARY=' + json.dumps(groups, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
