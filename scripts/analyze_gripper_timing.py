"""Read existing development traces; never load a model or create an environment."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from perceptbudget.action_events import EventConfig, detect_events

INDEX = PROJECT.parent / '任务产出/perceptbudget/M4/20261107T133827.733437Z-critical-high-mixed-expansion3-seeds/results.json'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    left = int(index)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (index - left)


def width(state):
    return abs(state[6] - state[7]) if len(state) >= 8 else None


def external_contacts(snapshot):
    pairs = []
    for contact in snapshot.get('contacts') or []:
        names = [contact.get('geom1') or '', contact.get('geom2') or '']
        if (contact.get('distance', 0) <= 0 and any('finger' in n for n in names)
                and any(n and not any(k in n for k in ('robot', 'gripper', 'finger')) for n in names)):
            pairs.append(names)
    return pairs


def grip_events(actions, history, origin, threshold=.2, confirmation=2):
    """Match online gripper logic; keep its two-control history for all candidates.

    Only gripper confirmation changes. Motion cues and their windows are not
    part of this replay. The baseline is compared with saved online forecasts.
    """
    def sign(action):
        return 1 if action[6] >= threshold else -1 if action[6] <= -threshold else 0
    state = next((sign(a) for a in reversed(history[-2:]) if sign(a)), 0)
    result = []
    for i in range(len(actions) - confirmation + 1):
        value = sign(actions[i])
        if value and value != state and all(sign(a) == value for a in actions[i:i + confirmation]):
            if state:
                result.append(('gripper_close' if value > 0 else 'gripper_open', origin + i))
            state = value
    return result


def analyze_episode(row):
    run = Path(row['run'])
    requests = json.loads((run / 'rtc/q5_requests.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())['episode']
    commits = {c['prediction_id']: c for c in summary['commits']}
    by_id = {r['prediction_id']: r for r in requests}
    controls = []
    index_mismatches = action_mismatches = 0
    for line in (run / 'rtc/trace.jsonl').open():
        saved = json.loads(line)
        if saved.get('event') != 'control_step':
            continue
        before, after = saved['before'], saved['after']
        source = saved.get('source') or {}
        pid = source.get('prediction_id')
        if pid is not None and pid in by_id:
            expected = saved['step'] - by_id[pid]['observation_step']
            index_mismatches += source.get('offset') != expected
            if 0 <= expected < len(by_id[pid]['actions']):
                action_mismatches += max(abs(a - b) for a, b in
                    zip(saved['planned_action'], by_id[pid]['actions'][expected])) > 1e-7
        controls.append(dict(step=saved['step'], action=saved['action'], source=source,
            inflight=saved.get('inference_active_at_action'),
            inflight_pid=saved.get('inference_prediction_id_at_action'),
            before_state=before['raw_state'], after_state=after['raw_state'],
            contacts=external_contacts(after),
            objects={k: v for k, v in before.get('observations', {}).items()
                     if k.endswith('_pos') and not k.startswith('robot') and '_to_' not in k}))
    if [r['step'] for r in controls] != list(range(len(controls))):
        raise ValueError('Non-consecutive control recording: ' + str(run))
    proofs = {}
    for request in requests:
        timing = request.get('m3_timing', request.get('m3_fast_decision', {}))
        proof = timing.get('critical_forecast')
        if proof:
            proofs.setdefault(proof['prediction_id'], proof)
    sensitivities = Counter()
    candidates = [('threshold_0.1', .1, 2), ('threshold_0.3', .3, 2),
                  ('threshold_0.5', .5, 2), ('confirmation_1', .2, 1), ('confirmation_3', .2, 3)]
    baseline_mismatches = 0
    for pid, proof in proofs.items():
        start = proof['source_step']
        request = by_id[pid]
        offset = start - request['observation_step']
        remaining = request['actions'][offset:offset + proof['result']['remaining_steps']]
        history = [r['action'] for r in controls[max(0, start - 2):start]]
        baseline = grip_events(remaining, history, start)
        recorded = [(e['kind'], e['target_step']) for e in proof['result']['events'] if e['kind'].startswith('gripper')]
        baseline_mismatches += baseline != recorded
        for name, threshold, confirmation in candidates:
            events = grip_events(remaining, history, start, threshold, confirmation)
            sensitivities[name + '_changed_forecasts'] += events != baseline
            sensitivities[name + '_changed_close_forecasts'] += [e for e in events if e[0] == 'gripper_close'] != [e for e in baseline if e[0] == 'gripper_close']
    g_values = [a[6] for r in requests for a in r['actions']]
    distribution = dict(predicted_actions=len(g_values), deadband_02=sum(abs(g) < .2 for g in g_values),
        abs_01_to_02=sum(.1 <= abs(g) < .2 for g in g_values),
        abs_02_to_03=sum(.2 <= abs(g) < .3 for g in g_values),
        abs_03_to_05=sum(.3 <= abs(g) < .5 for g in g_values),
        near_abs_one_01=sum(abs(abs(g) - 1) <= .1 for g in g_values),
        outside_abs_one=sum(abs(g) > 1 for g in g_values))
    actual = [e for e in detect_events([r['action'] for r in controls], EventConfig()) if e['kind'].startswith('gripper')]
    maximum_width = max(width(r['before_state']) for r in controls)
    closes = []
    for event in actual:
        if event['kind'] != 'gripper_close':
            continue
        step = event['offset']
        control = controls[step]
        pid = control['source'].get('prediction_id')
        request, commit = by_id.get(pid), commits.get(pid)
        if request is None or commit is None:
            raise ValueError('Close without a source prediction')
        hints = [dict(prediction_id=k, known_step=p['source_step'], target_step=e['target_step'])
                 for k, p in proofs.items() if p['source_step'] <= step
                 for e in p['result']['events']
                 if e['kind'] == 'gripper_close' and abs(e['target_step'] - step) <= 5]
        exact = [h for h in hints if h['target_step'] == step]
        refreshes = []
        for req in requests:
            timing = req.get('m3_timing', req.get('m3_fast_decision', {}))
            target = timing.get('critical_step')
            if (req['observation_step'] <= step and target is not None and abs(target - step) <= 5
                    and 'gripper_close' in (timing.get('candidate_event_kinds') or [])):
                c = commits.get(req['prediction_id'])
                refreshes.append(dict(prediction_id=req['prediction_id'], observation_step=req['observation_step'],
                    target_step=target, commit_step=c['commit_step'] if c else None))
        latest = max((r for r in requests if r['observation_step'] <= step), key=lambda r: r['observation_step'])
        after_window = controls[step:min(step + 6, len(controls))]
        close_after = controls[min(step + 3, len(controls) - 1)]
        pos = control['before_state'][:3]
        nearest = min(control['objects'].items(), key=lambda item: math.dist(pos, item[1][:3]), default=None)
        closes.append(dict(seed=row['seed'], suite=row['suite'], task_id=row['task_id'],
            outcome=row['outcome'], close_number=len(closes) + 1, close_step=step,
            confirmed_step=event['confirmed_offset'], source_prediction_id=pid,
            source_observation_step=request['observation_step'], source_commit_step=commit['commit_step'],
            observation_age_steps=step - request['observation_step'],
            adjustment_steps=step - commit['commit_step'],
            close_at_handoff=step - commit['commit_step'] <= 1,
            inference_active_at_close=control['inflight'], pending_prediction_id=control['inflight_pid'],
            latest_observation_step=latest['observation_step'],
            exact_forecast_visible=bool(exact), matched_forecast_visible=bool(hints),
            exact_forecast_advance_steps=step - min(h['known_step'] for h in exact) if exact else None,
            close_refreshes=refreshes,
            finger_external_contact_at_close=bool(control['contacts']),
            finger_external_contact_next5=any(r['contacts'] for r in after_window),
            gripper_width_before=width(control['before_state']),
            gripper_width_after3=width(close_after['after_state']),
            gripper_width_fraction_after3=width(close_after['after_state']) / maximum_width if maximum_width else None,
            xyz_travel_next3_m=math.dist(pos, close_after['after_state'][:3]),
            nearest_object=nearest[0] if nearest else None,
            nearest_object_center_distance_m=math.dist(pos, nearest[1][:3]) if nearest else None,
            nearest_object_scope='nearest recorded object center; not a known grasp target', run=str(run)))
    episode = dict(seed=row['seed'], suite=row['suite'], task_id=row['task_id'], outcome=row['outcome'],
        steps=len(controls), close_events=len(closes), open_events=sum(e['kind'] == 'gripper_open' for e in actual),
        recorded_forecasts=len(proofs), replay_gripper_mismatches=baseline_mismatches,
        source_index_mismatches=index_mismatches, source_action_mismatches=action_mismatches,
        sensitivity=dict(sensitivities), distribution=distribution, first_close=closes[0] if closes else None,
        run=str(run))
    return episode, closes


def group_metrics(episodes, closes):
    first = [r['first_close'] for r in episodes if r['first_close'] is not None]
    return dict(episodes=len(episodes), successes=sum(r['outcome'] == 'success' for r in episodes),
        episodes_with_close=len(first), close_events=len(closes),
        observation_age_median=quantile([r['observation_age_steps'] for r in first], .5),
        adjustment_steps_median=quantile([r['adjustment_steps'] for r in first], .5),
        adjustment_steps_p25=quantile([r['adjustment_steps'] for r in first], .25),
        first_close_at_handoff=sum(r['close_at_handoff'] for r in first),
        first_close_during_inference=sum(bool(r['inference_active_at_close']) for r in first),
        first_close_with_exact_forecast=sum(r['exact_forecast_visible'] for r in first),
        first_close_with_matched_forecast=sum(r['matched_forecast_visible'] for r in first),
        first_close_with_contact=sum(r['finger_external_contact_at_close'] for r in first),
        first_close_with_contact_next5=sum(r['finger_external_contact_next5'] for r in first),
        first_close_nearly_closed_after3=sum(r['gripper_width_fraction_after3'] is not None
            and r['gripper_width_fraction_after3'] <= .05 for r in first))


def report(root, episodes, closes, errors, total, complete=False):
    distribution, sensitivity = Counter(), Counter()
    for row in episodes:
        distribution.update(row['distribution'])
        sensitivity.update(row['sensitivity'])
    groups = {name: group_metrics([r for r in episodes if (r['outcome'] == 'success') == success],
        [r for r in closes if (r['outcome'] == 'success') == success])
        for name, success in [('success', True), ('failure', False)]}
    variants = {}
    for name in ('threshold_0.1', 'threshold_0.3', 'threshold_0.5', 'confirmation_1', 'confirmation_3'):
        variants[name] = dict(changed_forecasts=sensitivity[name + '_changed_forecasts'],
            changed_close_forecasts=sensitivity[name + '_changed_close_forecasts'],
            affected_episodes=sum(r['sensitivity'].get(name + '_changed_forecasts', 0) > 0 for r in episodes),
            affected_close_episodes=sum(r['sensitivity'].get(name + '_changed_close_forecasts', 0) > 0 for r in episodes))
    suites = {suite: {name: group_metrics(
        [r for r in episodes if r['suite'] == suite and (r['outcome'] == 'success') == success],
        [r for r in closes if r['suite'] == suite and (r['outcome'] == 'success') == success])
        for name, success in [('success', True), ('failure', False)]}
        for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')}
    result = dict(status='completed' if complete and not errors else 'partial' if complete else 'running',
        completed=len(episodes), total=total, errors=errors, groups=groups, per_suite=suites,
        predicted_gripper_distribution=dict(distribution), sensitivity=variants,
        recorded_forecasts=sum(r['recorded_forecasts'] for r in episodes),
        replay_gripper_mismatches=sum(r['replay_gripper_mismatches'] for r in episodes),
        source_index_mismatches=sum(r['source_index_mismatches'] for r in episodes),
        source_action_mismatches=sum(r['source_action_mismatches'] for r in episodes),
        scope='Old200 development cohort only; frozen real trajectories; no new inference or simulated outcome; '
              'command/contact/object-center diagnostics do not label premature grasp or successful grasp')
    save(root / 'summary.json', result)
    lines = ['# 旧200回合夹爪时序分析', '', f'已分析{len(episodes)}/{total}回合；状态{result["status"]}。', '',
        '|首次闭爪指标|最终成功回合|最终失败回合|', '|---|---:|---:|']
    for label, key in [('回合数', 'episodes'), ('有闭爪回合', 'episodes_with_close'),
        ('观测距闭爪中位步数', 'observation_age_median'), ('接管到闭爪中位步数', 'adjustment_steps_median'),
        ('接管后0或1步闭爪', 'first_close_at_handoff'), ('闭爪时存在在途推理', 'first_close_during_inference'),
        ('记录中有同一步闭爪预测', 'first_close_with_exact_forecast'),
        ('闭爪时有手指外部接触', 'first_close_with_contact')]:
        lines.append(f'|{label}|{groups["success"][key]}|{groups["failure"][key]}|')
    lines.extend(['', '|离线参数扰动|变化的预测段|涉及回合|', '|---|---:|---:|'])
    for name, value in variants.items():
        lines.append(f'|{name}|{value["changed_forecasts"]}|{value["affected_episodes"]}|')
    lines.extend(['', f'预测动作共{distribution["predicted_actions"]}条，夹爪幅值小于0.2共{distribution["deadband_02"]}条。',
        f'保存的在线预测{result["recorded_forecasts"]}段，夹爪回放不一致{result["replay_gripper_mismatches"]}段；'
        f'执行来源索引不一致{result["source_index_mismatches"]}步，计划动作不一致{result["source_action_mismatches"]}步。', '',
        '闭爪步表示实际执行闭爪指令的开始位置，不等于手指已经物理闭合；物理变化另见开度和接触记录。', '',
        '首次闭爪为主统计，避免失败后反复开合放大样本。接管附近闭爪、手指接触或最近物体中心距离均不能单独证明闭爪过早或抓取成功。'
        '参数扰动只重放旧动作，不能推算改动后的成功率、能耗或刷新频率。', ''])
    if complete:
        lines.extend(['## 结果判断', '',
            '200回合是原M4的开发记录，172成功、28超时；本次没有重测，也没有改变最终v4。', '',
            '在已保存的在线剩余动作预测中，夹爪分界从0.2改0.3或0.5未改变闭爪事件，改0.1只改变1段。'
            '因此不优先将这几个分界值送入闭环扫参。确认窗口1步或3步会改变部分事件，仍需检查具体变化是否对应有用的刷新机会。', '',
            'Object首次闭爪时，成功回合使用的动作源观测距该步中位8步，失败回合18步；'
            '接管到闭爪中位分别2步与7步。失败侧并非普遍只剩更短的调整窗口。'
            '这些相关差异值得结合接近动作和物体运动继续查，不能直接据此设一个全局年龄门槛。', '',
            '接下来优先复核闭爪前的接近、减速、接触和最近有效观测，分清过早闭爪、推偏后闭爪与后续松脱；'
            '保留成功样本作对照，再提出统一的参数或刷新时机候选。', ''])
    (root / 'results.md').write_text('\n'.join(lines))
    if complete:
        save(root / 'episodes.json', episodes)
        save(root / 'closes.json', closes)
        if closes:
            with (root / 'closes.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(closes[0]))
                writer.writeheader(); writer.writerows(closes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=INDEX)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = json.loads(args.index.read_text())['rows']
    root = args.output or INDEX.parent.parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-gripper-timing-old200')
    root.mkdir(parents=True, exist_ok=True)
    save(root / 'plan.json', dict(index=str(args.index), episodes=len(rows), GPU=False, model_loads=0,
        original_config='final v4 unchanged; source trajectories predate action-signal additions'))
    save('/tmp/perceptbudget-gripper-timing-current.json', dict(run=str(root)))
    episodes = json.loads((root / 'episodes.json').read_text()) if (root / 'episodes.json').exists() else []
    closes = json.loads((root / 'closes.json').read_text()) if (root / 'closes.json').exists() else []
    errors = []
    finished = {(r['seed'], r['suite'], r['task_id']) for r in episodes}
    report(root, episodes, closes, errors, len(rows))
    print('run_dir=' + str(root), flush=True)
    for i, row in enumerate(rows, 1):
        if (row['seed'], row['suite'], row['task_id']) in finished:
            continue
        try:
            episode, events = analyze_episode(row)
            episodes.append(episode); closes.extend(events)
        except Exception as exc:
            errors.append(dict(run=row['run'], error=f'{type(exc).__name__}: {exc}'))
        if i % 10 == 0:
            report(root, episodes, closes, errors, len(rows))
            print(f'PROGRESS={i}/{len(rows)} errors={len(errors)}', flush=True)
    report(root, episodes, closes, errors, len(rows), complete=True)
    print('SUMMARY=' + json.dumps(json.loads((root / 'summary.json').read_text()), ensure_ascii=False), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
