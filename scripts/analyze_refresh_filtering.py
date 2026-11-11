"""Audit saved refresh filters on development trajectories; no model or rollout.

Read the first actual scheduler decision for every active prediction, including
final chunks with no subsequent request. Outcomes and object contacts are only
offline grouping labels. Candidate eligibility is not a success counterfactual.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.analyze_gripper_timing import INDEX, save
from scripts.audit_rtc_system import latest_commit_at_step, phase_history_at_decision
from scripts.calibrate_motion_cues import filtered
from perceptbudget.action_events import EventConfig
from perceptbudget.action_signal_experiment import prepare as signal_prepare, extract, apply_candidate
from perceptbudget.critical_forecast import forecast

PREDICTION = re.compile(r'"critical_forecast": \{"version": "remaining_v1", "prediction_id": (\d+)')
V3_INDEX = INDEX.parent.parent / '20261110T054642Z-v3-failed25-five-seeds/results.json'
CONTACT_INDEX = INDEX.parent.parent / '20261111T035124Z-object-grasp-contact-old50/results.json'


def read_sources(run):
    sources = {}
    for line in (run / 'rtc/trace.jsonl').open():
        if not line.startswith('{"event": "m3_decision",'):
            continue
        match = PREDICTION.search(line)
        if match and int(match[1]) not in sources:
            decision = json.loads(line)
            sources[int(match[1])] = decision
    return sources


def signature(events):
    return [(e['kind'], e['target_step']) for e in events]


def analyze(row, contacts):
    run = Path(row['run'])
    cfg = json.loads((run / 'run.json').read_text())['config_snapshot']
    if cfg.get('m3_event_coverage_mode') != 'event_only':
        raise ValueError('This audit uses the recorded event_only response policy')
    requests = json.loads((run / 'rtc/q5_requests.json').read_text())
    by_id = {r['prediction_id']: r for r in requests}
    commits = json.loads((run / 'summary.json').read_text())['episode']['commits']
    sources = read_sources(run)
    results = {pid: decision['critical_forecast']['result'] for pid, decision in sources.items()}
    config = EventConfig(**cfg['m3_events'])
    settings = cfg.get('m3_action_signal_settings')
    saved_signals = {}
    if settings:
        saved_signals = {p['prediction_id']: p for p in map(json.loads,
            (run / 'rtc/action_signals.jsonl').read_text().splitlines())}
    mismatches = Counter()
    receipts, packets, signal_rows = [], [], []
    contact = contacts.get(row['run'])
    first_close = contact['closes'][0] if contact and contact['closes'] else None
    for pid, decision in sources.items():
        source = by_id[pid]
        proof = decision['critical_forecast']
        step, now = proof['source_step'], proof['source_monotonic']
        mismatches['source_decision'] += decision['step'] != step or decision['decision_monotonic'] != now
        history = {}
        for old_step in proof['history_steps']:
            previous = by_id[latest_commit_at_step(commits, old_step)['prediction_id']]
            history[old_step] = previous['actions'][old_step - previous['observation_step']]
        offset = step - source['observation_step']
        replay = forecast(source['actions'][offset:offset + proof['result']['remaining_steps']],
            previous_actions=list(history.values()), current_step=step, now=now,
            control_period=proof['source_period_seconds'], config=config)
        mismatches['raw_forecast'] += signature(replay['events']) != signature(proof['result']['events'])
        phases = phase_history_at_decision(commits, by_id, step, now)
        covered = []
        for commit in commits:
            if commit['commit_monotonic'] > now or commit['commit_step'] > step:
                continue
            timing = by_id[commit['prediction_id']].get('m3_timing', {})
            critical = timing.get('critical_step')
            if timing.get('reason') == 'action_event' and critical is not None and commit['commit_step'] <= critical:
                covered.extend(dict(kind=kind, target_step=critical)
                    for kind in timing.get('candidate_event_kinds') or [])
        packet = dict(proof=proof, timing=decision, history=history, phases=phases, covered=covered)
        local = []
        eligible, _, _ = filtered(pid, packet, results, by_id, cfg, config, receipts=local)
        for key, stage in (('refresh_phase_suppressed', 'phase'),
                           ('ordinary_turn_suppressed', 'ordinary_turn'), ('feedback_suppressed', 'feedback')):
            mismatches[key] += signature(decision.get(key, [])) != signature(
                [e for e in local if e['filter_stage'] == stage])
        # The latest committed predecessor is the reference used by production.
        previous = max((c['prediction_id'] for c in commits
            if c['prediction_id'] < pid), default=None)
        reference = (dict(origin=by_id[previous]['observation_step'], actions=by_id[previous]['actions'])
            if previous is not None else None)
        features = {f['target_step']: f for f in extract(source['actions'], origin=source['observation_step'],
            source_step=step, history=history, reference=reference, config=config, within_plan_only=True)}
        response = decision['feedback_review']['response_seconds']
        if settings:
            signal = signal_prepare(actions=source['actions'], origin=source['observation_step'],
                source_step=step, history=history, reference=reference, config=config, settings=settings,
                period=proof['source_period_seconds'], response_seconds=response,
                margin_seconds=decision['margin_seconds'],
                prediction_count=sum(c['commit_monotonic'] <= now for c in commits), covered_phases=phases)
            saved = saved_signals.get(pid)
            mismatches['signal_packet'] += saved is None or signal['review'] != saved['review'] or signal['opportunities'] != saved['opportunities']
            candidate = signal['review']['candidate']
            point_blocked = bool(candidate and any(c['kind'] == candidate['kind']
                and abs(c['target_step'] - candidate['target_step']) <= config.match_steps for c in covered))
            later = [o for o in signal['opportunities'] if o['admitted']
                and candidate and o['target_step'] > candidate['target_step']
                and not any(c['kind'] == 'action_revision' and abs(c['target_step'] - o['target_step']) <= config.match_steps for c in covered)]
            alternative = None
            would_add = False
            if point_blocked and later:
                opportunity = later[0]
                alternative = dict(kind='action_revision', target_step=opportunity['target_step'],
                    evidence=dict(geometry=opportunity['geometry'], disagreement=opportunity['disagreement'],
                        steps=opportunity['evidence_steps']))
                # Replay only this one decision using the original admission
                # helper. Do not invent the resulting future trajectory.
                revised = dict(signal, review=dict(signal['review'], candidate=alternative))
                eligible_with_alternative = apply_candidate(eligible, packet=revised,
                    step=step, end=step+proof['result']['remaining_steps'], prediction_id=pid,
                    covered=covered, tolerance=config.match_steps, period=proof['source_period_seconds'])
                would_add = any(e['kind'] == 'action_revision' for e in eligible_with_alternative)
            signal_rows.append(dict(prediction_id=pid, source_step=step,
                candidate=candidate, recorded_added=decision.get('action_signal_added', False),
                opportunities=signal['opportunities'], point_blocked=point_blocked,
                point_blocked_with_later_candidate=point_blocked and bool(later),
                covering_points=[c for c in covered if candidate and c['kind'] == candidate['kind']
                    and abs(c['target_step']-candidate['target_step']) <= config.match_steps],
                alternative=alternative, alternative_would_add_at_source=would_add,
                existing_earliest=min((e['target_step'] for e in eligible), default=None),
                recorded_critical=decision.get('critical_step'),
                next_actual_observation=by_id.get(pid+1, {}).get('observation_step')))
        period = proof['source_period_seconds']
        for receipt in local:
            target = receipt['target_step']
            receipt.update(source_step=step, observation_step=source['observation_step'],
                lead_from_discovery_steps=target - step,
                recorded_service_budget_seconds=decision['service_seconds'],
                recorded_service_margin_seconds=decision['margin_seconds'],
                seconds_until_target=(target - step)*period,
                feature=features.get(target),
                before_actual_first_close=bool(first_close and step <= target < first_close['close_step']),
                is_first_close_action_source=bool(first_close and first_close['observation_step'] == source['observation_step']))
            receipts.append(receipt)
        expected_critical = min((e['target_step'] for e in eligible), default=None)
        if settings and decision.get('action_signal_added'):
            selected = saved_signals[pid]['review']['candidate']['target_step']
            expected_critical = min(expected_critical, selected) if expected_critical is not None else selected
        mismatches['selected_critical'] += expected_critical != decision.get('critical_step')
        packets.append(dict(prediction_id=pid, source_step=step,
            first_close_step=first_close['close_step'] if first_close else None,
            period_seconds=period, recorded_service_seconds=decision['service_seconds'],
            feedback_window_end=decision['feedback_review']['fresh_plan_end_step']))
    return dict(seed=row['seed'], suite=row['suite'], task_id=row['task_id'], outcome=row['outcome'],
        run=row['run'], mismatches=dict(mismatches), packets=packets, events=receipts, signals=signal_rows)


def summarize(rows):
    events = [event for row in rows for event in row['events']]
    signals = [packet for row in rows for packet in row['signals']]
    blocked = [event for event in events if event['filter_stage'] == 'feedback']
    first_closes = [event for event in events if event['kind'] in ('slowdown', 'turn')
        and event['before_actual_first_close'] and event['is_first_close_action_source']]
    groups = {}
    for outcome in sorted({row['outcome'] for row in rows}):
        subset = [row for row in rows if row['outcome'] == outcome]
        near = [e for row in subset for e in row['events'] if e['kind'] in ('slowdown', 'turn')
            and e['before_actual_first_close'] and e['is_first_close_action_source']]
        groups[outcome] = dict(episodes=len(subset), close_source_motion=len(near),
            close_source_motion_stages=dict(Counter(e['filter_stage'] for e in near)))
    return dict(episodes=len(rows), predictions=sum(len(row['packets']) for row in rows),
        mismatches={key: sum(row['mismatches'].get(key, 0) for row in rows)
            for key in sorted({key for row in rows for key in row['mismatches']})},
        event_stages={kind: dict(Counter(e['filter_stage'] for e in events if e['kind'] == kind))
            for kind in sorted({e['kind'] for e in events})},
        feedback_reasons=dict(Counter(e['feedback']['reason'] for e in blocked)),
        first_close_motion_events=len(first_closes), groups=groups,
        signal_packets=len(signals), signal_opportunities=sum(len(p['opportunities']) for p in signals),
        signal_rejections=dict(Counter(o['reason'] for p in signals for o in p['opportunities'] if not o['admitted'])),
        point_blocked_signals=sum(p['point_blocked'] for p in signals),
        point_blocked_with_later_candidate=sum(p['point_blocked_with_later_candidate'] for p in signals),
        alternative_would_add_at_source=sum(p['alternative_would_add_at_source'] for p in signals),
        alternative_affected_episodes=sum(any(p['alternative_would_add_at_source'] for p in row['signals'])
            for row in rows))


def write_report(root, result, rows):
    lines = ['# 开发记录刷新过滤核查', '',
        '逐回合读取每段预测首次实际调度判断，包括没有后续请求的末段。未加载模型、未运行仿真、未修改正式配置。', '',
        '|记录|回合|预测段|事件及过滤回放差异|', '|---|---:|---:|---:|']
    for group, value in result['groups'].items():
        lines.append(f"|{group}|{value['episodes']}|{value['predictions']}|{sum(value['mismatches'].values())}|")
    lines.extend(['', '## 原200回合的过滤分布', '',
        '|事件|保留|同类点已覆盖|同类阶段已覆盖|普通转向|预测一致/新鲜窗口|',
        '|---|---:|---:|---:|---:|---:|'])
    for kind, counts in result['groups']['old200']['event_stages'].items():
        lines.append('|'+kind+'|'+'|'.join(str(counts.get(k, 0)) for k in
            ('eligible', 'point', 'phase', 'ordinary_turn', 'feedback'))+'|')
    lines.extend(['', '夹爪切换不受普通运动的新鲜窗口过滤；阶段覆盖按事件类型区分，不能把已处理的转向当作闭爪已处理。', '',
        'Object首次闭爪的动作来源中，闭爪前只有4个原始减速/转向信号：成功与失败各2个，'
        '都因与上一预测相符而过滤。9个失败回合中，另外7个在该未执行区没有这样的原始信号。'
        '此处只统计原减速/转向检测，不包括新旧动作差异特征，也不能将过滤自动当作错误。', '',
        '## 新旧动作分歧候选的遍历缺口', '',
        '旧v3开发25回合中，6段预测的首候选命中同类点覆盖；其中5段后面仍有准入候选。'
        '程序在首候选被覆盖后直接返回，未检查后面的候选。按原准入函数只替换当前候选回放，'
        '4段可加入更早截止点、涉及2个回合；另1段已有更早原事件，保持原处理即可。', '',
        '|seed/任务|当前步|被覆盖首候选|覆盖点|后续候选|原关键点|原下次取图步|',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for row in rows['v3_development25']:
        for packet in row['signals']:
            if not packet['point_blocked_with_later_candidate']:
                continue
            covered = ','.join(str(c['target_step']) for c in packet['covering_points'])
            lines.append(f"|{row['seed']}/{row['suite']}/{row['task_id']}|{packet['source_step']}|"
                f"{packet['candidate']['target_step']}|{covered}|{packet['alternative']['target_step']}|"
                f"{packet['recorded_critical']}|{packet['next_actual_observation']}|")
    lines.extend(['', '这证明候选选择存在遗漏，尚未证明修复能救回任务。受影响记录全部为Spatial，'
        '不能用它解释Object抓取失败。当前v4沿用该候选选择函数，正式逻辑尚未修改。', '',
        '建议的最小实验修复：按时间顺序跳过过期、已覆盖的候选，选下一个仍符合原规则的候选；'
        '继续遵守同类阶段去重，已有更早关键点时仍由原事件负责。每段预测仍最多增加一个截止点，'
        '不改阈值、动作、RTC或FlashRT。', '',
        '原200回合不带新旧动作分歧模块；v3的25回合来自旧五组开发失败样本，含2个复测成功。'
        '新25seed评测没有参与本次筛选或调参。这里只比较保存决策，不外推修复后的成功率、能耗或刷新间隔。', ''])
    (root / 'results.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.output or INDEX.parent.parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-refresh-filter-review-development')
    root.mkdir(parents=True, exist_ok=True)
    save('/tmp/perceptbudget-refresh-filter-review-current.json', dict(run=str(root)))
    contacts = {e['run']: e for e in json.loads(CONTACT_INDEX.read_text())['episodes']}
    sources = {'old200': INDEX, 'v3_development25': V3_INDEX}
    all_rows, errors, summaries = {}, [], {}
    print('run_dir=' + str(root), flush=True)
    save(root / 'plan.json', dict(sources={k: str(v) for k, v in sources.items()},
        model_loads=0, new25seed_used=False, production_changed=False))
    for group, index in sources.items():
        rows = []
        original = json.loads(index.read_text())['rows']
        for i, row in enumerate(original, 1):
            try:
                rows.append(analyze(row, contacts))
            except Exception as exc:
                errors.append(dict(group=group, run=row['run'], error=f'{type(exc).__name__}: {exc}'))
            if i % 20 == 0:
                print(f'PROGRESS {group}={i}/{len(original)} errors={len(errors)}', flush=True)
        all_rows[group] = rows
        summaries[group] = summarize(rows)
        save(root / f'{group}_episodes.json', rows)
    result = dict(status='completed' if not errors else 'partial', groups=summaries, errors=errors,
        scope='Saved first decisions and completed responses; no new inference, rollout, parameter selection, or success estimate')
    save(root / 'summary.json', result)
    write_report(root, result, all_rows)
    print('SUMMARY=' + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
