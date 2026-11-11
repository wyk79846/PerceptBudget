"""Compare motion-cue thresholds on frozen old200 plans, without new inference.

Completed-response history is held at the recorded value. These conditional
cue counts are not a replay of future rollouts, energy, or success rates.
"""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.analyze_gripper_timing import INDEX, save, quantile
from scripts.audit_rtc_system import phase_history_at_decision, latest_commit_at_step
from perceptbudget.action_events import EventConfig
from perceptbudget.critical_forecast import forecast
from perceptbudget.critical_wait import covered_by_phase
from perceptbudget.event_identity import review_turn, review_ordinary_turn
from perceptbudget.feedback_refresh import review_cues
from perceptbudget.key_phases import absolute_intervals

VARIANTS = {'baseline': {}, 'slowdown_035': {'slowdown_ratio': .35},
    'slowdown_065': {'slowdown_ratio': .65}, 'slowdown_080': {'slowdown_ratio': .8},
    'turn_45': {'turn_degrees': 45.}, 'turn_75': {'turn_degrees': 75.},
    'turn_90': {'turn_degrees': 90.}, 'motion_005': {'minimum_motion': .05},
    'motion_015': {'minimum_motion': .15},
    'approach_slowdown_065': {'slowdown_ratio': .65, 'scope': 'predicted_grasp_phase_only'},
    'approach_slowdown_080': {'slowdown_ratio': .8, 'scope': 'predicted_grasp_phase_only'}}


def prepare(row):
    run = Path(row['run'])
    cfg = json.loads((run / 'run.json').read_text())['config_snapshot']
    requests = json.loads((run / 'rtc/q5_requests.json').read_text())
    commits = json.loads((run / 'summary.json').read_text())['episode']['commits']
    by_id = {r['prediction_id']: r for r in requests}
    packets = {}
    for request in requests:
        timing = request.get('m3_timing', request.get('m3_fast_decision', {}))
        proof = timing.get('critical_forecast')
        if not proof or proof['prediction_id'] in packets:
            continue
        pid, start = proof['prediction_id'], proof['source_step']
        history = {}
        for step in proof['history_steps']:
            c = latest_commit_at_step(commits, step)
            r = by_id[c['prediction_id']]
            history[step] = r['actions'][step - r['observation_step']]
        phases = phase_history_at_decision(commits, by_id, start, proof['source_monotonic'])
        covered = []
        for commit in commits:
            if commit['commit_monotonic'] > proof['source_monotonic'] or commit['commit_step'] > start:
                continue
            t = by_id[commit['prediction_id']].get('m3_timing', {})
            if (t.get('reason') == 'action_event' and t.get('critical_step') is not None
                    and commit['commit_step'] <= t['critical_step']):
                covered.extend(dict(kind=k, target_step=t['critical_step'])
                               for k in t.get('candidate_event_kinds') or [])
        packets[pid] = dict(proof=proof, timing=timing, history=history, phases=phases, covered=covered)
    return cfg, by_id, packets


def filtered(pid, packet, results, by_id, cfg, settings, *, receipts=None):
    proof, timing = packet['proof'], packet['timing']
    source = by_id[pid]
    events = [dict(e, prediction_id=pid) for e in results[pid]['events']]
    phases = absolute_intervals(source['actions'], source['observation_step'], settings,
        events=[dict(kind=e['kind'], offset=e['target_step'] - source['observation_step']) for e in events])
    ordinary = set()
    if cfg.get('m3_ordinary_turn_policy'):
        for event in events:
            if event['kind'] != 'turn':
                continue
            r = review_ordinary_turn(event, actions=source['actions'], origin=source['observation_step'],
                source_step=proof['source_step'], history=packet['history'], phases=phases,
                events=events, config=settings, approach_delta=cfg.get('m3_full_approach_delta'))
            if r['ordinary']:
                ordinary.add(event['target_step'])
    reference_id = timing.get('feedback_review', {}).get('reference_prediction_id')
    reference = None
    if reference_id in results:
        reference = dict(prediction_id=reference_id, origin=by_id[reference_id]['observation_step'],
                         actions=by_id[reference_id]['actions'], events=results[reference_id]['events'])
    ignored = set()
    feedback_events = {}
    if cfg.get('m3_feedback_policy'):
        old = timing['feedback_review']
        review = review_cues(events, prediction_id=pid, origin=source['observation_step'],
            source_step=proof['source_step'], period=proof['source_period_seconds'],
            response_seconds=old['response_seconds'], margin_seconds=timing['margin_seconds'],
            minimum_steps=cfg['m3_feedback_minimum_steps'], actions=source['actions'],
            reference=reference, config=settings, policy=cfg['m3_feedback_policy'])
        ignored = {(e['kind'], e['target_step']) for e in review['events'] if e['suppress']}
        feedback_events = {(e['kind'], e['target_step']): e for e in review['events']}
    chunks = {k: dict(origin=r['observation_step'], phase_actions=r['actions']) for k, r in by_id.items()}
    surviving = []
    rejected = Counter()
    def record(event, stage, **details):
        if receipts is not None:
            receipts.append(dict(event, filter_stage=stage, **details))
    for event in events:
        points = [e for e in packet['covered'] if e['kind'] == event['kind']
            and abs(e['target_step'] - event['target_step']) <= settings.match_steps]
        if points:
            record(event, 'point', covered_points=points)
            rejected['point'] += 1; continue
        covering = [p for p in packet['phases'] if covered_by_phase(event, p)]
        restored = False
        if covering:
            restored = (event['kind'] == 'turn' and cfg.get('m3_event_identity')
                        and not review_turn(event, covering, chunks, settings)['same_event'])
            if not restored:
                record(event, 'phase', covered_phases=covering)
                rejected['phase'] += 1; continue
        if event['kind'] == 'turn' and event['target_step'] in ordinary and not restored:
            record(event, 'ordinary_turn')
            rejected['ordinary_turn'] += 1; continue
        if (event['kind'], event['target_step']) in ignored:
            record(event, 'feedback', feedback=feedback_events[(event['kind'], event['target_step'])])
            rejected['feedback'] += 1; continue
        record(event, 'eligible')
        surviving.append(event)
    return surviving, phases, dict(rejected)


def analyze(row):
    cfg, requests, packets = prepare(row)
    base = EventConfig(**cfg['m3_events'])
    all_results, outcomes, mismatches = {}, {}, 0
    for name, changes in VARIANTS.items():
        settings = replace(base, **{k: v for k, v in changes.items() if k != 'scope'})
        results = {}
        for pid, packet in packets.items():
            proof = packet['proof']; start = proof['source_step']; r = requests[pid]
            offset = start - r['observation_step']
            results[pid] = forecast(r['actions'][offset:offset + proof['result']['remaining_steps']],
                previous_actions=list(packet['history'].values()), current_step=start,
                now=proof['source_monotonic'], control_period=proof['source_period_seconds'], config=settings)
            if changes.get('scope'):
                original = all_results['baseline'][pid]
                phases = absolute_intervals(r['actions'], r['observation_step'], base,
                    events=[dict(kind=e['kind'], offset=e['target_step'] - r['observation_step']) for e in original['events']])
                closes = [e['target_step'] for e in original['events'] if e['kind'] == 'gripper_close']
                additions = []
                existing = {(e['kind'], e['target_step']) for e in original['events']}
                for phase in phases:
                    targets = [t for t in closes if phase['start_step'] <= t < phase['stop_step']]
                    if not targets:
                        continue
                    before_close = min(targets)
                    selected = [e for e in results[pid]['events'] if e['kind'] == 'slowdown'
                        and phase['start_step'] <= e['target_step'] < before_close
                        and r['actions'][e['target_step'] - r['observation_step']][6] <= -base.gripper_threshold
                        and (e['kind'], e['target_step']) not in existing]
                    if selected:
                        additions.append(dict(min(selected, key=lambda e: e['target_step']),
                            scope='predicted_grasp_phase_only'))
                results[pid] = dict(original, events=sorted(original['events'] + additions,
                    key=lambda e: (e['target_step'], e['kind'])))
            if name == 'baseline':
                signature = lambda events: [(e['kind'], e['target_step']) for e in events]
                mismatches += signature(results[pid]['events']) != signature(proof['result']['events'])
        all_results[name] = results
        outcomes[name] = {pid: filtered(pid, p, results, requests, cfg,
            base if changes.get('scope') else settings) for pid, p in packets.items()}
    counts = {name: Counter() for name in VARIANTS}
    changes = []
    for pid, packet in packets.items():
        base_events, base_phases, _ = outcomes['baseline'][pid]
        original_first = min((e['target_step'] for e in base_events), default=math.inf)
        close_targets = [e['target_step'] for e in all_results['baseline'][pid]['events'] if e['kind'] == 'gripper_close']
        grasp_phases = [p for p in base_phases if any(p['start_step'] <= t < p['stop_step'] for t in close_targets)]
        original_signature = [(e['kind'], e['target_step']) for e in base_events]
        for name in VARIANTS:
            events, _, rejected = outcomes[name][pid]
            c = counts[name]
            raw = all_results[name][pid]['events']
            c['raw_slowdown'] += sum(e['kind'] == 'slowdown' for e in raw)
            c['raw_turn'] += sum(e['kind'] == 'turn' for e in raw)
            c['surviving_motion'] += sum(e['kind'] in ('slowdown', 'turn') for e in events)
            c['changed_forecasts'] += [(e['kind'], e['target_step']) for e in events] != original_signature
            first = min((e['target_step'] for e in events), default=math.inf)
            if first < original_first:
                c['earlier_forecasts'] += 1
                related = any(p['start_step'] <= first < p['stop_step']
                    and any(t >= first and p['start_step'] <= t < p['stop_step'] for t in close_targets) for p in grasp_phases)
                c['earlier_grasp_phase' if related else 'earlier_other_phase'] += 1
                changes.append(dict(seed=row['seed'], suite=row['suite'], task_id=row['task_id'],
                    outcome=row['outcome'], variant=name, prediction_id=pid,
                    source_step=packet['proof']['source_step'],
                    original_first=None if math.isinf(original_first) else original_first,
                    candidate_first=first, earlier_grasp_phase=related,
                    close_targets=close_targets, run=row['run']))
            c['later_forecasts'] += first > original_first
    return dict(seed=row['seed'], suite=row['suite'], task_id=row['task_id'], outcome=row['outcome'],
        packets=len(packets), baseline_mismatches=mismatches,
        counts={name: dict(c) for name, c in counts.items()}, changes=changes)


def report(root, rows, errors, total, complete=False):
    totals = {name: Counter() for name in VARIANTS}
    for row in rows:
        for name, counts in row['counts'].items():
            totals[name].update(counts)
    result = dict(status='completed' if complete and not errors else 'partial' if complete else 'running',
        completed=len(rows), total=total, errors=errors,
        packets=sum(r['packets'] for r in rows), baseline_mismatches=sum(r['baseline_mismatches'] for r in rows),
        variants={name: dict(parameters=VARIANTS[name], **dict(counts),
            affected_episodes=sum(r['counts'][name].get('changed_forecasts', 0) > 0 for r in rows)) for name, counts in totals.items()},
        scope='Conditional cue comparison at original forecast discovery points, with recorded completed-response history; '
              'no new trajectories, service-time plans, success estimates, or energy estimates')
    for name, value in result['variants'].items():
        changed = [item for row in rows for item in row['changes'] if item['variant'] == name]
        advances = [item['original_first'] - item['candidate_first'] for item in changed
            if item['original_first'] is not None]
        lead = [min(target - item['candidate_first'] for target in item['close_targets']
                    if target >= item['candidate_first']) for item in changed
            if item['earlier_grasp_phase']]
        affected = [row for row in rows if row['counts'][name].get('earlier_forecasts', 0)]
        value.update(earlier_by_suite={suite: dict(Counter(row['outcome'] for row in affected
            if row['suite'] == suite)) for suite in sorted({row['suite'] for row in rows})},
            advance_steps_median=quantile(advances, .5),
            grasp_lead_steps_median=quantile(lead, .5))
    save(root / 'summary.json', result)
    lines = ['# 旧200回合减速与转向参数比较', '', f'完成{len(rows)}/{total}，状态{result["status"]}。', '',
        '保留原轨迹及已完成响应历史，重算事件、阶段、普通转向过滤与反馈过滤。原参数回放事件应与保存记录一致。', '',
        '|候选|原始减速|原始转向|过滤后运动事件|提前的预测段|其中闭爪前原阶段|其他阶段提前|受影响回合|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in result['variants'].items():
        lines.append('|'+name+'|'+'|'.join(str(value.get(k, 0)) for k in ('raw_slowdown', 'raw_turn',
            'surviving_motion', 'earlier_forecasts', 'earlier_grasp_phase', 'earlier_other_phase', 'affected_episodes'))+'|')
    lines.extend(['', f'保存预测段{result["packets"]}，原事件回放不一致{result["baseline_mismatches"]}。', '',
        '提前的候选是旧动作上的机会，不代表实际增加同样次数的推理。闭爪前阶段使用原预测的运动区间与闭爪位置标识，'
        '不是物体接触真值。这里不预测成功率、总能量或实际刷新间隔；这些需要候选闭环对照。', ''])
    (root / 'results.md').write_text('\n'.join(lines))
    if complete:
        save(root / 'episodes.json', rows)


def inspect_grasp_sources(root, source_index, grasp_index):
    """Locate raw slowdown cues in the plan that supplied each first close.

    Physical episode labels are used only to summarize the saved trajectories.
    This deliberately does not treat a raw motion cue as a scheduler decision.
    """
    original = {row['run']: row for row in source_index}
    contacts = json.loads(grasp_index.read_text())['episodes']
    rows = []
    for episode in contacts:
        if not episode['closes']:
            continue
        first = episode['closes'][0]
        cfg, requests, packets = prepare(original[episode['run']])
        base = EventConfig(**cfg['m3_events'])
        sources = [pid for pid, packet in packets.items()
            if requests[pid]['observation_step'] == first['observation_step']
            and packet['proof']['source_step'] <= first['close_step']]
        if not sources:
            continue
        pid = min(sources, key=lambda key: packets[key]['proof']['source_step'])
        packet, source = packets[pid], requests[pid]
        proof = packet['proof']
        offset = proof['source_step'] - source['observation_step']
        row = dict(seed=episode['seed'], task_id=episode['task_id'], outcome=episode['outcome'],
            close_step=first['close_step'], prediction_id=pid, observation_step=first['observation_step'],
            forecast_discovery_step=proof['source_step'], period_seconds=proof['source_period_seconds'],
            response_seconds=packet['timing'].get('feedback_review', {}).get('response_seconds'),
            run=episode['run'], candidates={})
        for ratio in (.5, .65, .8):
            result = forecast(source['actions'][offset:offset + proof['result']['remaining_steps']],
                previous_actions=list(packet['history'].values()), current_step=proof['source_step'],
                now=proof['source_monotonic'], control_period=proof['source_period_seconds'],
                config=replace(base, slowdown_ratio=ratio))
            candidates = [e for e in result['events']
                if e['kind'] == 'slowdown' and e['target_step'] < first['close_step']]
            nearest = max(candidates, key=lambda e: e['target_step'], default=None)
            row['candidates'][str(ratio)] = dict(
                nearest_slowdown_step=nearest['target_step'] if nearest else None,
                lead_steps=first['close_step'] - nearest['target_step'] if nearest else None)
        rows.append(row)
    groups = {}
    for outcome in sorted({row['outcome'] for row in rows}):
        group = [row for row in rows if row['outcome'] == outcome]
        groups[outcome] = dict(episodes=len(group), ratios={})
        for ratio in (.5, .65, .8):
            leads = [row['candidates'][str(ratio)]['lead_steps'] for row in group
                if row['candidates'][str(ratio)]['lead_steps'] is not None]
            groups[outcome]['ratios'][str(ratio)] = dict(has_preclose_slowdown=len(leads),
                median_lead_steps=quantile(leads, .5) if leads else None)
    save(root / 'object_first_close_slowdown.json', dict(
        scope='Raw slowdown cues in the saved forecast of the plan that supplied the first close; '
              'not filtered scheduling decisions or a closed-loop result',
        contact_source=str(grasp_index), analyzed=len(rows), total=len(contacts),
        groups=groups, episodes=rows))
    with (root / 'results.md').open('a') as stream:
        stream.write('\n## Object首次闭爪的来源预测\n\n'
            f'{len(contacts)}回合中{len(rows)}回合有可对齐的来源预测，其余未纳入本项。只统计原始减速信号。\n\n'
            '|原结果|可对齐回合|0.5有信号/提前步中位数|0.65有信号/提前步中位数|0.8有信号/提前步中位数|\n'
            '|---|---:|---:|---:|---:|\n')
        for outcome, group in groups.items():
            cells = [f"{group['ratios'][str(r)]['has_preclose_slowdown']}/{group['ratios'][str(r)]['median_lead_steps']}"
                for r in (.5, .65, .8)]
            stream.write(f"|{outcome}|{group['episodes']}|" + '|'.join(cells) + '|\n')
        stream.write('\n提前步数为最近一个减速点距首次闭爪的间隔；仅在有此信号的回合中计算。'
            '提高比例后能出现更多信号，但仍需经过阶段和反馈过滤，也不保证来得及完成下一次推理。\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--grasp-index', type=Path,
        help='Optional saved Object grasp-contact analysis for first-close source diagnostics')
    args = parser.parse_args()
    original = json.loads(INDEX.read_text())['rows']
    root = args.output or INDEX.parent.parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-motion-cue-calibration-old200')
    root.mkdir(parents=True, exist_ok=True)
    save('/tmp/perceptbudget-motion-calibration-current.json', dict(run=str(root)))
    save(root / 'plan.json', dict(index=str(INDEX), variants=VARIANTS, model_loads=0,
        selection='one parameter at a time, every old200 case; same settings across tasks/seeds'))
    rows, errors = [], []
    print('run_dir='+str(root), flush=True)
    for i, row in enumerate(original, 1):
        try:
            rows.append(analyze(row))
        except Exception as exc:
            errors.append(dict(run=row['run'], error=f'{type(exc).__name__}: {exc}'))
        if i % 20 == 0:
            report(root, rows, errors, len(original))
            print(f'PROGRESS={i}/{len(original)} errors={len(errors)}', flush=True)
    report(root, rows, errors, len(original), complete=True)
    if args.grasp_index:
        inspect_grasp_sources(root, original, args.grasp_index)
    print('SUMMARY='+json.dumps(json.loads((root/'summary.json').read_text()), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
