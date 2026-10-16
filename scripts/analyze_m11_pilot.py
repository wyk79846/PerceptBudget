"""CPU-only incremental M1.1 pilot analysis; writes derived artifacts only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from perceptbudget.stage_evidence import TASK_OBJECTS, annotate_episode
from scripts.analyze_async_q import (bootstrap_mean, compare_rows, describe,
    paired_success, prediction_events, summarize_event_types, buffer_statistics)


def normalize(trace):
    rows = [r for r in trace if r.get('event') == 'control_step']
    if [r['step'] for r in rows] != list(range(len(rows))):
        raise ValueError('control steps must be a complete ordered prefix from zero')
    if any(len(r.get('action', [])) != 7 for r in rows):
        raise ValueError('control action must have seven dimensions')
    return rows


def split_statistics(rows, annotation, requests):
    requests = {r['prediction_id']: r for r in requests}
    groups = {}
    for row, stage in zip(rows, annotation['steps']):
        source = row.get('source')
        origin = source.get('origin_step') if source else None
        request = requests.get(source.get('prediction_id')) if source else None
        expiry = origin + len(request['actions']) if origin is not None and request and request.get('actions') else None
        sample = dict(step=row['step'], buffer=row.get('queue_before_pop_actions'),
                      prediction_offset=source.get('offset') if source else None,
                      elapsed_age_steps=row['step']-origin if origin is not None else None,
                      original_H_expiry_step=expiry,
                      original_H_remaining_steps=expiry-row['step'] if expiry is not None else None,
                      underflow=source is None)
        contact = stage['evidence']['finger_target_contact']
        for key in ('phase:'+stage['phase'], 'targetcontact:'+str(contact)):
            groups.setdefault(key, []).append(sample)
    return {key: dict(note='phase/contact取动作后after；buffer取同一步动作前before-pop，年龄/expiry取本步执行边界；不是下一步buffer。',
                     n=len(samples), underflow_steps=sum(s['underflow'] for s in samples),
                     **{field: describe([s[field] for s in samples]) for field in
                        ('buffer', 'prediction_offset', 'elapsed_age_steps', 'original_H_remaining_steps')},
                     records=samples) for key, samples in groups.items()}


def pair_result(pairs, missing):
    tasks = [dict(task_id=a['task_id'], baseline=a['episode_id'], candidate=b['episode_id'],
                  trajectory=compare_rows(a['rows'], b['rows']),
                  baseline_status=a['status'], candidate_status=b['status']) for a, b in pairs]
    times = {}
    for field in ('real_seconds', 'simulated_seconds'):
        for scope in ('all_attempts', 'mutual_success'):
            times[field+'_'+scope+'_difference'] = bootstrap_mean([
                b[field]-a[field] for a, b in pairs if a.get(field) is not None and b.get(field) is not None
                and (scope == 'all_attempts' or a['status'] == b['status'] == 'success')])
    metrics = {field: bootstrap_mean([t['trajectory'][field]['mean'] for t in tasks
               if t['trajectory'][field]['mean'] is not None]) for field in
               ('translation_command_l2', 'rotation_command_l2', 'eef_xyz_distance_m')}
    return dict(success_difference=paired_success([a['status']=='success' for a, b in pairs],
                [b['status']=='success' for a, b in pairs]), tasks=tasks,
                missing_pairs=missing, time=times, trajectory_task_means=metrics)


def analyze(root):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    plan = [dict(task_id=t, mode=m, q=q, disturbed=d) for t in (0, 3)
            for m, qs in (('sync', (5, 8, 12)), ('async', (8, 12))) for q in qs for d in (False, True)]
    key = lambda e: (e['task_id'], e['mode'], e['q'], e['disturbed'])
    expected = {key(e) for e in plan}
    episodes, stages, frames, warnings, ids = {}, {}, [], [], set()
    for original in summary['episodes']:
        ep = dict(original)
        k, eid = key(ep), ep['episode_id']
        if k not in expected or k in episodes or eid in ids:
            raise ValueError('Unexpected or duplicate episode: '+eid)
        ids.add(eid)
        artifact = (root/ep['artifact_dir']).resolve()
        if not artifact.is_relative_to(root.resolve()):
            raise ValueError('artifact_dir escapes run')
        trace = [json.loads(line) for line in (artifact/'trace.jsonl').read_text().splitlines() if line.strip()]
        rows = normalize(trace)
        if len(rows) != ep['steps']:
            raise ValueError('Trace/episode step count mismatch: '+eid)
        request_path = artifact/f"q{ep['q']}_requests.json"
        requests = json.loads(request_path.read_text()) if request_path.exists() else []
        if not request_path.exists():
            warnings.append(eid+': missing requests; prediction evidence unavailable')
        annotation = annotate_episode(rows, *TASK_OBJECTS[ep['task_id']])
        stages[eid] = dict(**annotation, split=split_statistics(rows, annotation, requests))
        commits = ep.get('commits', [])
        actionable = prediction_events(rows, requests, commits)
        raw = prediction_events(rows, requests, [dict(prediction_id=r['prediction_id'],
            commit_step=r['observation_step']) for r in requests if r.get('actions')])
        ep.update(rows=rows, trace_coverage=len(rows)/ep['steps'] if ep['steps'] else None,
                  buffer=buffer_statistics(commits), raw_prediction_events=raw,
                  actionable_prediction_events=actionable,
                  raw_event_types=summarize_event_types(raw), actionable_event_types=summarize_event_types(actionable),
                  applied_arm_gain=describe([r.get('applied_arm_gain') for r in rows]))
        episodes[k] = ep
        candidates = {0: ['initial'], max(0, len(rows)-1): ['last_control'] } if rows else {}
        for segment in annotation['segments']:
            candidates.setdefault(segment['start_step'], []).append('phase_boundary:'+segment['phase'])
        for step in (29, 30, 31):
            if ep['disturbed'] and step < len(rows):
                candidates.setdefault(step, []).append('disturbance_window')
        contacts = annotation['steps']
        for previous, current in zip(contacts, contacts[1:]):
            if previous['evidence']['finger_target_contact'] != current['evidence']['finger_target_contact']:
                candidates.setdefault(current['step'], []).append('target_contact_change')
        for step, reasons in sorted(candidates.items()):
            frames.append(dict(episode_id=eid, artifact_dir=ep['artifact_dir'], step=step,
                before_frame_index=step, post_action_frame_index=step+1,
                archive=f"q{ep['q']}_frames.npz", reasons=reasons,
                review_status='待人工核帧', note='阶段/contact为after证据，对应s+1；before为s。终止帧已包含；双视角旋转180度。'))
    paired, disturbance = {}, {}
    for mode, qs in (('sync', (5, 8, 12)), ('async', (8, 12))):
        for q in qs:
            for disturbed in (False, True):
                pairs, missing = [], []
                for task in (0, 3):
                    a, b = episodes.get((task, 'sync', 5, disturbed)), episodes.get((task, mode, q, disturbed))
                    if a is None or b is None:
                        missing.append(dict(task_id=task, missing_baseline=a is None, missing_candidate=b is None))
                    else:
                        pairs.append((a, b))
                paired[f'{mode}-q{q}-'+('nudge' if disturbed else 'clean')] = pair_result(pairs, missing)
            pairs, missing = [], []
            for task in (0, 3):
                a, b = episodes.get((task, mode, q, False)), episodes.get((task, mode, q, True))
                if a is None or b is None:
                    missing.append(dict(task_id=task, missing_clean=a is None, missing_nudge=b is None))
                else:
                    pairs.append((a, b))
            disturbance[f'{mode}-q{q}'] = pair_result(pairs, missing)
    exploration = []
    for task in (0, 3):
        for disturbed in (False, True):
            for mode, qs in (('sync', (5, 8, 12)), ('async', (8, 12))):
                largest = None
                for q in qs:
                    ep = episodes.get((task, mode, q, disturbed))
                    if ep is None or ep['status'] != 'success':
                        break
                    largest = q
                exploration.append(dict(task_id=task, mode=mode, disturbed=disturbed,
                    largest_tested_successful_q=largest, local_exploratory_seconds=largest*.05 if largest else None,
                    B_safe=None, tested_q_order=list(qs)))
    missing = [p for p in plan if key(p) not in episodes]
    limitations = ['仅2个任务各条件1次；区间不证明非劣或普适可靠性，M1未据此全部完成。',
        '外扰仅正式step30、世界x+2cm；不代表广泛鲁棒性。人工核帧待主agent审查。',
        '阶段为保守proxy；unknown保持未知，证据及segments见stage_proxy.json。',
        '同控制步共同前缀轨迹只作描述；coverage之外及失败段尾不计安全。初态未由本分析独立核验。',
        'largest_tested_successful_q只沿本模式连续已测较短q成功延伸，乘0.05仅局部探索值，B_safe=null。',
        '事件是预测与随后执行命令的后验对照；gain会改变实际命令，actionable有自我一致性偏差，并非物理真值误差。',
        '配对成功率差用保守精确gain/loss区间；连续指标按任务bootstrap，少于2任务无CI。共同成功时间与全尝试终止耗时分开。']
    result = dict(source_summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  stage_source_sha256=hashlib.sha256((ROOT/'perceptbudget/stage_evidence.py').read_bytes()).hexdigest(),
                  status='complete_pilot' if not missing and all(e['status'] in ('success', 'timeout', 'failed', 'failure')
                  for e in episodes.values()) else 'partial', source_status=summary.get('status'),
                  expected_episodes=20, observed_episodes=len(episodes), missing_episodes=missing,
                  episodes=[{k: v for k, v in e.items() if k != 'rows'} for e in episodes.values()],
                  paired_vs_syncq5=paired, clean_vs_nudge=disturbance, quality_exploration=exploration,
                  warnings=warnings, limitations=limitations, B_safe=None)
    lines = ['# 注意：仅2个任务的M1.1小规模探索', '',
             f"状态：{result['status']}；记录{len(episodes)}/20回合；缺{len(missing)}回合。", '',
             '|条件|配对任务数|成功率差及95%CI|共同成功真实耗时差均值(s)|', '|---|---:|---|---:|']
    for name, pair in paired.items():
        success = pair['success_difference']
        lines.append(f"|{name}|{success['n']}|{success['difference']} / {success['ci95']}|{pair['time']['real_seconds_mutual_success_difference']['mean']}|")
    lines += ['', '|外扰对照条件|配对任务数|nudge−clean成功率差及95%CI|共同成功真实耗时差(s)|', '|---|---:|---|---:|']
    for name, pair in disturbance.items():
        success = pair['success_difference']
        lines.append(f"|{name}|{success['n']}|{success['difference']} / {success['ci95']}|{pair['time']['real_seconds_mutual_success_difference']['mean']}|")
    lines += ['', '|任务|模式/q|clean状态/步数/真实秒|nudge状态/步数/真实秒|', '|---|---|---|---|']
    for task in (0, 3):
        for mode, qs in (('sync', (5, 8, 12)), ('async', (8, 12))):
            for q in qs:
                cells = []
                for disturbed in (False, True):
                    e = episodes.get((task, mode, q, disturbed))
                    cells.append(f"{e['status']}/{e['steps']}/{e.get('real_seconds')}" if e else '缺失')
                lines.append(f"|{task}|{mode}/q{q}|{cells[0]}|{cells[1]}|")
    lines += ['', '差值为候选减基线。轨迹、逐任务coverage、时间及事件明细见analysis.json；阶段原证据见stage_proxy.json；候选核帧见review_frames.json。', '', *limitations,
              '', '记录警告：'+('；'.join(warnings) or '无；缺失配对逐项列于JSON。')]
    for name, value in (('analysis.json', result), ('stage_proxy.json', stages), ('review_frames.json', frames)):
        (root/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (root/'analysis.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    analyze(parser.parse_args().run_dir)
