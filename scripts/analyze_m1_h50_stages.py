"""CPU-only physical-stage supplement for the 20 indexed clean Spatial0/3 runs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perceptbudget.stage_evidence import annotate_episode

PHASES = ('transport_candidate', 'contact_candidate',
          'placement_approach_candidate', 'placement_candidate', 'unknown')
QS = (5, 10, 20, 30, 40)
MODES = ('sync_no_rtc', 'actual_history')
TARGET, GOAL = 'akita_black_bowl_1', 'plate_1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(annotation):
    c = Counter(s['phase'] for s in annotation['steps'])
    if set(c) - set(PHASES):
        raise ValueError('unexpected phase')
    return {p: c[p] for p in PHASES}


def analyze(batch):
    index_path = batch / 'index.json'
    fingerprint = sha(index_path)
    index = json.loads(index_path.read_text())
    episodes = []
    for task in (0, 3):
        for q in QS:
            for mode in MODES:
                matches = [r for r in index['cells'].values()
                           if r['cell'] == ['libero_spatial', task, q, mode]]
                if len(matches) != 1 or matches[0]['status'] != 'completed':
                    raise ValueError('missing/duplicate/incomplete selected cell')
                record = matches[0]
                worker = Path(record['result']['run'])
                if not worker.is_absolute():
                    worker = batch / worker
                trace = worker / 'rtc/trace.jsonl'
                events = [json.loads(line) for line in trace.read_text().splitlines()]
                rows = [r for r in events if r.get('event') == 'control_step']
                summary = json.loads((worker / 'summary.json').read_text())['episode']
                if (not rows or [r['step'] for r in rows] != list(range(summary['steps']))
                        or summary['task_id'] != task or summary['q'] != q
                        or any(r['task_id'] != task or r['q'] != q for r in rows)
                        or any(e.get('event') == 'external_disturbance' for e in events)):
                    raise ValueError('not a complete clean episode')
                annotation = annotate_episode(rows, TARGET, GOAL)
                contact = Counter(s['evidence']['finger_target_contact'] for s in annotation['steps'])
                episodes.append(dict(task_id=task, q=q, mode=mode, worker=str(worker),
                    outcome=summary['status'], trace_sha256=sha(trace),
                    summary_sha256=sha(worker / 'summary.json'),
                    phase_counts=counts(annotation),
                    contact_evidence_counts=dict(contact=contact[True], no_contact=contact[False],
                                                 unknown=contact[None]), **annotation))
    if sha(index_path) != fingerprint:
        raise ValueError('index changed during review')
    totals = {p: sum(e['phase_counts'][p] for e in episodes) for p in PHASES}
    return dict(index_path=str(index_path), index_sha256=fingerprint,
                annotator_sha256=sha(Path(__file__).resolve().parents[1] / 'perceptbudget/stage_evidence.py'),
                script_sha256=sha(Path(__file__)), source_status=index['status'],
                scope='20 clean indexed Spatial task0/task3 episodes; not all 300',
                contact_candidate_supported=False, totals=totals, episodes=episodes)


def export_frames(result, output):
    """Decode three existing raw frames verbatim; no rotate/crop/resize/overlay."""
    import numpy as np
    from PIL import Image
    chosen = []
    for task, phase in ((0, 'transport_candidate'), (3, 'placement_approach_candidate'),
                        (3, 'unknown')):
        options = [(e, s) for e in result['episodes'] if e['task_id'] == task
                   for s in e['steps'] if s['phase'] == phase]
        if not options:
            continue
        e, s = options[-1] if phase == 'unknown' else options[0]
        spool = Path(e['worker']) / f"rtc/q{e['q']}_frames_spool.npy"
        frames = np.load(spool, mmap_mode='r', allow_pickle=False)
        # Each step stores the pre-action observation; annotation uses AFTER.
        idx = s['step'] + 1
        if idx > len(e['steps']) or idx >= len(frames):
            raise ValueError('missing after-action frame')
        pixels = frames[idx]
        path = output / f"frame_task{task}_{e['mode']}_q{e['q']}_after{s['step']}_{phase}.png"
        Image.fromarray(pixels).save(path)
        if not np.array_equal(np.asarray(Image.open(path)), pixels):
            raise ValueError('frame export changed pixels')
        chosen.append(dict(path=str(path), source=str(spool), frame_index=idx,
                           step=s['step'], phase=phase, task_id=task,
                           pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                           visual_review='pending', orientation='raw recorded orientation'))
    return chosen


def markdown(result):
    lines = ['# M1 H50 阶段补充证据', '',
        '范围：主批次已有 Spatial0/task3 × 5q × sync/async，共20个clean回合。CPU离线复用 annotate_episode 默认阈值，无新增实验。', '',
        '所有阶段均为保守候选，不是真值；不以夹爪命令判定抓住。contact_candidate 未由现有标注器实现，表中0表示未分类，不表示没有接触；接触独立统计见JSON。unknown包含未达条件、短窗口及证据不足，不能等同失败。', '',
        '|task|q|mode|transport|contact*|approach|placement|unknown|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for e in result['episodes']:
        lines.append('|' + '|'.join(map(str, [e['task_id'], e['q'], e['mode'],
                     *(e['phase_counts'][p] for p in PHASES)])) + '|')
    lines += ['', '合计（控制步描述计数，不能当独立统计样本）：' + json.dumps(result['totals']), '',
        '边界为候选确认步；placement须有此前transport证据、目标支撑接触、释放与静止窗口。approach是下降/连续小调整代理，不能称精细放置真值。未推断B_safe，未重复动作事件或同步/异步性能统计。', '',
        '限制：只覆盖2个任务，每配置1回合；默认阈值未重新调参；unknown保留。接触是保存的仿真几何接触，非力传感器或可靠抓取标签。终止后缺少持续观察可能阻止placement确认。', '',
        '帧：只抽取3张已有spool原始帧，mmap只读；PNG无旋转/裁切/缩放/叠字且逐像素核对。annotation采用after场景，对应spool[step+1]。原始朝向保留，未生成视频。视觉核验记录见 visual_review.md（若存在），少量图不验证全部阶段边界。', '',
        '来源SHA、每步证据、连续区间、接触计数、阈值和帧出处均见 stages.json。原索引/数据/runtime均未写入；输出目录存在即拒绝覆盖。']
    return '\n'.join(lines) + '\n'


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def window_duty(windows, busy):
    """Intersect unions on real time; never bridge an excluded phase interval."""
    windows, busy = union(windows), union(busy)
    overlaps, gaps = [], []
    for a, b in windows:
        cursor = a
        for x, y in busy:
            lo, hi = max(a, x), min(b, y)
            if hi <= lo:
                continue
            if lo > cursor:
                gaps.append(lo - cursor)
            overlaps.append([lo, hi])
            cursor = hi
        if cursor < b:
            gaps.append(b - cursor)
    duration = sum(b-a for a, b in windows)
    seconds = sum(b-a for a, b in overlaps)
    return dict(T_seconds=duration, busy_seconds=seconds,
                d=seconds/duration if duration else None,
                windows=windows, busy_intersections=overlaps,
                idle_gaps_seconds=gaps)


def phase_duty(batch):
    directory = batch / 'stage_review_final'
    outputs = [directory / ('phase_duty.' + ext) for ext in ('json', 'md')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('phase_duty output already exists')
    label_path = directory / 'stages.json'
    labels = json.loads(label_path.read_text())
    fingerprints = {str(label_path): sha(label_path),
                    str(directory / 'stages.md'): sha(directory / 'stages.md')}
    episodes, groups = [], {}
    phases = tuple(p for p in PHASES if p != 'contact_candidate')
    expected = {(t, q, m) for t in (0, 3) for q in QS for m in MODES}
    identities = [(e['task_id'], e['q'], e['mode']) for e in labels['episodes']]
    if len(identities) != 20 or set(identities) != expected:
        raise ValueError('expected exactly 20 existing clean labels')
    for e in labels['episodes']:
        worker = Path(e['worker'])
        trace_path, summary_path = worker/'rtc/trace.jsonl', worker/'summary.json'
        requests_path = worker/f"rtc/q{e['q']}_requests.json"
        for path, old in ((trace_path, e['trace_sha256']), (summary_path, e['summary_sha256'])):
            if sha(path) != old:
                raise ValueError('source fingerprint mismatch: ' + str(path))
        for path in (trace_path, summary_path, requests_path):
            fingerprints[str(path)] = sha(path)
        rows = [r for r in map(json.loads, trace_path.read_text().splitlines())
                if r.get('event') == 'control_step']
        ep = json.loads(summary_path.read_text())['episode']
        requests = json.loads(requests_path.read_text())
        if (not rows or len(rows) != ep['steps'] or len(rows) != len(e['steps'])
                or [r['step'] for r in rows] != list(range(len(rows)))
                or ep['task_id'] != e['task_id'] or ep['q'] != e['q']):
            raise ValueError('episode/labels identity mismatch')
        start, stop = ep['start_monotonic'], ep['stop_monotonic']
        ticks = [r['tick_monotonic'] for r in rows] + [stop]
        values = [start, *ticks]
        if (any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                or start > ticks[0] or any(b <= a for a, b in zip(ticks, ticks[1:]))):
            raise ValueError('invalid task/tick clock')
        busy = []
        if not requests:
            raise ValueError('missing requests')
        for r in requests:
            a, b = r['inference_started_monotonic'], r['inference_ended_monotonic']
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (a, b)) or b < a:
                raise ValueError('invalid inference interval')
            busy.append((max(start, a), min(stop, b)))
        windows = {p: [] for p in phases}
        contacts = {p: [] for p in ('true', 'false', 'unknown')}
        for i, s in enumerate(e['steps']):
            if s['step'] != rows[i]['step'] or s['phase'] not in windows:
                raise ValueError('label step/phase mismatch')
            interval = (ticks[i], ticks[i+1])
            windows[s['phase']].append(interval)
            contact = s['evidence']['finger_target_contact']
            if contact is not None and type(contact) is not bool:
                raise ValueError('invalid contact boolean')
            contacts['unknown' if contact is None else str(contact).lower()].append(interval)
        item = dict(task_id=e['task_id'], q=e['q'], mode=e['mode'], worker=e['worker'],
                    task_seconds=stop-start, covered_tick_seconds=stop-ticks[0],
                    unassigned_before_first_tick=window_duty([(start, ticks[0])], busy),
                    phases={}, finger_target_contact={})
        for kind, buckets in (('phases', windows), ('finger_target_contact', contacts)):
            for name, intervals in buckets.items():
                metrics = window_duty(intervals, busy)
                item[kind][name] = metrics
                key = (e['mode'], e['q'], kind, name)
                group = groups.setdefault(key, dict(mode=e['mode'], q=e['q'], kind=kind,
                    label=name, episodes_total=0, episodes_with_window=0,
                    T_seconds=0., busy_seconds=0., idle_gaps_seconds=[]))
                group['episodes_total'] += 1
                group['episodes_with_window'] += bool(intervals)
                for field in ('T_seconds', 'busy_seconds'):
                    group[field] += metrics[field]
                group['idle_gaps_seconds'].extend(metrics['idle_gaps_seconds'])
        episodes.append(item)
    for g in groups.values():
        g['d'] = g['busy_seconds']/g['T_seconds'] if g['T_seconds'] else None
        g['idle_seconds'] = g['T_seconds']-g['busy_seconds']
        g['opportunity_symbolic'] = f"{1-g['d']:.9g} * (1-r)" if g['d'] is not None else None
        gaps = g['idle_gaps_seconds']
        g['idle_gap_count'] = len(gaps)
        g['idle_gap_max_seconds'] = max(gaps, default=None)
        g['idle_gap_mean_seconds'] = sum(gaps)/len(gaps) if gaps else None
    if any(sha(Path(path)) != digest for path, digest in fingerprints.items()):
        raise ValueError('source changed during analysis')
    result = dict(scope=labels['scope'], method='tick[i] to tick[i+1]; last to episode.stop; '
        'existing after-scene label assigned to whole tick window; inference interval union intersection; '
        'pool seconds across the two tasks, not mean of episode d; gaps stop at phase/episode boundaries',
        notes=['Synchronous waits inside tick windows included, asynchronous uses identical windows.',
               'Exclude warmup and drain by task start/stop clipping. Before first tick is unassigned '
               '(may include bootstrap inference); explicitly reported, not silently assigned unknown.',
               'Contact true/false/unknown is an independent overlapping partition; do not add contact '
               'metrics to phase metrics. Missing phase has T=0 and d=null.',
               'After-scene candidate labels are retrospective; attribution over the preceding tick '
               'window is a convention, not measured physical transition timing.',
               'No power per step; r=P_idle/P_on remains symbolic, no historical r used; '
               'idle gaps are wall-time without recorded inference, not GPU idle or realized savings.',
               'Two tasks only, one episode per task/mode/q; no generalization to all 300; unknown remains unknown.'],
        sources_sha256=fingerprints, script_sha256=sha(Path(__file__)),
        groups=list(groups.values()), episodes=episodes)
    lines = ['# 阶段占空比：20 clean回合有限证据', '',
        '仅Spatial0/task3，按模式/q合并两任务的时间求d=Σbusy/ΣT；不取逐回合d的平均。', '',
        '窗口为[tick_i,next_tick)，末步到episode.stop。沿用已有after场景标签覆盖整个tick窗口，是回顾性归属约定，不能定位真实物理转换时刻。同步推理等待计入其所在tick窗口，异步同规则。推理区间先求并集，再与阶段窗口相交。warm/drain不计入。', '',
        '首tick前任务时间单列unassigned_before_first_tick（可能包含bootstrap），不归unknown；JSON逐回合给出T/busy/d与窗口，用于核对完整任务窗口。相邻同类窗口合并，跨其他阶段/回合不连接idle间隙。间隙是记录推理区间之外的墙钟时间，不等于GPU空闲。', '',
        '|mode|q|phase|有窗口回合/2|T(s)|busy(s)|d|idle总(s)|gap数|max gap(s)|symbolic opportunity|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    def fmt(v):
        return 'NA' if v is None else f'{v:.6f}'
    for g in result['groups']:
        if g['kind'] != 'phases':
            continue
        lines.append('|'+'|'.join([g['mode'], str(g['q']), g['label'], str(g['episodes_with_window'])+'/2',
            *(fmt(g[k]) for k in ('T_seconds', 'busy_seconds', 'd', 'idle_seconds')),
            str(g['idle_gap_count']), fmt(g['idle_gap_max_seconds']), g['opportunity_symbolic'] or 'NA'])+'|')
    lines += ['', '## finger-target接触独立分区', '',
        '接触布尔来自已有阶段证据，非夹爪命令或可靠抓住标签。该分区与phase重叠，不能把接触时长/busy再加到阶段合计。', '',
        '|mode|q|contact|T(s)|busy(s)|d|gap数|max gap(s)|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for g in result['groups']:
        if g['kind'] == 'finger_target_contact':
            lines.append('|'+'|'.join([g['mode'], str(g['q']), g['label'],
                *(fmt(g[k]) for k in ('T_seconds', 'busy_seconds', 'd')),
                str(g['idle_gap_count']), fmt(g['idle_gap_max_seconds'])])+'|')
    lines += ['', '完整idle间隙列表、各回合窗口/交集、来源SHA256见phase_duty.json。缺失阶段T=0、d=NA，不能视为零推理占用。', '',
        '机会仅为symbolic (1-d)*(1-r)，r=P_idle/P_on待新标定；未使用旧r、未提供单步功率或实际节能结论。同步等待的占比也不能直接当异步节能收益。', '',
        '这是2任务、每条件各1回合的候选阶段描述，placement仅1回合出现，样本稀疏；unknown不作物理阶段推断。供Word按阶段讨论研究机会的有限证据，不推广到300全任务。']
    with outputs[0].open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with outputs[1].open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--phase-duty', action='store_true', help='reuse existing stage labels only')
    args = parser.parse_args()
    if args.phase_duty:
        result = phase_duty(args.batch)
        print(f"phase duty: {len(result['episodes'])} episodes, {len(result['groups'])} groups")
        return
    output = args.batch / 'stage_review_final'
    if output.exists():
        raise FileExistsError(output)
    result = analyze(args.batch)
    output.mkdir()
    result['frames'] = export_frames(result, output)
    (output / 'stages.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (output / 'stages.md').write_text(markdown(result))
    print(json.dumps(result['totals']))
    print(output)


if __name__ == '__main__':
    main()
