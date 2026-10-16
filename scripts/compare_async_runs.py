#!/usr/bin/env python3
"""Create an offline, provenance-preserving cross-run async comparison."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perceptbudget.experiment import create_run
from scripts.analyze_async_q import QS, analyze, render
from scripts.run_async_q import validate_pairing

FIXED = ('H', 'N', 'precision', 'state_enabled', 'model_id', 'flashrt_revision',
         'alignment_policy', 'pending_arm_gain')
DEFAULTS = {'action_handoff': 'crop_overwrite', 'refresh_anchor': 'request',
            'model_lifetime': 'task'}
LIMITATION = ('跨run汇总：来源为不同运行；task/session模型与环境生命周期可能变化，'
              '不是严格单因素对照。历史加载时间、总耗时及能耗不可无差别拼接，'
              '不能将差异仅归因于q或声称异步节能。配对仍须通过同任务初态及首帧检查。')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8')


def check_mode(value):
    """Reject conflicting metadata even when nested in episode/commit records."""
    if isinstance(value, dict):
        mode = value.get('execution_mode')
        if mode is not None and mode != 'asynchronous_q5_paired':
            raise ValueError('non-asynchronous execution_mode: '+str(mode))
        if (value.get('refresh_anchor', 'request') != 'request' or
                value.get('refresh_policy') == 'post_commit'):
            raise ValueError('post_commit/commit refresh is not the request baseline')
        for child in value.values():
            check_mode(child)
    elif isinstance(value, list):
        for child in value:
            check_mode(child)


def identity(summary):
    cfg = summary['config']
    result = {}
    for key in FIXED:
        if cfg.get(key) is None:
            raise ValueError('missing fixed config: '+key)
        result[key] = cfg[key]
    for key in ('action_handoff', 'refresh_anchor'):
        result[key] = cfg.get(key, DEFAULTS[key])
    for key in ('checkpoint_sha256', 'norm_stats_sha256'):
        if not summary.get(key):
            raise ValueError('missing '+key)
        result[key] = summary[key]
    for key in ('seed', 'init_state_index'):
        values = [d[key] for d in (summary, cfg, cfg.get('screen', {}),
                                  *summary['episodes']) if key in d]
        if not values or any(v != values[0] for v in values):
            raise ValueError('missing or inconsistent '+key)
        result[key] = values[0]
    return result


def compare(run_dirs, require_complete=False):
    sources, episodes, seen = [], [], set()
    fixed = None
    for directory in run_dirs:
        root = Path(directory).expanduser().resolve(strict=True)
        raw = (root/'summary.json').read_bytes()
        summary = json.loads(raw)
        check_mode(summary)
        cfg = summary['config']
        if cfg.get('execution_mode') != 'asynchronous_q5_paired':
            raise ValueError('source must declare asynchronous_q5_paired')
        current = identity(summary)
        if fixed is not None and current != fixed:
            raise ValueError('fixed configuration mismatch: '+', '.join(
                k for k in current if current[k] != fixed[k]))
        fixed = current
        if require_complete and summary.get('status') != 'completed':
            raise ValueError('source not completed: '+str(root))
        sources.append({'source_run': str(root),
                        'summary_sha256': hashlib.sha256(raw).hexdigest(),
                        'status': summary.get('status'), 'config': copy.deepcopy(cfg),
                        'model_lifetime': cfg.get('model_lifetime', 'task')})
        for original in summary['episodes']:
            ep = copy.deepcopy(original)
            task, q = ep['task_id'], ep['q']
            if type(task) is not int or task not in range(30) or type(q) is not int or q not in QS:
                raise ValueError('unexpected task/q: '+repr((task, q)))
            if (task, q) in seen:
                raise ValueError('duplicate task/q: '+repr((task, q)))
            seen.add((task, q))
            if require_complete and ep.get('status') not in ('success', 'timeout', 'failed', 'failure'):
                raise ValueError('incomplete episode: '+repr((task, q)))
            artifact = (root/ep.get('artifact_dir', f'tasks/task{task:02d}')).resolve(strict=True)
            # The analyzer also checks episode_start; reject rather than silently exclude it.
            trace = artifact/'trace.jsonl'
            if trace.exists():
                with trace.open() as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # Analyzer warns about partial JSONL.
                        if record.get('q') == q or record.get('event') == 'episode_start':
                            check_mode({k: record[k] for k in
                                        ('execution_mode', 'refresh_anchor', 'refresh_policy')
                                        if k in record})
            ep.update(artifact_dir=str(artifact), source_run=str(root))
            episodes.append(ep)
    if not sources or not episodes:
        raise ValueError('no source episodes')
    complete = seen == {(task, q) for task in range(30) for q in QS}
    if require_complete and not complete:
        raise ValueError('require-complete needs exactly task0..29 for q5/8/10/12/15')
    episodes.sort(key=lambda ep: (ep['task_id'], ep['q']))
    for task in sorted({ep['task_id'] for ep in episodes}):
        paired = [ep for ep in episodes if ep['task_id'] == task]
        for ep in paired:
            # Never trust a comparison to an earlier run's different baseline.
            ep.pop('initial_frame_comparison', None)
        validate_pairing(paired, ROOT)

    differences = {}
    configs = [source['config'] for source in sources]
    for key in sorted(set().union(*(cfg.keys() for cfg in configs))):
        values = [cfg.get(key, DEFAULTS.get(key)) for cfg in configs]
        if any(value != values[0] for value in values[1:]):
            differences[key] = values
    warnings = [LIMITATION]
    if differences:
        warnings.append('源配置差异（按sources顺序）：'+', '.join(differences))
    if len({source['model_lifetime'] for source in sources}) > 1:
        warnings.append('模型/环境生命周期变化：历史缺省task与新session混合；耗时与加载成本不可直接归因于q。')
    cfg = copy.deepcopy(configs[0])
    cfg.update(candidate_q=sorted({ep['q'] for ep in episodes}),
               **{k: fixed[k] for k in ('action_handoff', 'refresh_anchor')})
    summary = {'status': 'completed' if complete and all(s['status'] == 'completed' for s in sources) else 'partial',
               'execution_mode': cfg['execution_mode'], 'config': cfg,
               'cross_run': True, 'sources': sources, 'config_differences': differences,
               'warnings': warnings, 'limitations': [LIMITATION], 'episodes': episodes,
               'q_values': cfg['candidate_q'], 'expected_episodes': 150,
               **{k: fixed[k] for k in ('seed', 'init_state_index', 'checkpoint_sha256', 'norm_stats_sha256')}}
    output = create_run(cfg, 'M1', 'async-final', repo_root=ROOT)
    write_json(output/'summary.json', summary)
    result = analyze(output)
    result.update(cross_run=True, sources=sources, config_differences=differences,
                  limitations=[LIMITATION]+result.get('limitations', []))
    result['warnings'] = warnings + result['warnings']
    write_json(output/'analysis.json', result)
    (output/'analysis.md').write_text('> **跨run及生命周期限制：'+LIMITATION+'**\n\n'+
                                    '\n'.join('> '+w for w in warnings[1:])+'\n\n'+render(result),
                                    encoding='utf-8')
    metadata = json.loads((output/'run.json').read_text())
    metadata.update(status=summary['status'], cross_run=True, analysis_status='completed')
    write_json(output/'run.json', metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', type=Path, nargs='+')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    print(compare(args.run_dirs, args.require_complete))


if __name__ == '__main__':
    main()
