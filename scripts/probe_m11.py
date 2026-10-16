#!/usr/bin/env python3
"""Bounded M1.1 pilot: two tasks, full sync recording and one fixed disturbance."""

import argparse
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_q as base, probe_delay_buffer as probe
from scripts.run_async_q import (configure_handoff, file_sha256, handoff_options,
                                 memory_snapshot, validate_pairing)
from perceptbudget.disturbance import nudge_object
from perceptbudget.experiment import create_run, load_config
from perceptbudget.libero_input import load_libero_norm_stats

TARGET = 'akita_black_bowl_1'
SOURCES = ('scripts/probe_m11.py', 'scripts/probe_delay_buffer.py', 'scripts/eval_q.py',
           'scripts/run_async_q.py', 'perceptbudget/disturbance.py',
           'perceptbudget/stage_evidence.py', 'perceptbudget/fixtures.py',
           'perceptbudget/pairing.py', 'perceptbudget/libero_input.py')


def plan():
    return [dict(task_id=task, mode=mode, q=q, disturbed=disturbed,
                 episode_id=f'task{task:02d}/{mode}-q{q}-' + ('nudge' if disturbed else 'clean'))
            for task in (0, 3)
            for mode, qs in (('sync', (5, 8, 12)), ('async', (8, 12)))
            for disturbed in (False, True) for q in qs]


def make_hook(enabled, events):
    def hook(env, observation, step):
        if not enabled or step != 30:
            return observation, None
        if events:
            raise RuntimeError('Disturbance must occur exactly once')
        event = nudge_object(env, TARGET, (.02, 0., 0.))
        observation = env.env._get_observations(force_update=True)
        observed = observation[TARGET + '_pos']
        if any(abs(float(x)-y) > 1e-8 for x, y in zip(observed, event['after_pos'])):
            raise RuntimeError('Object observation cache did not refresh after disturbance')
        event.update(step=step, timing='before observation and action; sim time unchanged',
                     refreshed_observed_target_pos=observed.tolist())
        events.append(event)
        return observation, event
    return hook


def write_progress(root, summary):
    base._write_json_atomic(root / 'summary.json', summary)
    (root / 'progress.md').write_text(
        f"# M1.1小测\n\n状态：{summary['status']}；完成{len(summary['episodes'])}/20。\n\n"
        f"当前：{summary.get('active_episode', '无')}。\n", encoding='utf-8')


def validate_result(ep, config):
    if not ep.get('field_checks') or not all(ep['field_checks'].values()):
        raise RuntimeError('Incomplete episode evidence')
    if ep['disturbed'] and len(ep['disturbance_events']) != 1:
        raise RuntimeError('Requested disturbance did not occur')
    if not ep['disturbed'] and ep['disturbance_events']:
        raise RuntimeError('Unexpected disturbance in clean condition')
    if ep['mode'] == 'sync':
        if ep['underflow_steps'] or any(n != ep['q'] for n in ep['actual_refresh_intervals_steps']):
            raise RuntimeError('Synchronous q-prefix semantics violated')
        if any(c['control_steps_until_commit'] != 0 for c in ep['commits']):
            raise RuntimeError('Simulation progressed during synchronous inference')
    for c in ep['commits']:
        if c['new_prediction_remaining_actions'] != max(0, config['H'] - c['control_steps_until_commit']):
            raise RuntimeError('Invalid remaining-action horizon')


def run(root):
    import numpy as np
    config = json.loads((root / 'run.json').read_text())['config_snapshot']
    summary = json.loads((root / 'summary.json').read_text())
    for name, digest in summary['source_sha256'].items():
        if file_sha256(ROOT / name) != digest:
            raise RuntimeError('Pilot source changed; refusing unaudited resume: ' + name)
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl')
    torch, cuda = base._check_cuda_and_native(config)
    stats = load_libero_norm_stats(config['norm_stats_path'])
    loaded = time.monotonic()
    model = base._load_and_check_model(config, torch)
    summary.update(status='running', cuda=cuda, model_load_seconds=time.monotonic()-loaded,
                   checkpoint_sha256=file_sha256(config['model_path']),
                   norm_stats_sha256=file_sha256(config['norm_stats_path']))
    write_progress(root, summary)
    _, env_class = base._libero_api()
    try:
        for task_id in (0, 3):
            task_specs = [spec for spec in summary['plan'] if spec['task_id'] == task_id]
            done = {ep['episode_id'] for ep in summary['episodes']}
            if all(spec['episode_id'] in done for spec in task_specs):
                continue
            if memory_snapshot(torch)['MemAvailable_kib'] < 1024 * 1024:
                raise MemoryError('Less than 1 GiB available before environment allocation')
            task = base._load_tasks([task_id], config['screen']['suites'])[0]
            env = base._make_env(env_class, task)
            task_root = root / f'task{task_id:02d}'
            task_root.mkdir(exist_ok=True)
            try:
                metadata = env.env.parsed_problem
                base._write_json_atomic(task_root / 'problem.json', {
                    key: metadata.get(key) for key in ('goal_state', 'objects', 'fixtures', 'obj_of_interest')})
                obs, _ = base._prepare_env_episode(env, task, np)
                base._infer_actions(model, base._make_snapshot(obs, stats, config['state_enabled']),
                                    task['instruction'], config['H'], {})
                env._pb_lock_fixtures = True
                layout_path = task_root / 'fixture_layout.json'
                if layout_path.exists():
                    env._pb_fixture_layout = json.loads(layout_path.read_text())
                obs, _ = base._prepare_env_episode(env, task, np)
                base._write_json_atomic(layout_path, env._pb_fixture_layout)
                del obs
                for spec in task_specs:
                    if spec['episode_id'] in done:
                        continue
                    summary['active_episode'] = spec['episode_id']
                    write_progress(root, summary)
                    artifact = root / spec['episode_id']
                    artifact.mkdir(parents=True, exist_ok=True)
                    pending = artifact / 'pending_episode.json'
                    if pending.exists():
                        ep = json.loads(pending.read_text())
                    else:
                        print('start=' + spec['episode_id'], flush=True)
                        events = []
                        episode_config = dict(config, execution_mode=(
                            'synchronous_quality' if spec['mode'] == 'sync' else 'asynchronous_q5_paired'))
                        options = (dict(synchronous=True, condition='synchronous_quality')
                                   if spec['mode'] == 'sync' else handoff_options(config))
                        with (artifact / 'trace.jsonl').open('w') as stream:
                            ep = probe.run_episode(env, task, spec['q'], model, stats,
                                episode_config, np, torch, artifact, stream,
                                step_hook=make_hook(spec['disturbed'], events), **options)
                        ep.update(**spec, suite=task['suite'], artifact_dir=spec['episode_id'],
                                  disturbance_events=events,
                                  execution_mode=episode_config['execution_mode'])
                        base._write_json_atomic(pending, ep)
                    validate_result(ep, config)
                    baseline = next((e for e in summary['episodes'] if e['task_id'] == task_id
                                     and e['mode'] == 'sync' and e['q'] == 5 and not e['disturbed']), None)
                    validate_pairing([baseline, ep] if baseline else [ep], root)
                    base._write_json_atomic(artifact / 'episode.json', ep)
                    summary['episodes'].append(ep)
                    done.add(ep['episode_id'])
                    write_progress(root, summary)
                    print(f"completed={len(summary['episodes'])}/20 {spec['episode_id']} {ep['status']}", flush=True)
            finally:
                env.close()
                del env
                gc.collect()
                torch.cuda.synchronize()
                summary['last_cleanup_memory'] = memory_snapshot(torch)
        summary.update(status='completed', active_episode=None)
        write_progress(root, summary)
    except BaseException as exc:
        summary.update(status='interrupted', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        write_progress(root, summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    if args.resume:
        root = args.resume.resolve(strict=True)
        config = json.loads((root / 'run.json').read_text())['config_snapshot']
    else:
        config = copy.deepcopy(load_config(ROOT / 'config.json'))
        configure_handoff(config)
        config.update(execution_mode='m11_stage_disturbance_pilot', comparison_baseline='sync_q5',
                      disturbance={'target': TARGET, 'step': 30, 'delta_xyz_m': [.02, 0, 0]},
                      pilot_sync_q=[5, 8, 12], pilot_async_q=[8, 12])
        config['screen'].update(task_ids=[0, 3], trials_per_task=1)
        root = create_run(config, 'M1', 'm11-stage-disturbance', repo_root=ROOT)
        for name in SOURCES:
            destination = root / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        summary = dict(status='created', config=config, plan=plan(), expected_episodes=20,
                       episodes=[], source_sha256={name: file_sha256(ROOT / name) for name in SOURCES})
        write_progress(root, summary)
    print('run_dir=' + str(root), flush=True)
    with (Path(config['output_root']) / '.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root)
    # Analysis has no model imports and can be repaired without rerunning episodes.
    from scripts.analyze_m11_pilot import analyze
    analyze(root)


if __name__ == '__main__':
    main()
