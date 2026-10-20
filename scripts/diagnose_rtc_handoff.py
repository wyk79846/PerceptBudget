"""Explicit task/condition diagnostics; original expert shared across conditions.

The synchronous condition calls bootstrap (no guidance) for every prediction.
Only this diagnostic's episode call is patched; native warmup is unchanged.
Each worker is isolated and RAM-guarded. No production defaults are changed.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run
from scripts import eval_rtc_openpi as runner
from scripts import probe_delay_buffer as probe

DEFAULT_MODES = ('sync_no_rtc', 'controlled_d5', 'actual_est8')
MODES = DEFAULT_MODES + ('controlled_d8',)


def sync_episode(original, *args, **kwargs):
    adapter = kwargs.pop('rtc_adapter')
    kwargs.update(synchronous=True, condition='sync_original_expert_no_rtc')

    def infer(model, snapshot, prompt, horizon, request):
        if horizon != adapter.horizon:
            raise ValueError('diagnostic horizon mismatch')
        return adapter.bootstrap(snapshot, prompt, request)

    with patch.object(runner.base, '_infer_actions', infer):
        return original(*args, **kwargs)


def configuration(reference, mode):
    config = copy.deepcopy(reference)
    config.pop('controlled_delay_steps', None)
    config.update(rtc_paper_schedule=mode in ('actual_est8', 'controlled_d8'),
                  rtc_fixed_delay_estimate=8 if mode in ('actual_est8', 'controlled_d8') else None,
                  explicit_actual_delay_experiment=mode == 'actual_est8',
                  execution_mode='diagnostic_' + mode,
                  diagnostic_condition=mode)
    if mode == 'controlled_d5':
        config['controlled_delay_steps'] = 5
    if mode == 'controlled_d8':
        config['controlled_delay_steps'] = 8
    if mode == 'sync_no_rtc':
        # Preflight only; synchronous episodes have no delay history or guidance.
        config['rtc_fixed_delay_estimate'] = 0
    return config


def worker(reference_run, mode, parent, task_id=2):
    reference = json.loads((reference_run / 'run.json').read_text())['config_snapshot']
    config = configuration(reference, mode)
    config['screen']['task_ids'] = [task_id]
    if task_id != 2:
        config['diagnostic_prefix_bank'] = 'initial valid length minus3 through plus4; unseen lengths use slow reference'
    gate = json.loads((Path(config['verified_reference_run']) / 'summary.json').read_text())
    if not gate.get('passed') or not gate.get('weights_frozen') or not gate.get('author_order'):
        raise RuntimeError('requires passed frozen author-order gate')
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                      PYOPENGL_PLATFORM='egl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (Path(config['output_root']) / '.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = create_run(config, 'M1', 'rtc-diag-' + mode, repo_root=ROOT)
        print('run_dir=' + str(run), flush=True)
        (parent / (mode + '.json')).write_text(json.dumps({'run': str(run), 'status': 'running'}))
        result = {'status': 'invalid'}
        try:
            from contextlib import ExitStack
            from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
            prepare = OriginalOpenPiRTCAdapter.prepare_step_bank

            def prepare_for_task(adapter, lengths):
                valid = int(adapter.expert.prefix_mask.shape[1])
                exact = list(range(max(1, valid-3), min(592, valid+4)+1))
                adapter.config['source_step_prefixes'] = exact
                return prepare(adapter, exact)

            original = probe.run_episode
            with ExitStack() as stack:
                if task_id != 2:
                    stack.enter_context(patch.object(OriginalOpenPiRTCAdapter, 'prepare_step_bank', prepare_for_task))
                if mode == 'sync_no_rtc':
                    stack.enter_context(patch.object(probe, 'run_episode', lambda *a, **k: sync_episode(original, *a, **k)))
                result = runner.rollout(config, run, gate, task_id)
            if mode == 'sync_no_rtc':
                result['scope'] = 'synchronous original expert, no RTC; simulation waits for every inference'
            if mode == 'controlled_d8':
                result['scope'] = 'fixed eight simulated delay steps; pauses simulation if inference late; not real-time async'
                req = json.loads((run / 'rtc/q5_requests.json').read_text())
                committed = {c['prediction_id'] for c in result['episode']['commits'][1:]}
                for request in req:
                    if request['prediction_id'] in committed:
                        guidance = request['rtc_guidance']
                        assert guidance['delay_estimate_steps'] == guidance['actual_delay_steps'] == 8
            result.update(diagnostic_condition=mode, reference_run=str(reference_run))
            print(json.dumps(result), flush=True)
            return 0 if result['status'] == 'completed' else 2
        except BaseException as exc:
            result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            runner.base._write_json_atomic(run / 'summary.json', result)
            runner.base._save_run_status(run, result['status'])
            runner.base._write_json_atomic(parent / (mode + '.json'), dict(run=str(run), **result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--worker', choices=MODES)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--conditions', nargs='+', choices=MODES, default=DEFAULT_MODES)
    parser.add_argument('--task-ids', nargs='+', type=int, choices=range(10), default=[2])
    parser.add_argument('--task-id', type=int, choices=range(10), default=2, help='isolated worker task')
    args = parser.parse_args()
    if args.worker:
        return worker(args.reference_run, args.worker, args.parent, args.task_id)
    config = json.loads((args.reference_run / 'run.json').read_text())['config_snapshot']
    if (config['H'], config['N'], config['screen']['task_ids']) != (20, 5, [2]):
        raise ValueError('diagnostic is explicitly scoped to H20/N5/Spatial task2')
    parent = create_run(config, 'M1', 'rtc-handoff-comparison', repo_root=ROOT)
    print('comparison_dir=' + str(parent), flush=True)
    results = []
    combinations = [(task_id, mode) for task_id in args.task_ids for mode in args.conditions]
    for task_id, mode in combinations:
        directory = parent / f'task{task_id:02d}' if len(args.task_ids) > 1 else parent
        directory.mkdir(exist_ok=True)
        command = [sys.executable, str(ROOT / 'scripts/guarded_run.py'), '--',
                   sys.executable, '-u', str(Path(__file__).resolve()),
                   '--reference-run', str(args.reference_run), '--worker', mode,
                   '--parent', str(directory), '--task-id', str(task_id)]
        with (directory / (mode + '.log')).open('w') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        item = {'condition': mode, 'task_id': task_id, 'exit_code': code}
        path = directory / (mode + '.json')
        if path.exists():
            item.update(json.loads(path.read_text()))
        results.append(item)
        status = 'running' if code == 0 and len(results) < len(combinations) else 'completed' if code == 0 else 'stopped_invalid'
        successes = sum(r.get('episode', {}).get('status') == 'success' for r in results)
        runner.base._write_json_atomic(parent / 'summary.json', dict(status=status, expected=len(combinations),
                                       successes=successes, conditions=results))
        runner.base._save_run_status(parent, status)
        print(json.dumps(item), flush=True)
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
