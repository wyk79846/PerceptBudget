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
MODES = DEFAULT_MODES + ('controlled_d8','actual_history')


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
    config.update(rtc_paper_schedule=mode in ('actual_est8', 'controlled_d8','actual_history'),
                  rtc_fixed_delay_estimate=8 if mode in ('actual_est8', 'controlled_d8') else None,
                  explicit_actual_delay_experiment=mode in ('actual_est8','actual_history'),
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


def worker(reference_run, mode, parent, task_id=2, beta=5., q=5, horizon=20, checkpoint_profile='old', *, exact_flash=False, audit_attention=False, suffix_vjp=False, fused_norm=False, audit_norm=False, gemm_policy=None):
    reference = json.loads((reference_run / 'run.json').read_text())['config_snapshot']
    config = configuration(reference, mode)
    if gemm_policy is not None:
        config['native_gemm_policy']=gemm_policy
    config['rtc_beta'] = beta
    config['candidate_q'] = [q]
    config['H'] = horizon
    config['rtc_exact_flash_attention'] = bool(exact_flash)
    config['rtc_suffix_only_vjp'] = bool(suffix_vjp)
    config['rtc_audit_attention_same_input'] = bool(audit_attention)
    config['rtc_fused_norm']=bool(fused_norm)
    config['rtc_audit_fused_norm']=bool(audit_norm)
    if exact_flash:
        config['attention_validation_scope'] = 'same Flash SDPA backend eager/graph gate; not original eager numerical parity'
    if suffix_vjp:
        config['attention_validation_scope'] = 'same suffix-only eager VJP backend eager/graph gate; not original VJP numerical parity'
    if checkpoint_profile=='h50':
        from perceptbudget.h50_checkpoint import configure
        configure(config)
        if mode=='actual_history':
            # Initial prior from the same model/task's measured unpaused ticks;
            # online history thereafter contains actual consumed control steps.
            import numpy as np
            source=Path(config['output_root'])/'M1/20261020T074138.621142Z-rtc-diag-controlled_d5-c11f6716'
            controls=[json.loads(x) for x in (source/'rtc/trace.jsonl').open()]
            controls=[r for r in controls if r['event']=='control_step']
            intervals=[b['tick_monotonic']-a['tick_monotonic'] for a,b in zip(controls,controls[1:])
                       if a['step']%5!=0]
            config['rtc_initial_control_period_seconds']=float(np.median(intervals))
            config['rtc_initial_control_period_source']=str(source)
    if horizon != reference['H']:
        config['reference_horizon_transfer'] = dict(reference_H=25,experiment_H=horizon,
            limitation='explicit diagnostic horizon change; live exact-length graph/eager gate, not independent author-body horizon validation')
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
                exact = list(range(max(1, valid-3), min(712, valid+4)+1))
                adapter.config['source_step_prefixes'] = exact
                return prepare(adapter, exact)

            original = probe.run_episode
            with ExitStack() as stack:
                if task_id != 2 or checkpoint_profile=='h50':
                    stack.enter_context(patch.object(OriginalOpenPiRTCAdapter, 'prepare_step_bank', prepare_for_task))
                if mode == 'sync_no_rtc':
                    stack.enter_context(patch.object(probe, 'run_episode', lambda *a, **k: sync_episode(original, *a, **k)))
                result = runner.rollout(config, run, gate, task_id)
            if mode == 'sync_no_rtc':
                result['scope'] = 'synchronous original expert, no RTC; simulation waits for every inference'
            if mode == 'controlled_d8':
                result['scope'] = 'fixed eight simulated delay steps; pauses simulation if inference late; not real-time async'
                req = json.loads((run / f'rtc/q{q}_requests.json').read_text())
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
    parser.add_argument('--beta',type=float,choices=(1.,2.,5.,10.),default=5.,help='explicit RTC guidance cap; default unchanged')
    parser.add_argument('--q',type=int,choices=(5,8,15),default=5,help='minimum control-step interval between inference starts')
    parser.add_argument('--horizon',type=int,choices=(10,20,50),default=20,help='explicit action horizon experiment; does not change defaults')
    parser.add_argument('--checkpoint-profile',choices=('old','h50'),default='old')
    parser.add_argument('--gemm-policy',choices=('heuristic-first','autotune'),default=None,
                        help='explicit native GEMM selection policy; inherited when omitted')
    parser.add_argument('--exact-flash-attention',action='store_true',help='experimental original expert with exact-context Flash SDPA')
    parser.add_argument('--suffix-only-vjp',action='store_true',help='experimental eager attention with constant-prefix KV gradient elimination; not combined with Flash SDPA')
    parser.add_argument('--fused-norm',action='store_true',help='opt-in FP32-style AdaRMS fusion on exact FlashAttention')
    parser.add_argument('--audit-norm-inputs',action='store_true',help='same-input norm-fusion audit against retained FlashAttention before rollout')
    parser.add_argument('--audit-attention-inputs',action='store_true',help='initial same-KV/noise/target comparison before candidate rollout')
    args = parser.parse_args()
    if args.fused_norm and not args.exact_flash_attention:
        parser.error('fused norm requires --exact-flash-attention')
    if args.audit_norm_inputs and not args.fused_norm:
        parser.error('norm audit requires --fused-norm')
    if args.exact_flash_attention and args.suffix_only_vjp:
        parser.error('attention backends are alternatives; choose only one')
    if args.audit_attention_inputs and not (args.exact_flash_attention or args.suffix_only_vjp):
        parser.error('attention input audit requires an explicit candidate backend')
    selected=[args.worker] if args.worker else args.conditions
    delay=max({'sync_no_rtc':0,'controlled_d5':5,'controlled_d8':8,'actual_est8':8,'actual_history':0}[mode] for mode in selected)
    if args.q+delay>args.horizon:
        parser.error('q plus fixed delay exceeds horizon')
    if args.worker:
        return worker(args.reference_run, args.worker, args.parent, args.task_id,args.beta,args.q,args.horizon,args.checkpoint_profile,
                      **dict((k,v) for k,v in (('exact_flash',args.exact_flash_attention),
                                              ('audit_attention',args.audit_attention_inputs),
                                              ('suffix_vjp',args.suffix_only_vjp),
                                              ('fused_norm',args.fused_norm),('audit_norm',args.audit_norm_inputs),
                                              ('gemm_policy',args.gemm_policy)) if v))
    config = json.loads((args.reference_run / 'run.json').read_text())['config_snapshot']
    if (config['H'], config['N'], config['screen']['task_ids']) != (20, 5, [2]):
        raise ValueError('diagnostic is explicitly scoped to H20/N5/Spatial task2')
    config['rtc_beta']=args.beta
    if args.gemm_policy is not None:
        config['native_gemm_policy']=args.gemm_policy
    config['candidate_q']=[args.q]
    config['H']=args.horizon
    config['rtc_exact_flash_attention']=bool(args.exact_flash_attention)
    config['rtc_suffix_only_vjp']=bool(args.suffix_only_vjp)
    config['rtc_audit_attention_same_input']=bool(args.audit_attention_inputs)
    config['rtc_fused_norm']=bool(args.fused_norm)
    config['rtc_audit_fused_norm']=bool(args.audit_norm_inputs)
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
                   '--parent', str(directory), '--task-id', str(task_id),'--beta',str(args.beta),'--q',str(args.q),
                   '--horizon',str(args.horizon)]
        command+=['--checkpoint-profile',args.checkpoint_profile]
        if args.gemm_policy is not None:command+=['--gemm-policy',args.gemm_policy]
        if args.exact_flash_attention:
            command+=['--exact-flash-attention']
        if args.suffix_only_vjp:
            command+=['--suffix-only-vjp']
        if args.audit_attention_inputs:
            command+=['--audit-attention-inputs']
        if args.fused_norm:command+=['--fused-norm']
        if args.audit_norm_inputs:command+=['--audit-norm-inputs']
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
