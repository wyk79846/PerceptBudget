"""Durable ten-task actual-delay RTC screen; inference lives in existing worker."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perceptbudget.experiment import create_run, load_config
from scripts import eval_q as base


def read_case(path, task_id):
    record = json.loads((path/'run.json').read_text())['config_snapshot']
    result = json.loads((path/'summary.json').read_text())
    expected = dict(H=20, N=5, state_enabled=True, state_prompt_capacity=80,
                    rtc_paper_schedule=True, rtc_delay_buffer_size=10,
                    pending_arm_gain=1.0, rtc_native_fused_ffn=True,
                    rtc_fused_attention_ops=True, candidate_q=[5])
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('worker settings mismatch')
    if 'controlled_delay_steps' in record or result['status'] != 'completed':
        raise ValueError('not a completed actual-delay worker')
    if result['task']['task_id'] != task_id or result['task']['suite'] != 'libero_spatial':
        raise ValueError('wrong task')
    episode = result['conditions']['rtc']
    if not result['preflight_reference_audit']['passed'] or not episode['actual_delay_audit_passed']:
        raise ValueError('worker audit failed')
    return dict(task_id=task_id, source_run=str(path), **{
        k: episode[k] for k in ('status', 'steps', 'real_seconds', 'steady_inference_ms',
                                'post_bootstrap_control_hz', 'underflow_steps')})


def save(run, summary):
    cases = summary['cases']
    summary['valid_tasks'] = len(cases)
    summary['successes'] = sum(c['status'] == 'success' for c in cases)
    base._write_json_atomic(run/'summary.json', summary)
    lines = ['# RTC真实延迟异步：Spatial十任务', '',
             'H20/N5/BF16/state/容量80；算法1，s_min=5，最近10次实际延迟取max。',
             '原速动作；不固定推理步数，不人为冻结仿真。实际频率另记，不保证20Hz。',
             f"状态：{summary['status']}；有效{len(cases)}/10；成功{summary['successes']}。", '',
             '|任务|结果|步数|真实秒|推理中位ms|实际Hz|断供步|',
             '|---|---|---:|---:|---:|---:|---:|']
    for c in cases:
        lines.append('|{}|{}|{}|{:.3f}|{:.2f}|{:.2f}|{}|'.format(
            c['task_id'], c['status'], c['steps'], c['real_seconds'],
            c['steady_inference_ms']['median'], c['post_bootstrap_control_hz'], len(c['underflow_steps'])))
    lines += ['', '每任务一次、seed7/init0；全部新测，不复用此前三项。超时秒数不是成功完成时间。']
    (run/'summary.md').write_text('\n'.join(lines)+'\n')
    base._save_run_status(run, summary['status'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    config = load_config(repo_root=ROOT)
    if args.resume_from:
        run = args.resume_from
        summary = json.loads((run/'summary.json').read_text())
        for c in summary['cases']:
            if read_case(Path(c['source_run']), c['task_id']) != c:
                raise ValueError('resume source changed')
    else:
        config.update(H=20, N=5, state_enabled=True, state_prompt_capacity=80,
                      execution_mode='rtc_paper_schedule_actual_delay_screen', candidate_q=[5])
        run = create_run(config, 'M1', 'rtc-async-spatial-ten', repo_root=ROOT)
        (run/'workers').mkdir()
        shutil.copy2(__file__, run/'runner.py')
        summary = dict(status='running', expected_tasks=10, cases=[])
    summary['status'] = 'running'
    print('run_dir='+str(run), flush=True)
    save(run, summary)
    for task_id in range(10):
        if any(c['task_id'] == task_id for c in summary['cases']):
            continue
        log = run/'workers'/f'task{task_id:02d}_{len(list((run/"workers").iterdir())):02d}.log'
        command = [config['python_executable'], '-u', str(ROOT/'scripts/guarded_run.py'), '--',
            config['python_executable'], '-u', str(ROOT/'scripts/eval_rtc_controlled.py'),
            '--actual-delay', '--paper-schedule', '--rtc-only', '--suite', 'libero_spatial',
            '--task-id', str(task_id), '--q', '5', '--denoising-steps', '5',
            '--attention-backend', 'flash-varlen', '--deterministic-attention', '--state',
            '--preflight-rtc', '--native-fused-ffn', '--fused-attention-ops', '--state-prompt-capacity', '80']
        print(f'task={task_id} start', flush=True)
        with log.open('w') as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            paths = [s.split('=', 1)[1] for s in log.read_text().splitlines() if s.startswith('run_dir=')]
            if proc.returncode or len(paths) != 1:
                raise RuntimeError(f'worker exit={proc.returncode}, run paths={len(paths)}')
            summary['cases'].append(read_case(Path(paths[0]), task_id))
        except Exception as exc:
            summary.update(status='stopped_invalid', error=str(exc), failed_task=task_id, worker_log=str(log))
            save(run, summary)
            raise
        save(run, summary)
        print(f'task={task_id} result={summary["cases"][-1]["status"]}', flush=True)
    summary['status'] = 'completed'
    save(run, summary)


if __name__ == '__main__':
    main()
