"""Five fresh paired Spatial tasks, original vs exact FlashAttention RTC.

Explicit H50/N5/q15/state/seed7/init0. Sequential isolated guarded workers,
alternating order, durable progress, no changes to production defaults.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run
from scripts import eval_q as base
from scripts.audit_rtc_system import audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    config=load_config(repo_root=ROOT)
    config.update(H=50,N=5,candidate_q=[15],state_enabled=True,
                  paired_attention_tasks=list(range(5)),paired_attention_backends=['original','flash'])
    parent=args.resume.resolve() if args.resume else create_run(config,'M1','rtc-attention-paired-five',repo_root=ROOT)
    print('comparison_dir='+str(parent),flush=True)
    path=parent/'summary.json'
    summary=json.loads(path.read_text()) if args.resume else dict(status='running',expected=10,records=[])
    reference=Path(config['output_root'])/'M1/20261020T035931.024165Z-openpi-rtc-actual-cfaef1e3'
    for task in range(5):
        for backend in (('original','flash') if task%2==0 else ('flash','original')):
            if any(r['task_id']==task and r['backend']==backend and r.get('status')=='completed' for r in summary['records']):
                continue
            dest=parent/f'task{task:02d}_{backend}'
            dest.mkdir(exist_ok=True)
            command=[sys.executable,str(ROOT/'scripts/guarded_run.py'),'--',sys.executable,'-u',
                str(ROOT/'scripts/diagnose_rtc_handoff.py'),'--reference-run',str(reference),
                '--worker','actual_history','--parent',str(dest),'--task-id',str(task),
                '--horizon','50','--q','15','--checkpoint-profile','h50']
            if backend=='flash':command+=['--exact-flash-attention','--audit-attention-inputs']
            # Preserve logs from invalid attempts on an explicit resume.
            with (dest/'worker.log').open('a') as log:
                code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            item=dict(task_id=task,backend=backend,exit_code=code,status='invalid')
            result=dest/'actual_history.json'
            if result.exists():item.update(json.loads(result.read_text()))
            if code:
                item['status']='invalid'
            if code==0 and item.get('status')=='completed':
                checked=audit(Path(item['run']))
                base._write_json_atomic(Path(item['run'])/'system_audit.json',checked)
                item['system_audit']=checked
                if not checked['passed']:
                    item.update(status='invalid',error='independent system audit failed')
            summary['records'].append(item)
            valid=[r for r in summary['records'] if r.get('status')=='completed']
            summary.update(status='stopped_invalid' if item['status']!='completed' else
                           'completed' if len(valid)==10 else 'running',completed=len(valid),
                successes={b:sum(r.get('episode',{}).get('status')=='success' for r in valid if r['backend']==b)
                           for b in ('original','flash')})
            if item['status']=='completed':
                peer=next((r for r in valid if r['task_id']==task and r['backend']!=backend),None)
                if peer:
                    equal=all(item['episode'][k]==peer['episode'][k] for k in
                              ('initial_sim_state_sha256','initial_frames_sha256'))
                    item['paired_initial_hashes_equal']=equal
                    if not equal:
                        summary['status']='stopped_pair_mismatch'
            base._write_json_atomic(path,summary)
            base._save_run_status(parent,summary['status'])
            e=item.get('episode',{})
            print(json.dumps(dict(task_id=task,backend=backend,status=item['status'],outcome=e.get('status'),
                steps=e.get('steps'),seconds=e.get('real_seconds'),median_ms=item.get('steady_inference_median_ms'),
                completed=summary['completed'],run=item.get('run'))),flush=True)
            if summary['status'].startswith('stopped'):return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
