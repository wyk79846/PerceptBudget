"""Four explicitly paired tasks: old/new checkpoints, H20, q5/q8, d5 RTC."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run
from scripts import eval_q as base


def main():
    config=load_config(repo_root=ROOT)
    config.update(H=20,N=5,candidate_q=[5,8],state_enabled=True,
                  checkpoint_comparison=['old','h50'],controlled_delay_steps=5,rtc_beta=5.)
    parent=create_run(config,'M1','rtc-checkpoint-comparison',repo_root=ROOT)
    print('comparison_dir='+str(parent),flush=True)
    reference=Path(config['output_root'])/'M1/20261020T035931.024165Z-openpi-rtc-actual-cfaef1e3'
    summary=dict(status='running',expected=4,conditions=[])
    for profile in ('h50','old'):
        for q in (5,8):
            dest=parent/f'{profile}_q{q}'
            dest.mkdir()
            command=[sys.executable,str(ROOT/'scripts/guarded_run.py'),'--',sys.executable,'-u',
                     str(ROOT/'scripts/diagnose_rtc_handoff.py'),'--reference-run',str(reference),
                     '--worker','controlled_d5','--parent',str(dest),'--task-id','0',
                     '--horizon','20','--q',str(q),'--beta','5','--checkpoint-profile',profile]
            with (dest/'worker.log').open('w') as stream:
                code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
            item=dict(profile=profile,q=q,exit_code=code)
            result=dest/'controlled_d5.json'
            if result.exists(): item.update(json.loads(result.read_text()))
            summary['conditions'].append(item)
            summary['status']='stopped_invalid' if code else 'completed' if len(summary['conditions'])==4 else 'running'
            base._write_json_atomic(parent/'summary.json',summary)
            base._save_run_status(parent,summary['status'])
            e=item.get('episode',{})
            print(json.dumps(dict(profile=profile,q=q,status=item.get('status'),outcome=e.get('status'),
                                  steps=e.get('steps'),seconds=e.get('real_seconds'),run=item.get('run'))),flush=True)
            if code: return code
    return 0


if __name__=='__main__':
    raise SystemExit(main())
