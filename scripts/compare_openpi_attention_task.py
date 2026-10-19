"""One matched controlled-delay task: original vs attention-only candidate."""
import json
from pathlib import Path
import subprocess
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import create_run,load_config
from scripts import eval_q as base


def main():
    config=load_config(repo_root=ROOT)
    config.update(H=20,N=5,state_enabled=True,state_prompt_capacity=80,
                  controlled_delay_steps=5,task_id=0)
    run=create_run(config,'M1','attention-controlled-pair',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    shutil.copy2(__file__,run/'compare_openpi_attention_task.py')
    gate=Path(config['output_root'])/'M1/20261019T092705.748264Z-openpi-rtc-gate-d9835f75'
    result=dict(status='running',scope='Spatial0 H20 N5 state on q5 d5 controlled wait; not actual async or historical exact reproduction',conditions=[])
    try:
        for name,flags in (('original',[]),('varlen',['--experimental-varlen','--cuda-graph'])):
            print('starting='+name,flush=True)
            log=run/(name+'.log')
            command=[sys.executable,'-u','scripts/eval_rtc_openpi.py','--verified-run',str(gate),
                     '--controlled-delay-five','--task-id','0',*flags]
            with log.open('w') as stream:
                process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            line=next((s for s in log.read_text().splitlines() if s.startswith('run_dir=')),None)
            child=Path(line.split('=',1)[1]) if line else None
            summary=json.loads((child/'summary.json').read_text()) if child and (child/'summary.json').exists() else {}
            result['conditions'].append(dict(name=name,returncode=process.returncode,
                source_run=str(child),summary=summary))
            base._write_json_atomic(run/'summary.json',result)
            print('finished='+name+' '+json.dumps(dict(status=summary.get('status'),
                episode_status=summary.get('episode',{}).get('status'),
                steps=summary.get('episode',{}).get('steps'),
                steady_inference_median_ms=summary.get('steady_inference_median_ms'))),flush=True)
            if process.returncode or summary.get('status')!='completed':
                raise RuntimeError('condition failed; stop without discarding attempted result')
        result['status']='completed'
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        base._write_json_atomic(run/'summary.json',result)


if __name__=='__main__':
    main()
