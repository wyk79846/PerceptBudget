"""Durable paired Spatial screen, q5/N5/state/d5; reuse verified task0/1."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import create_run,load_config
from scripts import eval_q as base

REUSE={
 (0,'baseline'):'20261018T151118.052027Z-rtc-controlled-d5-q5-state-30d6fa72',
 (0,'fused'):'20261018T155540.802683Z-rtc-controlled-d5-q5-state-6ee45566',
 (1,'baseline'):'20261018T155811.642732Z-rtc-controlled-d5-q5-state-88813eff',
 (1,'fused'):'20261018T155940.284624Z-rtc-controlled-d5-q5-state-47de848b',
}


def read_case(path,task_id,mode,config):
    saved=json.loads((path/'run.json').read_text())['config_snapshot']
    keys=('H','N','state_enabled','controlled_delay_steps','candidate_q',
          'rtc_attention_backend','rtc_deterministic_attention','checkpoint_path',
          'flashrt_revision','rtc_validation_before_task')
    for key in keys:
        if saved.get(key)!=config.get(key): raise ValueError(f'incompatible reuse: {path} {key}')
    if bool(saved.get('rtc_native_fused_ffn',False))!=(mode=='fused'):
        raise ValueError('wrong fused mode')
    for key in ('seed','init_state_index'):
        if saved['screen'][key]!=config['screen'][key]: raise ValueError('wrong initial condition')
    s=json.loads((path/'summary.json').read_text())
    if s['status']!='completed' or s['task']['task_id']!=task_id or s['task']['suite']!='libero_spatial':
        raise ValueError('incomplete or wrong task')
    e=s['conditions']['rtc']
    if not s['preflight_reference_audit']['passed'] or not e['controlled_delay_audit_passed']:
        raise ValueError('audit failed')
    return {'task_id':task_id,'mode':mode,'source_run':str(path),
            'status':e['status'],'steps':e['steps'],'real_seconds':e['real_seconds'],
            'steady_inference_ms':e['steady_inference_ms'],'underflows':len(e['underflow_steps']),
            'initial_sim_state_sha256':e['initial_sim_state_sha256'],
            'initial_frames_sha256':e['initial_frames_sha256']}


def save(run,result):
    cases=result['cases']
    result['valid_cases']=len(cases)
    result['successes']={m:sum(c['status']=='success' for c in cases if c['mode']==m)
                         for m in ('baseline','fused')}
    rows=['# RTC Spatial：未融合与融合成对比较','',
          'H20/N5/q5/d5/state；受控延迟，非真实20Hz。旧 flash 后端9/10不混入本组。',
          f"状态：{result['status']}；有效 {len(cases)}/20。",'',
          '| task | 未融合结果/步 | 融合结果/步 | 未融合运行秒 | 融合运行秒 | 推理中位ms（未融合/融合） |',
          '|---:|---|---|---:|---:|---|']
    for task_id in range(10):
        pair={c['mode']:c for c in cases if c['task_id']==task_id}
        if len(pair)==2:
            a,b=pair['baseline'],pair['fused']
            for key in ('initial_sim_state_sha256','initial_frames_sha256'):
                if a[key]!=b[key]: raise ValueError(f'paired initial condition mismatch task{task_id}: {key}')
        def cell(m,key,default='—'): return pair.get(m,{}).get(key,default)
        def desc(m): return f"{cell(m,'status')}/{cell(m,'steps')}"
        ms=[pair[m]['steady_inference_ms'].get('median') if m in pair else None for m in ('baseline','fused')]
        rows.append(f"| {task_id} | {desc('baseline')} | {desc('fused')} | {cell('baseline','real_seconds')} | {cell('fused','real_seconds')} | {ms} |")
    rows+=['','超时运行秒不是完成时间。每任务各一次，不把轨迹步数变化全部归因于计算提速。',
           'task0/1复用已完成且配置匹配的独立记录，其余按任务交错后端顺序；来源见summary.json。']
    base._write_json_atomic(run/'summary.json',result)
    (run/'summary.md').write_text('\n'.join(rows)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args()
    config=load_config(repo_root=ROOT)
    config.update(H=20,N=5,state_enabled=True,controlled_delay_steps=5,candidate_q=[5],
        rtc_attention_backend='flash-varlen',rtc_deterministic_attention=True,rtc_validation_before_task=True)
    config['screen'].update(seed=7,init_state_index=0,suites=['libero_spatial'])
    if args.resume_from:
        run=args.resume_from
        result=json.loads((run/'summary.json').read_text())
        result['status']='running'
    else:
        run=create_run(config,'M1','rtc-ffn-paired-spatial',repo_root=ROOT)
        result={'status':'running','expected_cases':20,'cases':[]}
        for (task_id,mode),name in REUSE.items():
            result['cases'].append(read_case(Path(config['output_root'])/'M1'/name,task_id,mode,config))
        (run/'workers').mkdir()
        shutil.copy2(__file__,run/'runner.py')
    print('run_dir='+str(run),flush=True)
    save(run,result)
    for task_id in range(10):
        modes=('baseline','fused') if task_id%2==0 else ('fused','baseline')
        for mode in modes:
            if any(c['task_id']==task_id and c['mode']==mode for c in result['cases']): continue
            log=run/'workers'/f'task{task_id:02d}_{mode}_{len(list((run/"workers").iterdir())):02d}.log'
            command=[config['python_executable'],'-u',str(ROOT/'scripts/guarded_run.py'),'--',
                config['python_executable'],'-u',str(ROOT/'scripts/eval_rtc_controlled.py'),
                '--rtc-only','--suite','libero_spatial','--task-id',str(task_id),'--q','5',
                '--denoising-steps','5','--attention-backend','flash-varlen',
                '--deterministic-attention','--state','--preflight-rtc']
            if mode=='fused': command.append('--native-fused-ffn')
            print(f'task={task_id} mode={mode} start',flush=True)
            with log.open('w') as stream:
                proc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            try:
                paths=[line.split('=',1)[1] for line in log.read_text().splitlines() if line.startswith('run_dir=')]
                if proc.returncode or len(paths)!=1: raise ValueError(f'worker exit {proc.returncode}')
                case=read_case(Path(paths[0]),task_id,mode,config)
                result['cases'].append(case)
                save(run,result)
                print(f"task={task_id} mode={mode} result={case['status']} steps={case['steps']}",flush=True)
            except Exception as exc:
                result.update(status='stopped_invalid',error=str(exc),failed_worker_log=str(log))
                save(run,result)
                raise
    result['status']='completed'
    save(run,result)
    print('completed '+json.dumps(result['successes']),flush=True)


if __name__=='__main__': main()
