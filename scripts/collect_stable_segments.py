"""Offline Continue-segment dataset from saved development traces; no VLA load."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from perceptbudget.stable_segments import WINDOW,FEATURE_NAMES,PhysicalTeacher,online_features,phase_flags


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def process_case(row,role,folder):
    import numpy as np
    run=Path(row['run']);teacher=PhysicalTeacher();history=[];samples=[];review=[];events=[]
    counters=Counter();decision=None;last_normal=-WINDOW;first_state=None
    for line in (run/'rtc/trace.jsonl').open():
        record=json.loads(line)
        if record.get('event')=='m3_decision':decision=record;continue
        if record.get('event')!='control_step':continue
        if first_state is None:first_state=record['before']['raw_state']
        flags=phase_flags(record,decision);physical=teacher.observe(record)
        feature=online_features(record);history.append((record['step'],feature))
        counters['control_steps']+=1
        counters['continue_candidate_steps']+=int(flags['eligible'])
        counters['ordinary_declared_steps']+=int(flags['ordinary_declared'])
        for event in physical['events']:
            events.append(dict(event,phase=flags))
        if not flags['eligible'] or len(history)<WINDOW:continue
        label=1 if physical['events'] else 0 if physical['normal'] else -1
        if label==0 and record['step']-last_normal<WINDOW:continue
        if label==0:last_normal=record['step']
        if label<0:
            counters['unlabeled_steps']+=1
            if physical['ambiguous_closed_without_confirmed_object'] and len(review)<8:
                review.append(dict(step=record['step'],observation_step=record['step']+1,
                    reason='closed_without_confirmed_object',phase=flags))
            continue
        samples.append(dict(index=len(samples),step=record['step'],observation_step=record['step']+1,
            history_start_step=history[-WINDOW][0],label=label,
            label_kind=physical['events'][0]['kind'] if label else physical['normal_kind'],
            phase=flags,teacher=physical,features=[r[1] for r in history[-WINDOW:]]))
    positives=[s for s in samples if s['label']==1]
    normals=[s for s in samples if s['label']==0]
    # Limit repetitive normal frames; choose evenly without task outcome labels.
    if len(normals)>32:
        normals=[normals[round(i*(len(normals)-1)/31)] for i in range(32)]
    selected=sorted(positives+normals,key=lambda s:s['step'])
    features=np.asarray([s.pop('features') for s in selected],dtype=np.float32).reshape(-1,WINDOW,len(FEATURE_NAMES))
    labels=np.asarray([s['label'] for s in selected],dtype=np.int8)
    np.savez_compressed(folder/'samples.npz',features=features,labels=labels,
        observation_steps=np.asarray([s['observation_step'] for s in selected],dtype=np.int32))
    details=dict(seed=row['seed'],suite=row['suite'],task_id=row['task_id'],split=role,
        source_run=str(run),episode_outcome=row['outcome'],counts=dict(counters),samples=selected,
        physical_events=events,review=review,
        positives=len(positives),negatives=len(normals),
        ordinary_positive_samples=sum(s['label']==1 and s['phase']['ordinary_declared'] for s in selected),
        teacher_truth_scope='Local feedback evidence; not a task-failure timestamp oracle')
    save(folder/'case.json',details)
    return details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--render-review',action='store_true')
    args=parser.parse_args();root=args.run_dir;root.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'plan.json').read_text());rows=json.loads(Path(plan['index']).read_text())['rows']
    prior=json.loads(Path(plan['split_reference']).read_text())['episodes']
    roles={e['source_run']:e['safe_split'] for e in prior}
    started=time.monotonic();cases=[];errors=[]
    def progress(status,stage):
        stats=dict(status=status,stage=stage,completed=len(cases)+len(errors),total=len(rows),
            episodes_ok=len(cases),errors=len(errors),elapsed_seconds=time.monotonic()-started,
            positive_samples=sum(c['positives'] for c in cases),normal_samples=sum(c['negatives'] for c in cases),
            physical_events=sum(len(c['physical_events']) for c in cases),model_loads=0)
        save(root/'progress.json',stats)
        save('/tmp/perceptbudget-stable-segments-current.json',dict(run=str(root),**stats))
        (root/'progress.md').write_text('# 平稳执行片段：离线数据整理\n\n'
            f"状态：{status}；阶段：{stage}；完成{stats['completed']}/{stats['total']}回合。\n\n"
            f"反馈异常样本{stats['positive_samples']}；明确正常样本{stats['normal_samples']}；解析错误{stats['errors']}。\n\n"
            '使用已有轨迹，不加载VLA、不更改v4。仿真物体/接触信息仅作为标签，在线特征与标签分开保存。\n')
        print('STABLE_DATA_PROGRESS='+json.dumps(stats,ensure_ascii=False),flush=True)
    progress('running','extracting')
    for i,row in enumerate(rows):
        folder=root/'episodes'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=True)
        try:
            if (folder/'case.json').exists():case=json.loads((folder/'case.json').read_text())
            else:case=process_case(row,roles[row['run']],folder)
            case['case_id']=i;cases.append(case)
        except Exception as exc:
            errors.append(dict(case_id=i,run=row['run'],error=type(exc).__name__+': '+str(exc)))
        progress('running','extracting')
    save(root/'errors.json',errors)
    samples=[];review=[]
    for c in cases:
        for s in c['samples']:
            samples.append(dict(case_id=c['case_id'],seed=c['seed'],suite=c['suite'],task_id=c['task_id'],split=c['split'],
                source_run=c['source_run'],**s))
        for s in c['review']:review.append(dict(case_id=c['case_id'],source_run=c['source_run'],suite=c['suite'],task_id=c['task_id'],**s))
    quotas=plan.get('sample_quotas',dict(libero_10=120,libero_spatial=40,libero_object=40,libero_goal=40))
    chosen=[];shortfalls={}
    for suite,quota in quotas.items():
        pool=[s for s in samples if s['suite']==suite]
        positives=[s for s in pool if s['label']==1]
        negatives=[s for s in pool if s['label']==0]
        # Preserve every available positive before distributing normals across episodes.
        take=positives[:quota]
        groups={}
        for s in negatives:groups.setdefault(s['case_id'],[]).append(s)
        for offset in range(max((len(v) for v in groups.values()),default=0)):
            for group in groups.values():
                if len(take)>=quota:break
                if offset<len(group):take.append(group[offset])
            if len(take)>=quota:break
        chosen.extend(take);shortfalls[suite]=max(0,quota-len(take))
    save(root/'selected_samples.json',chosen)
    save(root/'manifest.json',dict(plan=plan,feature_names=FEATURE_NAMES,window_steps=WINDOW,
        online_feature_scope='Only causal robot state, velocity, applied action and control dt; no object/contact/task identity input',
        label_scope='1 local physical cue, refresh benefit unverified;0 open-free or consistent-carry evidence; ambiguous omitted',
        cases=[{k:c[k] for k in ('case_id','seed','suite','task_id','split','source_run','episode_outcome','positives','negatives','counts')} for c in cases],
        samples=samples,review=review))
    selected=[]
    for kind in ('positive','normal','uncertain'):
        pool=([s for s in samples if s['label']==1] if kind=='positive' else [s for s in samples if s['label']==0] if kind=='normal' else review)
        by_case={}
        for s in pool:by_case.setdefault(s['case_id'],s)
        buckets={}
        for s in by_case.values():buckets.setdefault(s['suite'],[]).append(s)
        balanced=[]
        for offset in range(max((len(v) for v in buckets.values()),default=0)):
            for bucket in buckets.values():
                if len(balanced)>=16:break
                if offset<len(bucket):balanced.append(bucket[offset])
            if len(balanced)>=16:break
        selected.extend(dict(s,review_kind=kind) for s in balanced)
    save(root/'review_selection.json',selected)
    summary=dict(episodes=len(cases),errors=len(errors),positive_samples=sum(s['label']==1 for s in samples),
        normal_samples=sum(s['label']==0 for s in samples),
        ordinary_positive_samples=sum(s['label']==1 and s['phase']['ordinary_declared'] for s in samples),
        positive_kinds=dict(Counter(s['label_kind'] for s in samples if s['label']==1)),
        by_split={role:dict(positive=sum(s['split']==role and s['label']==1 for s in samples),normal=sum(s['split']==role and s['label']==0 for s in samples))
            for role in ('train','validation','test')},
        review_items=len(review),model_loads=0,policy_modified=False,automatic_labels_require_review=True,
        selected_samples=len(chosen),sample_quotas=quotas,quota_shortfalls=shortfalls,
        selected_counts=dict(Counter(s['suite'] for s in chosen)),refresh_benefit_validated=False,training_started=False)
    save(root/'summary.json',summary)
    with (root/'samples.csv').open('w') as stream:
        fields=['case_id','seed','suite','task_id','split','step','observation_step','label','label_kind','source_run']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(samples)
    if args.render_review and selected:
        progress('running','rendering_review')
        result=subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/render_stable_review.py'),'--run-dir',str(root)],cwd=PROJECT)
        summary['render_exit_code']=result.returncode
    status='completed' if not errors and summary.get('render_exit_code',0)==0 else 'needs_review'
    summary['status']=status;summary['elapsed_seconds']=time.monotonic()-started
    save(root/'summary.json',summary)
    lines=['# 平稳执行片段：首批数据','',f"读取{len(cases)}/{len(rows)}回合；反馈异常样本{summary['positive_samples']}，明确正常样本{summary['normal_samples']}，其中严格ordinary标记内异常{summary['ordinary_positive_samples']}。",'',
        '标签只表示局部物理线索，尚未验证刷新是否有益；不使用回合最终成败替代局部标签。物体位置和接触真值仅供离线标注；在线特征不包含它们。',
        '主筛选为无推理、正在等待刷新且离已知关键步超过2步的Continue片段；另保留原控制器ordinary标记，二者分开统计。',
        '正常标签仅覆盖张爪自由运动或物体与夹爪保持一致运动；其余未确定片段不强行标正常。','',
        '|划分|反馈异常|明确正常|','|---|---:|---:|']
    for role,v in summary['by_split'].items():lines.append(f"|{role}|{v['positive']}|{v['normal']}|")
    lines+=['','[样本索引](samples.csv) · [完整元数据](manifest.json) · [待抽查样本](review_selection.json) · [错误记录](errors.json)',
        '[首批240配额样本](selected_samples.json)',
        '', '图像补采只恢复已保存的仿真状态渲染，未重新运行VLA策略。若渲染失败，见render_status.json；不把旧请求画面充作当前观测。',
        '数据生成完成不等于自动标签已人工审定，也不等于已训练或验证在线异常检测器。']
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    progress(status,'finished')


if __name__=='__main__':main()
