"""Train SAFE-MLP on cached endpoints using the official implementation.

No policy/model inference, simulation, frequency changes or external logging.
The135 candidate grid is the pi0/LIBERO grid from the author's batch script.
Only seen-task validation chooses the model; tests are scored after selection.
"""
import argparse
from collections import Counter
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from perceptbudget.safe_paper import (AUTHOR_ROOT,CHOICES,aggregate_corners,compact_loss,
    compact_weights,conformal_bands,make_model,paper_split)
from scripts.analyze_gripper_timing import save
from scripts.train_failure_probe import auc,metrics


def train(feature_root,root,epochs=1000):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MALLOC_ARENA_MAX='2')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    root.mkdir(exist_ok=True)
    manifest=json.loads((feature_root/'manifest.json').read_text())
    episodes,items=manifest['episodes'],manifest['items']
    features=np.load(feature_root/'features.npy')
    if features.shape!=(len(items),4,1024) or not np.load(feature_root/'completed.npy').all():
        raise ValueError('Need the completed four-corner feature extraction')
    if not np.isfinite(features).all() or any(e.get('method')=='q10' for e in episodes):
        raise ValueError('Expected finite features without q10')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    splits=paper_split(episodes)
    by_episode={i:[] for i in range(len(episodes))}
    for item in items:by_episode[item['episode']].append(item['index'])
    cutoffs={}
    for e in episodes:
        key=(e['suite'],e['task_id'])
        cutoffs[key]=min(cutoffs.get(key,e['steps']),e['steps'])
    prefixes={e['episode']:[i for i in by_episode[e['episode']] if items[i]['ready_step'] is not None
        and items[i]['ready_step']<=cutoffs[e['suite'],e['task_id']]] for e in episodes}
    visible={e['episode']:[i for i in by_episode[e['episode']] if items[i]['ready_step'] is not None
        and items[i]['ready_step']<=e['steps']] for e in episodes}
    if any(not x for x in prefixes.values()):raise ValueError('A rollout has no available feature at common cutoff')
    labels=np.asarray([e['failure'] for e in episodes],dtype=int)
    train_ids=splits['train']
    train_rows=[i for ep in train_ids for i in by_episode[ep]]
    weights=compact_weights(episodes,by_episode,train_ids,device)
    grid=list(product(CHOICES,CHOICES,(1e-5,3e-5,1e-4,3e-4,1e-3),(1e-3,1e-2,1e-1)))
    plan=dict(feature_root=str(feature_root),source=json.loads((AUTHOR_ROOT/'SOURCE.json').read_text()),
        epochs=epochs,candidates=len(grid),learning_rates=[1e-5,3e-5,1e-4,3e-4,1e-3],
        regularizations=[1e-3,1e-2,1e-1],horizon=list(CHOICES),denoise=list(CHOICES),seed=11,
        architecture='Author IndepModel: input ->256 ReLU ->1 sigmoid',batch_size=512,
        optimizer='Adam, default beta/eps, no weight_decay; author bias-excluded explicit L2',
        score='Cumulative sigmoid, not running mean',normalization='Author default: no feature normalization',
        loss='Exact author mean per rollout, inverse smoothed class frequency; compact gradient equivalent',
        primary_alpha=.2,calibration='Official Tfunc/Mean functional CP; seen validation successes;30/70 split',
        task_split=('Prior train/validation/test membership frozen; add seen successful controls60/40 per task'
            if manifest.get('frozen_paper_split_reference') else
            'Fixed unseen task7/8/9 in each suite; seeded60/40 rollout split within seen task0..6'),
        splits={name:dict(episode_ids=ids,episodes=len(ids),failures=int(labels[ids].sum()),
            features=sum(len(by_episode[i]) for i in ids)) for name,ids in splits.items()},
        scope='SAFE method transferred to cached pi0.5/RTC Full-Fast data; not original pi0 benchmark reproduction')
    plan_path=root/'training_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Existing run has a different training plan')
    save(plan_path,plan)
    records=json.loads((root/'candidates.json').read_text()) if (root/'candidates.json').exists() else []
    best=torch.load(root/'best.pt',map_location='cpu',weights_only=True) if (root/'best.pt').exists() else None
    curve_path=root/'curves.jsonl'
    def endpoint(prob,mean=False):
        return np.asarray([prob[prefixes[e['episode']]].mean() if mean else
            prob[prefixes[e['episode']]].sum() for e in episodes])
    def progress(stage,**extra):
        elapsed=time.monotonic()-started
        row=dict(status='running',stage=stage,completed_candidates=len(records),total_candidates=len(grid),
            elapsed_seconds=elapsed,**extra)
        save(root/'progress.json',row)
        text=('# SAFE官方方法：训练与测试\n\n'
            f"{len(records)}/{len(grid)}组完成；当前{extra.get('model',len(records))}组，{extra.get('epoch',0)}/{epochs}轮。"
            f" 已用{elapsed/60:.1f}分钟。\n\n"
            f"训练{plan['splits']['train']['episodes']}回合/{len(train_rows)}条特征；验证{plan['splits']['validation']['episodes']}回合；"
            f"未见任务测试{plan['splits']['test']['episodes']}回合。测试集不参与选模。\n")
        (root/'progress.md').write_text(text)
        (feature_root/'progress.md').write_text(text+'\n[训练进度](paper_training/progress.md)\n')
        print('SAFE_TRAIN_PROGRESS='+json.dumps(row,ensure_ascii=False),flush=True)
    current_geometry=None
    progress('starting')
    try:
        with curve_path.open('a') as curves:
            for number,(horizon,denoise,lr,reg) in enumerate(grid,1):
                if number<=len(records):continue
                geometry=(horizon,denoise)
                if geometry!=current_geometry:
                    data=aggregate_corners(features,horizon,denoise)
                    x=torch.as_tensor(data,device=device)
                    x_train=x[train_rows].contiguous()
                    current_geometry=geometry
                torch.manual_seed(11)
                model=make_model(x.shape[1],device)
                optimizer=torch.optim.Adam(model.parameters(),lr=lr)
                for epoch in range(1,epochs+1):
                    model.train();optimizer.zero_grad(set_to_none=True)
                    loss,monitor=compact_loss(model,x_train,weights,reg)
                    loss.backward();optimizer.step()
                    if epoch%100==0 or epoch==epochs:
                        value=float(loss.detach())
                        if not np.isfinite(value):raise RuntimeError('Nonfinite training loss')
                        curves.write(json.dumps(dict(model=number,epoch=epoch,total_loss=value,
                            monitor_loss=float(monitor.detach())))+'\n');curves.flush()
                        progress('training',model=number,epoch=epoch,horizon=horizon,denoise=denoise,
                            learning_rate=lr,regularization=reg,loss=value)
                model.eval()
                with torch.no_grad():prob=model.projector(x).flatten().cpu().numpy()
                scores=endpoint(prob)
                record=dict(model=number,horizon=horizon,denoise=denoise,learning_rate=lr,
                    regularization=reg,epochs=epochs,width=x.shape[1],loss=float(loss.detach()),
                    parameters=sum(p.numel() for p in model.parameters()),
                    train_auc=auc(labels[train_ids],scores[train_ids]),
                    validation_auc=auc(labels[splits['validation']],scores[splits['validation']]))
                if best is None or record['validation_auc']>best['record']['validation_auc']:
                    best=dict(record=record,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                    torch.save(best,root/'best.pt')
                records.append(record);save(root/'candidates.json',records)
                progress('candidate_completed',model=number,epoch=epochs,best_validation_auc=best['record']['validation_auc'])
        choice=best['record']
        model=make_model(choice['width'],device);model.load_state_dict(best['state_dict']);model.eval()
        x=torch.as_tensor(aggregate_corners(features,choice['horizon'],choice['denoise']),device=device)
        with torch.no_grad():prob=model.projector(x).flatten().cpu().numpy()
        scores=endpoint(prob);means=endpoint(prob,mean=True)
        counts=np.asarray([len(prefixes[i]) for i in range(len(episodes))],dtype=float)
        curves=[np.cumsum(prob[visible[i]]) for i in range(len(episodes))]
        successful_cal=[i for i in splits['validation'] if not episodes[i]['failure']]
        bands=conformal_bands(curves,successful_cal)
        cp={}
        for alpha,band in bands.items():
            alerts=np.asarray([bool(np.any(s[:len(prefixes[i])]>=band[:len(prefixes[i])])) for i,s in enumerate(curves)])
            full_alerts=np.asarray([bool(np.any(s>=band[:len(s)])) for s in curves])
            cp[alpha]=dict(common_cutoff={name:metrics(labels[ids],alerts[ids],.5) for name,ids in splits.items()},
                full_available_rollout={name:metrics(labels[ids],full_alerts[ids],.5) for name,ids in splits.items()})
        primary_band=bands['0.2']
        primary_alert=np.asarray([bool(np.any(s[:len(prefixes[i])]>=primary_band[:len(prefixes[i])])) for i,s in enumerate(curves)])
        split_metrics={name:dict(cp['0.2']['common_cutoff'][name],auc=auc(labels[ids],scores[ids]),
            mean_score_auc=auc(labels[ids],means[ids]),query_count_baseline_auc=auc(labels[ids],counts[ids])) for name,ids in splits.items()}
        old_test=[i for i in splits['test'] if episodes[i].get('source_group')=='old200' and episodes[i]['task_id']>=8]
        old_metrics=dict(metrics(labels[old_test],primary_alert[old_test],.5),auc=auc(labels[old_test],scores[old_test]),
            mean_score_auc=auc(labels[old_test],means[old_test]),query_count_baseline_auc=auc(labels[old_test],counts[old_test]))
        roles={i:name for name,ids in splits.items() for i in ids}
        predictions=[]
        for i,e in enumerate(episodes):
            crossed=np.flatnonzero(curves[i][:len(prefixes[i])]>=primary_band[:len(prefixes[i])])
            predictions.append(dict(e,pilot_split=e['split'],split=roles[i],score=float(scores[i]),
                mean_score=float(means[i]),available_predictions=int(counts[i]),
                common_cutoff_step=cutoffs[e['suite'],e['task_id']],predicted_failure=bool(primary_alert[i]),
                first_detection_step=items[visible[i][int(crossed[0])]]['ready_step'] if len(crossed) else None))
        per_suite={}
        for suite in sorted({e['suite'] for e in episodes}):
            ids=[i for i in splits['test'] if episodes[i]['suite']==suite]
            per_suite[suite]=dict(metrics(labels[ids],primary_alert[ids],.5),auc=auc(labels[ids],scores[ids]))
        best.update(alpha=.2,cp_band=torch.tensor(primary_band),source_features=str(feature_root),
            normalization='none',architecture='official SAFE IndepModel',training_plan=str(plan_path))
        torch.save(best,root/'failure_probe.pt')
        summary=dict(status='completed',elapsed_seconds=time.monotonic()-started,device=device,
            selected=choice,metrics=split_metrics,old40_test=old_metrics,per_suite_test=per_suite,
            functional_cp=cp,primary_alpha=.2,calibration_successes=len(successful_cal),
            splits=plan['splits'],training_features=len(train_rows),candidates=len(records),
            source=plan['source'],scope=plan['scope'],
            notes=['Official network, per-rollout mean loss with class-count smoothing and bias-excluded L2.',
                'Nine endpoint aggregations x five learning rates x three regularizations; final epoch1000; validation-only selection.',
                'Unseen task7/8/9 fixed before training; original old40 held-out tasks8/9 retained as a subset.',
                'Adaptive RTC request spacing differs from paper fixed chunk execution; query-count baseline is reported.',
                'Functional CP uses official Tfunc/Mean implementation and30/70 successful validation split; no nominal coverage claim across unseen tasks.',
                'Additional62 rollouts were failure-selected; this is enriched development data, not a natural-prevalence final evaluation.',
                'Offline detector evaluation only; no policy recovery rollout or robotics success improvement was tested.'])
        comparison=None
        if manifest.get('reference_training'):
            prior_root=Path(manifest['reference_training'])
            prior=json.loads((prior_root/'training_results.json').read_text())
            comparison=dict(reference=str(prior_root),
                intervention=manifest['intervention'],
                before=dict(test=prior['metrics']['test'],old40=prior['old40_test'],
                    per_suite_test=prior['per_suite_test'],calibration_successes=prior['calibration_successes']),
                after=dict(test=split_metrics['test'],old40=old_metrics,
                    per_suite_test=per_suite,calibration_successes=len(successful_cal)))
            summary['success_controls_comparison']=comparison
            save(root/'success_controls_comparison.json',comparison)
        save(root/'training_results.json',summary);save(root/'predictions.json',predictions)
        save(root/'cp_bands.json',{k:v.tolist() for k,v in bands.items()})
        lines=['# SAFE官方方法：训练与测试结果','',
            f"已完成{len(records)}组训练；每组{epochs}轮；耗时{summary['elapsed_seconds']/60:.1f}分钟。主模型与正式v4未改。",'',
            '|划分|回合|特征条数|AUC|识别失败|成功回合误报|','|---|---:|---:|---:|---:|---:|']
        for name,result in split_metrics.items():
            lines.append(f"|{name}|{result['episodes']}|{plan['splits'][name]['features']}|{result['auc']:.3f}|{result['tp']}/{result['failures']}|{result['fp']}/{result['episodes']-result['failures']}|")
        lines+=['',f"选中动作位置{choice['horizon']}、去噪步{choice['denoise']}；学习率{choice['learning_rate']}，L2={choice['regularization']}；仅根据已见任务验证AUC选择。",
            f"主报警阈值为官方默认alpha=0.2的随时间CP带，使用{len(successful_cal)}个成功验证回合校准。",'',
            f"原同一40回合测试：AUC={old_metrics['auc']:.3f}；失败识别{old_metrics['tp']}/{old_metrics['failures']}；成功误报{old_metrics['fp']}/{old_metrics['episodes']-old_metrics['failures']}。",
            f"该40回合的平均分辅助AUC={old_metrics['mean_score_auc']:.3f}；推理次数基线AUC={old_metrics['query_count_baseline_auc']:.3f}。",'',
            '训练、验证来自同一批已见任务的不同回合，测试任务完全留出。所有测试统计在选模完成后计算。',
            '这是SAFE方法在pi0.5/RTC自适应刷新记录上的移植；数据、主模型和采样节奏不同于论文实验，检测成绩不等于机器人成功率或恢复效果。','',
            '[完整指标](training_results.json) · [逐回合预测](predictions.json) · [全部训练候选](candidates.json) · [固定训练计划](training_plan.json)']
        if comparison:
            lines+=['','## 补成功样本前后，同一固定测试集','',
                '|固定测试集|版本|AUC|识别失败|成功误报|','|---|---|---:|---:|---:|']
            for cohort,label in (('test','79回合'),('old40','原40回合')):
                for key,version in (('before','补充前'),('after','补充后')):
                    value=comparison[key][cohort]
                    lines.append(f"|{label}|{version}|{value['auc']:.3f}|{value['tp']}/{value['failures']}|{value['fp']}/{value['episodes']-value['failures']}|")
            lines+=['','原训练、验证、测试回合成员均保持，新增成功仅进入已见任务训练/验证；网络、网格与主alpha不变。',
                '[补成功对照完整数据](success_controls_comparison.json)']
        (root/'training_results.md').write_text('\n'.join(lines)+'\n')
        save(root/'progress.json',dict(status='completed',completed_candidates=len(records),total_candidates=len(grid),
            elapsed_seconds=summary['elapsed_seconds'],selected=choice))
        text='# SAFE官方方法：完成\n\n[训练与测试结果](training_results.md)\n'
        (root/'progress.md').write_text(text)
        (feature_root/'progress.md').write_text('# 首末特征采集、SAFE训练测试已完成\n\n[训练与测试结果](paper_training/training_results.md)\n')
        print('SAFE_PAPER_RESULT='+json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException as exc:
        save(root/'progress.json',dict(status='invalid',error=type(exc).__name__+': '+str(exc),
            completed_candidates=len(records),total_candidates=len(grid)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--epochs',type=int,default=1000)
    args=parser.parse_args()
    root=args.output or args.features/'paper_training'
    with (args.features.parent.parent/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train(args.features,root,args.epochs)


if __name__=='__main__':
    main()
