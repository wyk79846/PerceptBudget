"""Train a small SAFE-style MLP on frozen saved expert features.

Tasks, rather than frames/seeds, are held out. Report count-normalized risk at
an equal control-step cutoff within each task, alongside the original running
sum and a query-count baseline. No closed-loop intervention is evaluated here.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.analyze_gripper_timing import save


def auc(labels, scores):
    import numpy as np
    y=np.asarray(labels,dtype=bool);x=np.asarray(scores,dtype=float)
    positives=int(y.sum());negatives=len(y)-positives
    if not positives or not negatives:return None
    order=np.argsort(x,kind='stable');ranks=np.empty(len(x),dtype=float)
    start=0
    while start<len(x):
        stop=start+1
        while stop<len(x) and x[order[stop]]==x[order[start]]:stop+=1
        ranks[order[start:stop]]=(start+1+stop)/2
        start=stop
    return float((ranks[y].sum()-positives*(positives+1)/2)/(positives*negatives))


def metrics(labels, scores, threshold):
    import numpy as np
    y=np.asarray(labels,dtype=bool);p=np.asarray(scores,dtype=float)>=threshold
    tp=int((y&p).sum());tn=int((~y&~p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum())
    recall=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    return dict(episodes=len(y),failures=int(y.sum()),auc=auc(y,scores),
        balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
        failure_recall=recall,false_alarm_rate=1-specificity if specificity is not None else None,
        tp=tp,tn=tn,fp=fp,fn=fn)


def choose_threshold(labels, scores):
    import numpy as np
    values=np.unique(scores)
    candidates=[float(np.nextafter(values[0],-np.inf)),
        *map(float,(values[:-1]+values[1:])/2),float(np.nextafter(values[-1],np.inf))]
    return max(candidates,key=lambda t:(metrics(labels,scores,t)['balanced_accuracy'],t))


def train(root):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    plan=json.loads((root/'manifest.json').read_text())
    episodes,items=plan['episodes'],plan['items']
    features=np.load(root/'features.npy')
    completed=np.load(root/'completed.npy')
    if not completed.all() or not np.isfinite(features).all():
        raise RuntimeError('Feature extraction is not complete')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    by_episode={e['episode']:[item['index'] for item in items if item['episode']==e['episode']] for e in episodes}
    cutoffs={(e['suite'],e['task_id']):min(p['steps'] for p in episodes
        if (p['suite'],p['task_id'])==(e['suite'],e['task_id'])) for e in episodes}
    prefixes={e['episode']:[i for i in by_episode[e['episode']]
        if items[i]['ready_step'] is not None and items[i]['ready_step']<=cutoffs[e['suite'],e['task_id']]]
        for e in episodes}
    if any(not indices for indices in prefixes.values()):
        raise RuntimeError('At least one episode has no available feature before common cutoff')
    splits={name:[e['episode'] for e in episodes if e['split']==name] for name in ('train','validation','test')}
    labels=np.asarray([e['failure'] for e in episodes],dtype=np.int64)
    train_indices=np.asarray([i for ep in splits['train'] for i in by_episode[ep]])
    class_counts=Counter(labels[splits['train']].tolist())
    y=torch.as_tensor(np.asarray([labels[items[i]['episode']] for i in train_indices]),device=device,dtype=torch.float32)
    weights=[]
    for ep in splits['train']:
        length=len(by_episode[ep])
        # Equivalent to summing L1 losses of cumulative sigmoid scores over
        # every prefix; labels are rollout outcomes, no failure timestamps.
        weights.extend((length-position)/class_counts[int(labels[ep])] for position in range(length))
    weights=torch.tensor(weights,device=device,dtype=torch.float32);weights/=weights.sum()
    results=[];best=None;curves=[]
    def endpoint(probabilities):
        means=np.asarray([probabilities[prefixes[e['episode']]].mean() for e in episodes])
        sums=np.asarray([probabilities[prefixes[e['episode']]].sum() for e in episodes])
        return means,sums
    def progress(**extra):
        record=dict(status='running',elapsed_seconds=time.monotonic()-started,**extra)
        save(root/'training_progress.json',record)
        (root/'progress.md').write_text('# SAFE：小MLP训练\n\n'
            f"{len(episodes)}回合、{len(items)}条特征已采齐；模型训练{record.get('model',0)}/4，epoch={record.get('epoch',0)}/1000。\n\n"
            '任务0–5训练、6–7验证、8–9测试；同一任务的所有seed放在同一组。\n')
        print('PROBE_TRAINING='+json.dumps(record,ensure_ascii=False),flush=True)
    for aggregation in (0,1):
        x=features[:,aggregation]
        mean=x[train_indices].mean(axis=0);std=np.maximum(x[train_indices].std(axis=0),1e-5)
        normalized=torch.as_tensor((x-mean)/std,device=device)
        indices=torch.tensor(train_indices,device=device)
        for learning_rate in (3e-4,1e-3):
            number=len(results)+1
            torch.manual_seed(11)
            model=torch.nn.Sequential(torch.nn.Linear(1024,256),torch.nn.ReLU(),torch.nn.Linear(256,1)).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-3)
            local_best=None
            for epoch in range(1,1001):
                model.train();optimizer.zero_grad(set_to_none=True)
                p=model(normalized[indices]).flatten().sigmoid()
                loss=((y*(1-p)+(1-y)*p)*weights).sum()
                loss.backward();optimizer.step()
                if epoch%25==0 or epoch==1:
                    model.eval()
                    with torch.no_grad():prob=model(normalized).flatten().sigmoid().cpu().numpy()
                    means,sums=endpoint(prob)
                    val=splits['validation'];quality=auc(labels[val],means[val])
                    curves.append(dict(model=number,aggregation=aggregation,lr=learning_rate,epoch=epoch,
                        loss=float(loss.detach()),train_auc=auc(labels[splits['train']],means[splits['train']]),validation_auc=quality))
                    if local_best is None or quality>local_best['validation_auc']:
                        local_best=dict(validation_auc=quality,epoch=epoch,
                            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                if epoch%100==0:progress(model=number,aggregation=aggregation,lr=learning_rate,epoch=epoch,best_validation_auc=local_best['validation_auc'])
            model.load_state_dict(local_best['state']);model.eval()
            with torch.no_grad():prob=model(normalized).flatten().sigmoid().cpu().numpy()
            means,sums=endpoint(prob);val=splits['validation']
            threshold=choose_threshold(labels[val],means[val])
            record=dict(model=number,aggregation=plan['feature_names'][aggregation],aggregation_index=aggregation,
                learning_rate=learning_rate,best_epoch=local_best['epoch'],threshold=threshold,
                metrics={name:metrics(labels[ids],means[ids],threshold) for name,ids in splits.items()},
                cumulative_auc={name:auc(labels[ids],sums[ids]) for name,ids in splits.items()})
            results.append(record)
            if best is None or record['metrics']['validation']['auc']>best['record']['metrics']['validation']['auc']:
                best=dict(record=record,state=local_best['state'],mean=mean,std=std,
                    means=means.copy(),sums=sums.copy())
            save(root/'training_candidates.json',results)
    counts=np.asarray([len(prefixes[e['episode']]) for e in episodes],dtype=float)
    prediction_rows=[dict(e,common_cutoff_step=cutoffs[e['suite'],e['task_id']],
        available_predictions=int(counts[e['episode']]),score=float(best['means'][e['episode']]),
        cumulative_score=float(best['sums'][e['episode']]),
        predicted_failure=bool(float(best['means'][e['episode']])>=best['record']['threshold'])) for e in episodes]
    torch.save(dict(state_dict=best['state'],mean=torch.from_numpy(best['mean']),std=torch.from_numpy(best['std']),
        aggregation_index=best['record']['aggregation_index'],threshold=best['record']['threshold'],
        architecture='1024 -> 256 ReLU -> 1 sigmoid',source_manifest=str(root/'manifest.json')),
        root/'failure_probe.pt')
    summary=dict(status='completed',elapsed_seconds=time.monotonic()-started,device=device,
        parameters=sum(p.numel() for p in model.parameters()),selected=best['record'],candidates=results,
        query_count_baseline_auc={name:auc(labels[ids],counts[ids]) for name,ids in splits.items()},
        splits={name:dict(episodes=len(ids),failures=int(labels[ids].sum())) for name,ids in splits.items()},
        evaluation='At each task common minimum episode length; exclude features not ready by cutoff. '
            'Primary risk is mean instantaneous SAFE-MLP output, alongside original cumulative score and query-count baseline.',
        source_scope=plan['source_scope'],
        scope='Offline development pilot only; detection metrics are not policy success or closed-loop recovery rates. '
            'Checkpoint, FlashRT and finalv4 were not changed')
    if plan.get('reuse_root'):
        old_ids={name:[i for i in ids if episodes[i]['source_group']=='old200'] for name,ids in splits.items()}
        reference=torch.load(Path(plan['reuse_root'])/'failure_probe.pt',map_location='cpu',weights_only=True)
        old_model=torch.nn.Sequential(torch.nn.Linear(1024,256),torch.nn.ReLU(),torch.nn.Linear(256,1)).to(device)
        old_model.load_state_dict(reference['state_dict']);old_model.eval()
        old_x=features[:,reference['aggregation_index']]
        normalized_old=torch.as_tensor((old_x-reference['mean'].numpy())/reference['std'].numpy(),device=device)
        with torch.no_grad():old_prob=old_model(normalized_old).flatten().sigmoid().cpu().numpy()
        old_means,_=endpoint(old_prob)
        summary['same_old200_comparison']=dict(
            before={name:metrics(labels[ids],old_means[ids],reference['threshold']) for name,ids in old_ids.items()},
            after={name:metrics(labels[ids],best['means'][ids],best['record']['threshold']) for name,ids in old_ids.items()},
            note='Identical original held-out tasks and observation cutoffs; no test-based model or threshold selection')
        summary['additional_failure_recall']={}
        for method in sorted({e['method'] for e in episodes if e['source_group']=='new_failed_only'}):
            summary['additional_failure_recall'][method]={}
            for name,ids in splits.items():
                subset=[i for i in ids if episodes[i]['source_group']=='new_failed_only' and episodes[i]['method']==method]
                summary['additional_failure_recall'][method][name]=dict(episodes=len(subset),
                    detected=int((best['means'][subset]>=best['record']['threshold']).sum()))
        summary['sampling_note']=('Additional rollouts were selected for failure. Mixed-cohort accuracy/recall '
            'does not estimate natural failure prevalence; q10 has no newly collected successful controls. '
            'Also compare both probes on the identical old200 held-out tasks.')
    save(root/'training_results.json',summary);save(root/'training_predictions.json',prediction_rows)
    save(root/'training_curves.json',curves)
    lines=[f"# SAFE小探针：{len(episodes)}回合离线结果",'','只训练小MLP，主模型冻结；完整特征来自保存输入的离线重放。','',
        '|划分|回合|失败|失败检测AUC|平衡准确率|失败召回|误报率|','|---|---:|---:|---:|---:|---:|---:|']
    for name,metrics_ in best['record']['metrics'].items():
        lines.append(f"|{name}|{metrics_['episodes']}|{metrics_['failures']}|{metrics_['auc']:.3f}|{metrics_['balanced_accuracy']:.3f}|{metrics_['failure_recall']:.3f}|{metrics_['false_alarm_rate']:.3f}|")
    lines+=['',f"网络参数{summary['parameters']:,}；训练耗时{summary['elapsed_seconds']:.1f}秒（{device}）。",
        f"选择{best['record']['aggregation']}，学习率{best['record']['learning_rate']}，epoch{best['record']['best_epoch']}；仅按验证集AUC选择。",'',
        '每套任务中0–5训练、6–7验证、8–9测试；同任务的所有seed在同一划分，未拆帧混入测试。',
        '每任务在相同控制步截止点比较，主要报告平均风险，另保存原累积分数与推理次数基线，防止仅靠运行长度判断超时。',
        '这是SAFE风格的工程小测，不是原论文数据/超参的完整复现。检测到失败不等于刷新一次就能恢复；本轮没有闭环改动。', '',
        '[逐回合预测](training_predictions.json) · [全部候选](training_candidates.json) · [训练曲线](training_curves.json) · [采集清单](manifest.json)']
    if 'same_old200_comparison' in summary:
        comparison=summary['same_old200_comparison']
        lines+=['','新增部分只采失败，合并表不代表自然成败比例。下表保留原200回合的同一测试集，对照旧模型与补充训练后的模型：','',
            '|原40回合测试集|AUC|识别失败|成功回合误报|','|---|---:|---:|---:|']
        for key,label in (('before','补采前'),('after','补采后')):
            result=comparison[key]['test']
            lines.append(f"|{label}|{result['auc']:.3f}|{result['tp']}/{result['failures']}|{result['fp']}/{result['episodes']-result['failures']}|")
        lines+=['','新增q10未采成功对照，其单独召回不等于已验证正常任务不会误报。']
    (root/'training_results.md').write_text('\n'.join(lines)+'\n')
    (root/'progress.md').write_text(f'# SAFE：完成\n\n{len(episodes)}回合、{len(items)}次推理特征已提取，MLP训练与离线验证完成。\n\n[训练结果](training_results.md)\n')
    save(root/'training_progress.json',dict(status='completed',elapsed_seconds=time.monotonic()-started,selected=best['record']))
    print('PROBE_TRAINING_RESULT='+json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--wait-for-features',action='store_true')
    args=parser.parse_args()
    while args.wait_for_features and not (args.run_dir/'extraction_summary.json').exists():
        progress_path=args.run_dir/'progress.json'
        if progress_path.exists() and json.loads(progress_path.read_text()).get('status')=='invalid':
            raise RuntimeError('Feature extraction stopped; inspect its progress.json')
        time.sleep(3)
    with (args.run_dir.parent.parent/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        train(args.run_dir)


if __name__=='__main__':
    main()
