"""Train the optional wrist CNN on saved RGB images and independent labels."""
import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def frame_label(row,truth,teacher):
    """Simulator-only supervision, never an input to the RGB classifier."""
    from perceptbudget.stable_segments import object_positions,distance
    state=row['after']['raw_state'];closed=row['action'][6]>=.2
    if closed and truth['normal'] and truth['normal_kind']=='consistent_carry':return 0
    recent=teacher.history[-3:]
    if closed and teacher.close and len(recent)==3:
        pos=object_positions(row['after']);origin=teacher.close['origin']
        following=any(p[2]-origin[name][2]>.005 and distance(p,state[:3])<.15
            for name,p in pos.items() if name in origin)
        if (state[2]-teacher.close['eef_z']>=.02
                and all(not any(r['contacts'].values()) for r in recent) and not following):return 1
    return None


def prepare(root):
    import numpy as np
    from PIL import Image,ImageDraw
    from perceptbudget.stable_segments import PhysicalTeacher
    dataset=PROJECT.parent/'任务产出/perceptbudget/M4/20261111T090955Z-stable-segments-dataset'
    cache=PROJECT.parent/'任务产出/perceptbudget/M4/20261111T113141Z-v4-visual-monitor'
    plan=json.loads((root/'plan.json').read_text());items=[];images=[]
    split_for={s:split for split,key in [('train','train_seeds'),('validation','validation_seeds'),('test','test_seeds')] for s in plan[key]}
    for file in sorted([*cache.glob('*/observations.npz'),*(cache/'validation').glob('*/observations.npz')]):
        case=json.loads((dataset/'episodes'/file.parent.name/'case.json').read_text())
        if case['seed'] not in split_for:continue
        with np.load(file,allow_pickle=False) as stored:
            data=stored['images'];steps=stored['frame_steps'] if 'frame_steps' in stored else np.arange(1,len(data)+1)*5
            frames=dict(zip(map(int,steps),data))
        teacher=PhysicalTeacher()
        for line in (Path(case['source_run'])/'rtc/trace.jsonl').open():
            row=json.loads(line)
            if row.get('event')!='control_step':continue
            truth=teacher.observe(row);at=row['step']+1
            if at not in frames:continue
            state=row['after']['raw_state'];label=frame_label(row,truth,teacher)
            if label is None:continue
            reason=('bilateral_contact_lifted_object_and_consistent_motion',
                    'closed_command_lift_no_object_contact_or_object_following')[label]
            image=np.asarray(Image.fromarray(frames[at]).resize((128,128),Image.Resampling.BILINEAR))
            index=len(items);images.append(image)
            items.append(dict(index=index,case_id=int(file.parent.name),seed=case['seed'],
                suite=case['suite'],task_id=case['task_id'],split=split_for[case['seed']],
                source_run=case['source_run'],step=at,label=label,label_name=('held','empty')[label],
                label_reason=reason,width_m=abs(state[6]-state[7])))
    labels=np.asarray([r['label'] for r in items],dtype=np.int64)
    np.savez_compressed(root/'dataset.npz',images=np.asarray(images,dtype=np.uint8),labels=labels)
    save(root/'manifest.json',dict(items=items,classes=['held','empty'],
        scope='RGB-only input; simulator contact/motion used only for frame labeling; no episode outcomes.',
        excluded_unresolved_frames=True,physical_labels_require_visual_review=True))
    groups={s:Counter(r['label_name'] for r in items if r['split']==s) for s in ('train','validation','test')}
    save(root/'dataset_summary.json',dict(frames=len(items),splits={s:dict(c) for s,c in groups.items()},
        episodes=len({r['case_id'] for r in items}),input_shape=[128,128,3],new_vla_runs=0))
    # Stratified montage for checking label semantics, not selecting model parameters.
    selected=[]
    for split in ('train','validation','test'):
        for label in (0,1):
            rows=[r for r in items if r['split']==split and r['label']==label]
            if rows:selected.extend(rows[int(i)] for i in np.linspace(0,len(rows)-1,min(8,len(rows))).astype(int))
    for page,start in enumerate(range(0,len(selected),24),1):
        canvas=Image.new('RGB',(768,4*152),'white');draw=ImageDraw.Draw(canvas)
        for offset,item in enumerate(selected[start:start+24]):
            x=(offset%6)*128;y=(offset//6)*152
            canvas.paste(Image.fromarray(images[item['index']]),(x,y+24))
            draw.text((x+2,y+2),f"{item['index']} {item['label_name']} {item['split'][:2]}",fill='black')
        canvas.save(root/f'label-review-{page:02d}.png')
    save(root/'label_review_selection.json',selected)
    print('OCCUPANCY_DATA='+json.dumps({s:dict(c) for s,c in groups.items()}),flush=True)
    return np.asarray(images),labels,items


def metrics(labels,probabilities,threshold=.5):
    import numpy as np
    predictions=probabilities[:,1]>=threshold
    matrix=[[int(np.sum((labels==truth)&(predictions==guess))) for guess in (0,1)] for truth in (0,1)]
    tn,fp=matrix[0];fn,tp=matrix[1]
    return dict(confusion_held_empty=matrix,accuracy=(tn+tp)/len(labels),
        empty_precision=tp/(tp+fp) if tp+fp else None,
        empty_recall=tp/(tp+fn) if tp+fn else None,
        false_empty_rate=fp/(tn+fp) if tn+fp else None,
        balanced_accuracy=.5*(tn/(tn+fp)+tp/(tp+fn)))


def train(root,images,labels,items):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    from perceptbudget.grip_occupancy import GripOccupancyCNN
    torch.set_num_threads(4);torch.manual_seed(11)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    x=torch.as_tensor(images.transpose(0,3,1,2).copy(),device=device,dtype=torch.float32)/255.
    y=torch.as_tensor(labels,device=device)
    indices={s:np.asarray([r['index'] for r in items if r['split']==s]) for s in ('train','validation','test')}
    assert len(set(r['case_id'] for r in items if r['split']=='train') & set(r['case_id'] for r in items if r['split']!='train'))==0
    for split,ids in indices.items():
        if len(set(labels[ids].tolist()))!=2:raise RuntimeError('Need both classes in '+split)
    model=GripOccupancyCNN().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    train_ids=indices['train'];counts=np.bincount(labels[train_ids],minlength=2)
    class_weights=torch.tensor(1./counts,device=device,dtype=torch.float32)
    class_weights/=class_weights.mean()
    loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights)
    best=None;curves=[];started=time.monotonic()
    for epoch in range(1,51):
        model.train();order=torch.as_tensor(train_ids[np.random.default_rng(11+epoch).permutation(len(train_ids))],device=device)
        losses=[]
        for batch in order.split(32):
            image=x[batch]
            gain=.85+.3*torch.rand((len(batch),1,1,1),device=device)
            image=(image*gain).clamp(0,1)
            optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(image),y[batch]);loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_ids=indices['validation'];logits=model(x[val_ids]);val_loss=float(torch.nn.functional.cross_entropy(logits,y[val_ids]));prob=logits.softmax(1).cpu().numpy()
        record=dict(epoch=epoch,train_loss=sum(losses)/len(losses),validation_loss=val_loss,
            validation=metrics(labels[val_ids],prob),seconds=time.monotonic()-started)
        curves.append(record)
        if best is None or val_loss<best['validation_loss']:
            best=dict(epoch=epoch,validation_loss=val_loss,state=copy.deepcopy(model.state_dict()))
        if epoch==1 or epoch%5==0:
            save(root/'training_progress.json',dict(status='training',**record));print('OCCUPANCY_TRAIN='+json.dumps(record),flush=True)
    model.load_state_dict(best['state']);model.eval()
    probabilities={}
    with torch.inference_mode():
        for split,ids in indices.items():probabilities[split]=torch.cat([model(batch).softmax(1) for batch in x[ids].split(64)]).cpu().numpy()
    # Calibrate abstention only on validation data, never on test outcomes.
    val_prob=probabilities['validation'];val_y=labels[indices['validation']]
    candidates=[]
    for confidence in np.linspace(.5,.99,50):
        accepted=val_prob.max(1)>=confidence;pred=val_prob.argmax(1)
        empty=accepted&(pred==1);held=accepted&(pred==0)
        if empty.sum()<2 or held.sum()<2:continue
        precision=float((val_y[empty]==1).mean());correct=float((pred[accepted]==val_y[accepted]).mean())
        if precision>=.95 and correct>=.95:candidates.append((float(accepted.mean()),float(confidence)))
    confidence=max(max(candidates,key=lambda v:(v[0],-v[1]))[1] if candidates else .99,
                   float(np.quantile(val_prob.max(1),.05)))
    report=dict(status='completed',device=device,parameters=sum(p.numel() for p in model.parameters()),
        epochs=50,best_epoch=best['epoch'],training_seconds=time.monotonic()-started,
        classes=['held','empty'],runtime_states=['held','empty','unknown'],
        confidence_threshold=confidence,validation_precision_target_met=bool(candidates),splits={})
    report['unknown_scope']='Confidence rejection, not a trained or validated occlusion-recognition class.'
    prediction_rows=[]
    for split,ids in indices.items():
        prob=probabilities[split];accepted=prob.max(1)>=confidence;pred=prob.argmax(1)
        report['splits'][split]=dict(frames=len(ids),episodes=len({items[i]['case_id'] for i in ids}),
            **metrics(labels[ids],prob),accepted=int(accepted.sum()),coverage=float(accepted.mean()),
            accepted_accuracy=float((pred[accepted]==labels[ids][accepted]).mean()) if accepted.any() else None,
            false_empty_accepted=int(np.sum(accepted&(pred==1)&(labels[ids]==0))),
            empty_true_accepted=int(np.sum(accepted&(pred==1)&(labels[ids]==1))),
            empty_total=int(np.sum(labels[ids]==1)))
        for index,p,ok in zip(ids,prob,accepted):
            prediction_rows.append(dict(items[int(index)],empty_probability=float(p[1]),
                predicted_state=('held','empty')[int(p.argmax())] if ok else 'unknown'))
    checkpoint=dict(state_dict={k:v.cpu() for k,v in best['state'].items()},confidence=confidence,
        classes=['held','empty'],input_size=128,architecture='GripOccupancyCNN',plan=json.loads((root/'plan.json').read_text()))
    torch.save(checkpoint,root/'model.pt')
    def measure(device_name):
        net=GripOccupancyCNN().to(device_name);net.load_state_dict(checkpoint['state_dict']);net.eval();sample=x[:1].to(device_name)
        times=[]
        with torch.inference_mode():
            for i in range(60):
                if device_name=='cuda':torch.cuda.synchronize()
                before=time.perf_counter();net(sample)
                if device_name=='cuda':torch.cuda.synchronize()
                if i>=10:times.append(1000*(time.perf_counter()-before))
        return dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),scope='batch1 model only; camera, resizing and VLA concurrency excluded')
    report['latency']={name:measure(name) for name in (('cpu','cuda') if device=='cuda' else ('cpu',))}
    save(root/'summary.json',report);save(root/'curves.json',curves);save(root/'predictions.json',prediction_rows)
    save(root/'training_progress.json',dict(status='completed',best_epoch=best['epoch'],epochs=50))
    print('OCCUPANCY_RESULT='+json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared',action='store_true',help='Train an already collected RGB dataset without collecting again')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    with (PROJECT.parent/'任务产出/perceptbudget/.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepared:
            import numpy as np
            with np.load(root/'dataset.npz',allow_pickle=False) as data:images,labels=data['images'],data['labels']
            items=json.loads((root/'manifest.json').read_text())['items']
        else:images,labels,items=prepare(root)
        train(root,images,labels,items)
    return 0


if __name__=='__main__':raise SystemExit(main())
