"""Expand wrist-CNN training data by rendering saved old physics, not VLA rollouts."""
import argparse
from collections import Counter,defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0')
    import numpy as np
    from PIL import Image,ImageDraw
    from perceptbudget.stable_segments import PhysicalTeacher
    from scripts.train_grip_occupancy import frame_label,save
    plan=json.loads((root/'plan.json').read_text());roles={seed:s for s,key in [('train','train_seeds'),('validation','validation_seeds'),('test','test_seeds')] for seed in plan[key]}
    source=PROJECT.parent/'任务产出/perceptbudget/M4/20261111T090955Z-stable-segments-dataset'
    candidates=[];started=time.monotonic()
    for file in sorted((source/'episodes').glob('*/case.json')):
        case=json.loads(file.read_text())
        if case['seed'] not in roles:continue
        teacher=PhysicalTeacher()
        for line in (Path(case['source_run'])/'rtc/trace.jsonl').open():
            row=json.loads(line)
            if row.get('event')!='control_step':continue
            truth=teacher.observe(row);at=row['step']+1
            if at%3:continue
            label=frame_label(row,truth,teacher)
            if label is None:continue
            snapshot=row['after'];obs=snapshot['observations']
            candidates.append(dict(case_id=int(file.parent.name),seed=case['seed'],suite=case['suite'],
                task_id=case['task_id'],split=roles[case['seed']],source_run=case['source_run'],step=at,
                label=label,label_name=('held','empty')[label],width_m=abs(snapshot['raw_state'][6]-snapshot['raw_state'][7]),
                physics=snapshot['sim_state'],poses={k:v for k,v in obs.items() if k.endswith(('_pos','_quat'))}))
    rng=np.random.default_rng(23);selected=[]
    def take(rows,count):
        groups=defaultdict(list)
        for row in rows:groups[row['case_id']].append(row)
        groups={k:[v[int(i)] for i in rng.permutation(len(v))] for k,v in groups.items()}
        output=[]
        while len(output)<count and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(output)<count:output.append(groups[key].pop())
        return output
    for split,budget in [('train',1000),('validation',300),('test',300)]:
        rows=[r for r in candidates if r['split']==split]
        empty=take([r for r in rows if r['label']==1],budget//2)
        held=take([r for r in rows if r['label']==0],budget-len(empty))
        selected.extend(empty+held)
    selected.sort(key=lambda r:(r['suite'],r['task_id'],r['case_id'],r['step']))
    save(root/'collection_progress.json',dict(status='rendering',completed=0,total=len(selected),new_vla_runs=0))
    print('OCCUPANCY_SELECTED='+json.dumps(dict(total=len(selected),train=sum(r['split']=='train' for r in selected))),flush=True)
    from scripts import eval_q as base
    from perceptbudget.visual_monitor import fresh_wrist_image
    import mujoco
    import robosuite.utils.binding_utils as binding
    base._patch_egl_cleanup();tasks=base._load_tasks(list(range(40)),['libero_spatial','libero_object','libero_goal','libero_10'])
    lookup={(t['suite'],t['suite_task_id']):t for t in tasks};_,env_class=base._libero_api()
    binary=Path(plan['environment_model_cache_source'])/'environment_models'
    def factory(cls,xml):
        key=hashlib.sha256((mujoco.__version__+'\0'+xml).encode()).hexdigest()
        return cls(mujoco.MjModel.from_binary_path(str(binary/(key+'.mjb'))))
    images=[];items=[];env=None;last=None
    with (PROJECT.parent/'任务产出/perceptbudget/.async_gpu.lock').open('a') as lock,patch.object(binding.MjSim,'from_xml_string',classmethod(factory)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for index,row in enumerate(selected):
                key=row['suite'],row['task_id']
                if key!=last:
                    if env is not None:env.close()
                    env=base._make_env(env_class,lookup[key]);env.reset();last=key
                    for name,sensor in env.env._observables.items():
                        if name.endswith('_image'):sensor.set_enabled(name=='robot0_eye_in_hand_image')
                sim=env.env.sim;sim.set_state_from_flattened(np.asarray(row['physics'],dtype=np.float64))
                for name,fixture in env.env.fixtures_dict.items():
                    pos=row['poses'].get(name+'_pos');quat=row['poses'].get(name+'_quat')
                    if pos is None or quat is None:continue
                    body=sim.model.body_name2id(fixture.root_body)
                    if sim.model.body_parentid[body]!=0:continue
                    sim.model.body_pos[body]=pos;sim.model.body_quat[body]=np.asarray(quat)[[3,0,1,2]]
                sim.forward();image,stamp=fresh_wrist_image(env)
                images.append(np.asarray(Image.fromarray(image).resize((128,128),Image.Resampling.BILINEAR)))
                item={k:v for k,v in row.items() if k not in ('physics','poses')};item['index']=index;items.append(item)
                if (index+1)%100==0:
                    status=dict(status='rendering',completed=index+1,total=len(selected),new_vla_runs=0,seconds=time.monotonic()-started)
                    save(root/'collection_progress.json',status);print('OCCUPANCY_COLLECT='+json.dumps(status),flush=True)
        finally:
            if env is not None:env.close()
    np.savez_compressed(root/'dataset.npz',images=np.asarray(images,dtype=np.uint8),labels=np.asarray([r['label'] for r in items]))
    save(root/'manifest.json',dict(items=items,classes=['held','empty'],new_vla_runs=0,
        source='Rendered saved old physics; known robot/contact/object states used only as labels, not network inputs.'))
    save(root/'dataset_summary.json',dict(frames=len(items),episodes=len({r['case_id'] for r in items}),
        splits={s:dict(Counter(r['label_name'] for r in items if r['split']==s)) for s in roles.values()},new_vla_runs=0))
    review=[]
    for split in ('train','validation','test'):
        for label in (0,1):
            rows=[r for r in items if r['split']==split and r['label']==label]
            review.extend(rows[int(i)] for i in np.linspace(0,len(rows)-1,8).astype(int))
    for page,start in enumerate(range(0,len(review),24),1):
        canvas=Image.new('RGB',(768,608),'white');draw=ImageDraw.Draw(canvas)
        for offset,row in enumerate(review[start:start+24]):
            x=offset%6*128;y=offset//6*152;canvas.paste(Image.fromarray(images[row['index']]),(x,y+24))
            draw.text((x+2,y+2),f"{row['index']} {row['label_name']} {row['split'][:2]}",fill='black')
        canvas.save(root/f'label-review-{page:02d}.png')
    save(root/'collection_progress.json',dict(status='completed',completed=len(items),total=len(items),seconds=time.monotonic()-started,new_vla_runs=0))
    return 0


if __name__=='__main__':raise SystemExit(main())
