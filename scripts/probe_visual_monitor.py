"""Render sparse old Continue observations and evaluate a local visual monitor."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+')
    p.add_argument('--reuse-images', action='store_true')
    p.add_argument('--image-cache',type=Path,help='Read existing sparse images without copying or re-rendering')
    p.add_argument('--detector',choices=('original_v1','updated_v2'),default='original_v1')
    args = p.parse_args()
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0')
    import numpy as np
    from scripts import eval_q as base
    from perceptbudget.visual_monitor import WristSlipMonitor
    from perceptbudget.stable_segments import phase_flags
    base._patch_egl_cleanup()
    tasks = base._load_tasks(list(range(40)), ['libero_spatial','libero_object','libero_goal','libero_10'])
    lookup = {(t['suite'], t['suite_task_id']): t for t in tasks}
    _, env_class = base._libero_api()
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    selected = [(int(f.parent.name), json.loads(f.read_text()))
                for f in sorted((args.dataset/'episodes').glob('*/case.json'))
                if args.cases is None or int(f.parent.name) in args.cases]
    results = []; begin = time.monotonic()
    env = None; last_key = None
    try:
        for case_id, case in selected:
            folder = root/f'{case_id:04d}'; folder.mkdir(exist_ok=True)
            image_path = (args.image_cache/f'{case_id:04d}' if args.image_cache else folder)/'observations.npz'
            records=[]; decision=None; opened=False
            for line in (Path(case['source_run'])/'rtc/trace.jsonl').open():
                row=json.loads(line)
                if row.get('event')=='m3_decision': decision=row
                if row.get('event')!='control_step': continue
                opened = opened or row['action'][6] <= -.2
                if (row['step']+1)%5: continue
                records.append(dict(step=row['step']+1, state=row['after']['raw_state'],
                    action=row['action'], phase=phase_flags(row,decision),
                    opened_since_previous_sample=opened,
                    physics=row['after']['sim_state'],
                    observations=row['after']['observations']))
                opened=False
            if image_path.exists():
                with np.load(image_path) as data: images=data['images']
            else:
                if args.reuse_images or args.image_cache: raise RuntimeError('Missing saved sparse images')
                key=(case['suite'],case['task_id'])
                if key!=last_key:
                    if env is not None: env.close()
                    env=base._make_env(env_class,lookup[key]);env.reset();last_key=key
                images=[]; sim=env.env.sim
                for r in records:
                    sim.set_state_from_flattened(np.asarray(r['physics'],dtype=np.float64))
                    for name,fixture in env.env.fixtures_dict.items():
                        pos=r['observations'].get(name+'_pos');quat=r['observations'].get(name+'_quat')
                        if pos is None or quat is None:continue
                        body=sim.model.body_name2id(fixture.root_body)
                        if sim.model.body_parentid[body]!=0:continue
                        sim.model.body_pos[body]=np.asarray(pos)
                        sim.model.body_quat[body]=np.asarray(quat)[[3,0,1,2]]
                    sim.forward()
                    # Exactly the sensor orientation used by the live policy.
                    obs=env.env._get_observations(force_update=True)
                    images.append(obs['robot0_eye_in_hand_image'])
                images=np.asarray(images,dtype=np.uint8)
                np.savez_compressed(image_path,images=images)
            monitor=WristSlipMonitor(detector=args.detector);samples=[]
            for r,image in zip(records,images):
                if r['opened_since_previous_sample']:
                    monitor.reset_grasp();monitor.previous=None
                if not r['phase']['eligible'] or r['action'][6] < .2:continue
                result=monitor.observe(image,r['state'],r['action'],step=r['step'],
                    eligible=r['phase']['eligible'])
                result['phase']=r['phase'];samples.append(result)
            events=case['physical_events']
            alarms=[r for r in samples if r['alarm']]
            # An independent local physical cue is only a descriptive label.
            # No final task outcome is used to choose thresholds or alarm time.
            matches=[]
            for a in alarms:
                match=[e for e in events if 0 <= a['step']-e['observation_step'] <= 15]
                matches.append(dict(step=a['step'],matched=match))
            report=dict(case_id=case_id,suite=case['suite'],task_id=case['task_id'],
                seed=case['seed'],split=case['split'],samples=samples,events=events,
                alarms=matches,visible_texture_locked=any(s['locked'] for s in samples),
                source_run=case['source_run'],detector=args.detector,image_source=str(image_path))
            save(folder/'monitor.json',report);results.append(report)
            summary=dict(status='running',detector=args.detector,completed=len(results),total=len(selected),
                samples=sum(len(r['samples']) for r in results),
                locked_episodes=sum(r['visible_texture_locked'] for r in results),
                alarms=sum(len(r['alarms']) for r in results),
                physically_matched=sum(bool(a['matched']) for r in results for a in r['alarms']),
                seconds=time.monotonic()-begin,model_loads=0)
            save(root/'progress.json',summary)
            (root/'progress.md').write_text(f"已完成{len(results)}/{len(selected)}条旧轨迹；实际Continue采样{summary['samples']}次，报警{summary['alarms']}次，其中{summary['physically_matched']}次有局部物理异常记录对应。\n")
            print('VISUAL_MONITOR_PROGRESS='+json.dumps(summary),flush=True)
    finally:
        if env is not None: env.close()
    samples=[s for r in results for s in r['samples']]
    summary.update(status='completed',mean_detector_ms=1000*sum(s['detector_seconds'] for s in samples)/len(samples),
        scope='Every5th step only during closed-gripper Continue; no inflight images or online object/contact labels',
        note='Unmatched alarms need image review; physical cues do not establish refresh benefit')
    save(root/'summary.json',summary)
    save(root/'progress.json',summary)
    save(root/'results.json',results)
    return 0


if __name__=='__main__':raise SystemExit(main())
