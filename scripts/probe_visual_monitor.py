"""Render sparse old Continue observations and evaluate a local visual monitor."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from perceptbudget.visual_monitor import detector_name


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+')
    p.add_argument('--reuse-images', action='store_true')
    p.add_argument('--image-cache',type=Path,help='Read existing sparse images without copying or re-rendering')
    p.add_argument('--detector',type=detector_name,choices=('fixed_reference','updated_reference'),default='fixed_reference')
    p.add_argument('--grasp-check',action='store_true')
    p.add_argument('--render-missing',action='store_true',help='Render only new state-triggered sample times from saved physics')
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
            records=[]; decision=None; opened=False;pending_controls=[]
            sampler=WristSlipMonitor(detector=args.detector,grasp_check=args.grasp_check)
            for line in (Path(case['source_run'])/'rtc/trace.jsonl').open():
                row=json.loads(line)
                if row.get('event')=='m3_decision': decision=row
                if row.get('event')!='control_step': continue
                at=row['step']+1
                sampler.on_control(row['action'],row['after']['raw_state'],step=at)
                pending_controls.append(dict(action=row['action'],state=row['after']['raw_state'],step=at))
                opened = opened or row['action'][6] <= -.2
                flags=phase_flags(row,decision)
                periodic=at%5==0
                forced=flags['eligible'] and row['action'][6]>=.2 and sampler.grasp_check_due()
                if not periodic and not forced: continue
                records.append(dict(step=at, state=row['after']['raw_state'],
                    action=row['action'], phase=flags,controls=pending_controls,
                    sampling_trigger='periodic' if periodic else 'grasp_check',
                    opened_since_previous_sample=opened,
                    physics=row['after']['sim_state'],
                    observations=row['after']['observations']))
                if flags['eligible'] and row['action'][6]>=.2:sampler.note_grasp_sample()
                pending_controls=[]
                opened=False
            image_by_step={}
            if image_path.exists():
                with np.load(image_path) as data:
                    steps=data['frame_steps'] if 'frame_steps' in data else np.arange(1,len(data['images'])+1)*5
                    image_by_step.update((int(t),im) for t,im in zip(steps,data['images']))
            supplement=folder/'supplemental_observations.npz'
            if supplement.exists():
                with np.load(supplement) as data:
                    image_by_step.update((int(t),im) for t,im in zip(data['frame_steps'],data['images']))
            missing=[r for r in records if r['step'] not in image_by_step]
            if missing:
                if not args.render_missing and (args.reuse_images or args.image_cache):
                    raise RuntimeError('Missing saved sparse images')
                key=(case['suite'],case['task_id'])
                if key!=last_key:
                    if env is not None: env.close()
                    env=base._make_env(env_class,lookup[key]);env.reset();last_key=key
                rendered={}; sim=env.env.sim
                for r in missing:
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
                    rendered[r['step']]=obs['robot0_eye_in_hand_image'].copy()
                image_by_step.update(rendered)
                if args.image_cache:
                    steps=sorted(set(image_by_step)-set(range(5,max(image_by_step)+1,5)))
                    np.savez_compressed(supplement,images=np.asarray([image_by_step[t] for t in steps],dtype=np.uint8),
                        frame_steps=np.asarray(steps))
                else:
                    steps=sorted(image_by_step)
                    np.savez_compressed(image_path,images=np.asarray([image_by_step[t] for t in steps],dtype=np.uint8),
                        frame_steps=np.asarray(steps))
            monitor=WristSlipMonitor(detector=args.detector,grasp_check=args.grasp_check);samples=[]
            for r in records:
                image=image_by_step[r['step']]
                for update in r['controls']:
                    monitor.on_control(update['action'],update['state'],step=update['step'])
                if r['opened_since_previous_sample']:
                    monitor.reset_grasp();monitor.previous=None
                if not r['phase']['eligible'] or r['action'][6] < .2:continue
                if r['sampling_trigger']=='grasp_check' and not monitor.grasp_check_due():continue
                result=monitor.observe(image,r['state'],r['action'],step=r['step'],
                    eligible=r['phase']['eligible'])
                result['phase']=r['phase'];samples.append(result)
                result['sampling_trigger']=r['sampling_trigger']
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
                source_run=case['source_run'],detector=args.detector,grasp_check=args.grasp_check,
                image_source=str(image_path),supplement=str(supplement) if supplement.exists() else None,
                extra_grasp_samples=sum(s['sampling_trigger']=='grasp_check' for s in samples))
            save(folder/'monitor.json',report);results.append(report)
            summary=dict(status='running',detector=args.detector,grasp_check=args.grasp_check,
                extra_grasp_samples=sum(r['extra_grasp_samples'] for r in results),completed=len(results),total=len(selected),
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
        scope='Closed-gripper Continue: periodic5-step images plus optional causal grasp-state samples; no online object/contact labels',
        note='Unmatched alarms need image review; physical cues do not establish refresh benefit')
    save(root/'summary.json',summary)
    save(root/'progress.json',summary)
    save(root/'results.json',results)
    return 0


if __name__=='__main__':raise SystemExit(main())
