"""Restore saved physics states to render paired review images; never run VLA."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0')
    import numpy as np
    from PIL import Image,ImageDraw
    from scripts import eval_q as base
    from perceptbudget.libero_input import raw_libero_state
    base._patch_egl_cleanup()
    tasks=base._load_tasks(list(range(40)),['libero_spatial','libero_object','libero_goal','libero_10'])
    lookup={(t['suite'],t['suite_task_id']):t for t in tasks}
    _,env_class=base._libero_api()
    selected=json.loads((root/'review_selection.json').read_text())
    previous=root/'render_status.json'
    if previous.exists():
        with (root/'render_attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(json.loads(previous.read_text()),ensure_ascii=False)+'\n')
    grouped=defaultdict(list)
    for item in selected:grouped[item['suite'],item['task_id']].append(item)
    output=root/'review_images';output.mkdir(exist_ok=True)
    records=[];errors=[];tiles=[]
    for key,items in grouped.items():
        env=None
        try:
            env=base._make_env(env_class,lookup[key]);env.reset()
            for item in items:
                targets={max(0,item.get('history_start_step',item['observation_step']-4))+1:item.get('history_start_step',item['observation_step']-4),item['observation_step']:item['step']}
                snapshots={}
                for line in (Path(item['source_run'])/'rtc/trace.jsonl').open():
                    row=json.loads(line)
                    if row.get('event')=='control_step' and row['step'] in targets.values():
                        snapshots[row['step']+1]=row['after']
                pair=[];paths=[];diagnostics=[]
                for observation_step,snapshot in sorted(snapshots.items()):
                    saved=np.asarray(snapshot['sim_state'],dtype=np.float64)
                    sim=env.env.sim
                    sim.set_state_from_flattened(saved)
                    # Fixed fixture transforms are absent from MjSimState.
                    # Restore available world-root poses from this same trace.
                    fixtures=[]
                    numeric=snapshot.get('observations',{})
                    for name,fixture in env.env.fixtures_dict.items():
                        pos=numeric.get(name+'_pos');quat=numeric.get(name+'_quat')
                        if pos is None or quat is None:continue
                        body=sim.model.body_name2id(fixture.root_body)
                        if sim.model.body_parentid[body]!=0:continue
                        sim.model.body_pos[body]=np.asarray(pos)
                        q=np.asarray(quat);sim.model.body_quat[body]=q[[3,0,1,2]]
                        fixtures.append(name)
                    sim.forward()
                    obs=env.env._get_observations(force_update=True)
                    error=float(np.max(np.abs(raw_libero_state(obs)-np.asarray(snapshot['raw_state']))))
                    physics_error=float(np.max(np.abs(sim.get_state().flatten()-saved)))
                    if physics_error>1e-12:raise RuntimeError(f'Rendering changed saved physics state by{physics_error}')
                    # Cached proprioception in the source trace need not equal
                    # freshly recomputed sensors after mj_forward. Log that
                    # offset; equality of the saved physics is the render gate.
                    diagnostics.append(dict(observation_step=observation_step,
                        saved_physics_max_abs=physics_error,cached_vs_recomputed_state_max_abs=error,
                        restored_fixed_fixtures=fixtures))
                    frame=np.concatenate((obs['agentview_image'],obs['robot0_eye_in_hand_image']),axis=1)
                    path=output/f"{item['case_id']:04d}-{item['observation_step']:04d}-{item['review_kind']}-t{observation_step}.png"
                    Image.fromarray(frame).save(path);pair.append(frame);paths.append(str(path))
                if len(pair)!=2:raise RuntimeError('Need two distinct saved observation states for image review')
                tile=Image.new('RGB',(512,284),'white');tile.paste(Image.fromarray(np.concatenate(pair,axis=1)).resize((512,256)),(0,28))
                ImageDraw.Draw(tile).text((3,6),f"{key[0]} {key[1]} case{item['case_id']} t{item['observation_step']} {item['review_kind']}",fill='black')
                tiles.append(tile);records.append(dict(item,images=paths,state_restored=True,simulation_steps=0,
                    state_diagnostics=diagnostics))
        except Exception as exc:
            errors.append(dict(suite=key[0],task_id=key[1],error=type(exc).__name__+': '+str(exc)))
        finally:
            if env is not None:env.close()
    for page,start in enumerate(range(0,len(tiles),8),1):
        grid=Image.new('RGB',(1024,284*4),'white')
        for i,tile in enumerate(tiles[start:start+8]):grid.paste(tile,((i%2)*512,(i//2)*284))
        grid.save(output/f'review-page-{page:02d}.png')
    report=dict(status='completed' if not errors else 'partial',model_loaded=False,simulation_steps=0,
        image_source='Re-rendered exact saved simulator states; no old request image presented as new',
        completed=len(records),requested=len(selected),records=records,errors=errors)
    (root/'render_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('STABLE_REVIEW_RENDER='+json.dumps({k:v for k,v in report.items() if k not in ('records','errors')},ensure_ascii=False),flush=True)
    return 0 if not errors else 1


if __name__=='__main__':raise SystemExit(main())
