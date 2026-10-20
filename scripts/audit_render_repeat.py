"""Camera-only replay of saved actions; never loads or modifies the policy."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import create_run,load_config
from scripts import eval_q as base


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-render',action='store_true')
    parser.add_argument('--gl-probe',action='store_true',help='Temporary readback/dither probe inside sensor render; restored before return')
    parser.add_argument('--trials',type=int,choices=(1,3),default=3)
    args=parser.parse_args()
    import numpy as np
    config=load_config(repo_root=ROOT)
    os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl')
    old=Path(config['output_root'])/'M1/20261019T122945.216798Z-openpi-rtc-actual-bf2821fc'
    new=old.parent/'20261019T125112.959616Z-openpi-rtc-actual-5384eecc'
    controls=[r for r in map(json.loads,(old/'rtc/trace.jsonl').read_text().splitlines()) if r['event']=='control_step']
    historical=[np.load(p/'rtc/q5_frames.npz')['frames'][20,:,256:].copy() for p in (old,new)]
    def diff(a,b):
        return dict(channels=int(np.count_nonzero(a!=b)),max_abs=int(np.abs(a.astype(int)-b.astype(int)).max()))
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run=create_run(config,'M1','render-repeat-audit',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        shutil.copy2(__file__,run/'audit_render_repeat.py')
        base._patch_egl_cleanup()
        task=base._load_tasks([2],['libero_spatial'])[0]
        _,cls=base._libero_api()
        result=dict(status='running',scope=f'camera-only action replay, {args.trials} independent environments; no model')
        result['trials']=[]
        try:
            for trial in range(args.trials):
                np.random.seed(7);random.seed(7)
                env=base._make_env(cls,task)
                env._pb_lock_fixtures=True
                try:
                    # Match native warmup reset, then episode reset, with the same fixture lock.
                    base._prepare_env_episode(env,task,np)
                    np.random.seed(7);random.seed(7)
                    obs,_=base._prepare_env_episode(env,task,np)
                    capture={}
                    sim=env.env.sim
                    original_render=sim.render
                    current_step=-1
                    def render_hook(*a,**kw):
                        frame=original_render(*a,**kw)
                        if current_step==19 and kw.get('camera_name')=='robot0_eye_in_hand' and not capture:
                            capture['sim_time']=float(sim.data.time)
                            capture['inputs']={}
                            for owner,names in ((sim.model,('body_pos','body_quat','cam_pos','cam_quat',
                                'geom_size','geom_rgba','mat_rgba','tex_rgb','light_pos','light_dir',
                                'light_diffuse','light_specular','mesh_vert')),
                                (sim.data,('cam_xpos','cam_xmat','geom_xpos','geom_xmat','xpos','xmat','light_xpos','light_xdir'))):
                                for name in names:
                                    if hasattr(owner,name):
                                        capture['inputs'][name]=hashlib.sha256(np.asarray(getattr(owner,name)).tobytes()).hexdigest()
                            capture['first']=frame.copy()
                            if args.gl_probe:
                                from OpenGL import GL
                                ctx=sim._render_context_offscreen
                                readbacks=[ctx.read_pixels(256,256).copy() for _ in range(30)]
                                capture['same_frame_readback']=[diff(frame,x) for x in readbacks]
                                capture['offscreen_samples']=int(ctx.con.offSamples)
                                before=bool(GL.glIsEnabled(GL.GL_DITHER))
                                baseline=[original_render(*a,**kw).copy() for _ in range(200)]
                                capture['dither_before_pixels']=[x[61,187].tolist() for x in baseline]
                                capture['dither_before_repeats']=[diff(baseline[0],x) for x in baseline]
                                try:
                                    GL.glDisable(GL.GL_DITHER)
                                    off=[original_render(*a,**kw).copy() for _ in range(200)]
                                    capture['dither_disabled_state_after_render']=bool(GL.glIsEnabled(GL.GL_DITHER))
                                    capture['dither_off_pixels']=[x[61,187].tolist() for x in off]
                                    capture['dither_off_repeats']=[diff(off[0],x) for x in off]
                                finally:
                                    if before:GL.glEnable(GL.GL_DITHER)
                                    else:GL.glDisable(GL.GL_DITHER)
                            capture['repeated']=[original_render(*a,**kw).copy() for _ in range(200 if args.gl_probe else 20)]
                        return frame
                    if args.capture_render:
                        sim.render=render_hook
                    for step in range(20):
                        current_step=step
                        obs,_,_,_=env.step(np.asarray(controls[step]['action'],dtype=np.float32))
                    sim.render=original_render
                    from scripts.probe_delay_buffer import snapshot_scene
                    scene=snapshot_scene(env,obs)
                    saved_scene=controls[20]['before']
                    reference=obs['robot0_eye_in_hand_image'].copy()
                    sim=env.env.sim
                    # Determine robosuite's image convention directly against its observation.
                    raw=sim.render(width=256,height=256,camera_name='robot0_eye_in_hand')
                    flip=diff(raw[::-1],reference)['channels']<diff(raw,reference)['channels']
                    images=[]
                    for _ in range(30):
                        frame=sim.render(width=256,height=256,camera_name='robot0_eye_in_hand')
                        images.append((frame[::-1] if flip else frame).copy())
                    from OpenGL import GL
                    def glstring(name):
                        value=GL.glGetString(name)
                        return value.decode() if value else None
                    row=dict(trial=trial,scene_exact=scene==saved_scene,
                        scene_fields_equal={k:scene[k]==saved_scene[k] for k in scene},
                        historical_diff=[diff(reference,h) for h in historical],
                        observation_vs_poststep=[diff(reference,x) for x in images],
                        frozen_poststep_repeats=[diff(images[0],x) for x in images],
                        pixel_values=[x[61,187].tolist() for x in [reference,*images]],flip=flip,
                        gl=dict(vendor=glstring(GL.GL_VENDOR),renderer=glstring(GL.GL_RENDERER),
                                version=glstring(GL.GL_VERSION),dither=bool(GL.glIsEnabled(GL.GL_DITHER))))
                    if capture:
                        first=capture.pop('first');repeated=capture.pop('repeated')
                        # Capture is at exactly the original sensor sampling point, before any sim advance.
                        capture['repeated_diff']=[diff(first,x) for x in repeated]
                        capture['first_vs_observation']=diff(first,reference)
                        capture['pixel_values']=[x[61,187].tolist() for x in [first,*repeated]]
                        row['sensor_render_capture']=capture
                        np.savez_compressed(run/f'sensor{trial}.npz',first=first,repeated=np.stack(repeated))
                    np.savez_compressed(run/f'trial{trial}.npz',observation=reference,renders=np.stack(images))
                    result['trials'].append(row)
                    print('trial='+json.dumps(dict(trial=trial,scene_exact=row['scene_exact'],
                        historical_diff=row['historical_diff'],pixel=row['pixel_values'][0],
                        sensor_repeat_changes=sum(d['channels']>0 for d in row.get('sensor_render_capture',{}).get('repeated_diff',[])))),flush=True)
                finally:
                    env.close()
            result['status']='completed'
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(run/'summary.json',result)


if __name__=='__main__':main()
