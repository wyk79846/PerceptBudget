#!/usr/bin/env python3
"""Same recorded history/handoff, one open-loop Full-vs-Fast intervention.

Not a real-time or closed-loop success/energy comparison. No model loaded.
"""
import argparse,fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--replay-run',type=Path,required=True)
    p.add_argument('--cache-from',type=Path,required=True)
    p.add_argument('--success-reference',type=Path,help='Validate per-goal evaluation on a recorded success')
    a=p.parse_args()
    os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl')
    import numpy as np
    from scripts import eval_q as base
    from perceptbudget.experiment import create_run,_write_json_atomically
    from perceptbudget.m4_acceptance import copy_precompiled_environment_cache,precompiled_environments
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.h50_checkpoint import decode_actions
    cfg=json.loads((a.source_run/'run.json').read_text())['config_snapshot']
    records=json.loads((a.source_run/'rtc/q5_requests.json').read_text())
    source_summary=json.loads((a.source_run/'summary.json').read_text())['episode']
    trace=[json.loads(x) for x in open(a.source_run/'rtc/trace.jsonl')]
    controls=[r for r in trace if r['event']=='control_step']
    replay=json.loads((a.replay_run/'summary.json').read_text())
    selected=next(r for r in records if r['observation_step']==replay['observation_step'])
    commit=next(c for c in source_summary['commits'] if c['prediction_id']==selected['prediction_id'])
    next_commit=next(c for c in source_summary['commits'] if c['prediction_id']==selected['prediction_id']+1)
    start=commit['commit_step'];stop=next_commit['commit_step'];origin=selected['observation_step']
    data=np.load(a.replay_run/'actions.npz');stats=load_libero_norm_stats(cfg['norm_stats_path'])
    full=decode_actions(data['Full'],stats)
    # Do not alter the gripper/control convention. Bind decoding to recording.
    original=decode_actions(np.asarray(selected['actions_raw'],np.float32),stats)
    if not np.array_equal(original,np.asarray(selected['actions'],np.float32)):raise ValueError('Decode mismatch')
    run=create_run(cfg,'M4','fast-physics-intervention',repo_root=ROOT);print('run_dir='+str(run),flush=True)
    result=dict(status='running',model_loaded=False,source=str(a.source_run),replay=str(a.replay_run),handoff_step=start,stop_step=stop,cases={},
                scope='Identical recorded prefix and handoff. Full computation latency held equal by construction; open-loop action-quality diagnostic only, no success/energy claim')
    env=None
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            task=base._load_tasks([cfg['screen']['task_ids'][0]],[cfg['screen']['suites'][0]])[0]
            tasks=[(task['suite'],task['task_id'])]
            copy_precompiled_environment_cache(a.cache_from,run,tasks,cfg)
            base._patch_egl_cleanup()
            with precompiled_environments(run,tasks,cfg):
                _,cls=base._libero_api();env=base._make_env(cls,task);env._pb_lock_fixtures=True
                for branch in ['recorded_Fast','counterfactual_Full']:
                    obs,_=base._prepare_env_episode(env,task,np)
                    max_prefix=0.;max_recorded=0.;rows=[]
                    for step in range(stop):
                        before=env.env.sim.get_state().flatten().copy()
                        expected=np.asarray(controls[step]['before']['sim_state'])
                        if step<start:
                            max_prefix=max(max_prefix,float(np.max(np.abs(before-expected))))
                            if max_prefix>1e-8:raise ValueError('Recorded prefix physics mismatch; no intervention attribution')
                        if branch=='recorded_Fast':
                            max_recorded=max(max_recorded,float(np.max(np.abs(before-expected))))
                            if max_recorded>1e-8:raise ValueError('Recorded action replay physics mismatch')
                        action=full[step-origin] if branch=='counterfactual_Full' and step>=start else np.asarray(controls[step]['action'],np.float32)
                        obs,_,_,_=env.step(action)
                        rows.append(dict(pudding=np.asarray(obs['chocolate_pudding_1_pos']).tolist(),gripper=np.asarray(obs['robot0_gripper_qpos']).tolist(),goals=[bool(env.env._eval_predicate(g)) for g in env.env.parsed_problem['goal_state']],step=step+1,cup=np.asarray(obs['porcelain_mug_1_pos']).tolist(),plate=np.asarray(obs['plate_1_pos']).tolist(),eef=np.asarray(obs['robot0_eef_pos']).tolist()))
                    result['cases'][branch]=dict(max_pudding_height_m=max(r['pudding'][2] for r in rows[start:]),any_pudding_goal=any(r['goals'][1] for r in rows[start:]),prefix_physics_max_abs=max_prefix,recorded_physics_max_abs=max_recorded if branch=='recorded_Fast' else None,positions=rows[start-1:])
                    print(branch+'='+json.dumps(result['cases'][branch]['positions'][-1]),flush=True)
                if a.success_reference:
                    reference=[json.loads(x) for x in open(a.success_reference/'rtc/trace.jsonl')]
                    successful=[r for r in reference if r['event']=='control_step' and r['success']]
                    if not successful:raise ValueError('No known successful step')
                    goals=env.env.parsed_problem['goal_state']
                    def evaluate(saved):
                        env.env.sim.set_state_from_flattened(np.asarray(saved))
                        env.env.sim.forward()
                        return [bool(env.env._eval_predicate(g)) for g in goals]
                    validated=evaluate(successful[0]['after']['sim_state'])
                    if not all(validated):raise ValueError('Saved success goal reconstruction failed')
                    snapshots={}
                    for step in [110,140,180,210,300,len(controls)]:
                        snapshots[str(step)]=evaluate(controls[step-1]['after']['sim_state'])
                    result['recorded_goals']=dict(predicates=goals,success_reference_validated=validated,source_snapshots=snapshots)
                env.close();env=None
            result['status']='completed'
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if env is not None:env.close()
        _write_json_atomically(run/'summary.json',result)
        print('PHYSICS_INTERVENTION_STATUS='+json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
