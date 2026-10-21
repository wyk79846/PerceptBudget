#!/usr/bin/env python3
"""Isolate LIBERO environment residency without loading model weights."""
import fcntl
import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spatial10',action='store_true')
    parser.add_argument('--resident-source',type=Path,help='Hold one recorded model/expert geometry across resets; no rollout')
    args=parser.parse_args()
    import ctypes
    import numpy as np
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from perceptbudget.device_quality import install_render_buffer_cleanup, SUITES
    from perceptbudget.rtc_memory import memory_record
    from scripts import eval_q as base
    from scripts.complete_m1_h50 import calibrate
    config = load_config(repo_root=ROOT)
    if args.resident_source:
        saved=json.loads((args.resident_source/'run.json').read_text())['config_snapshot']
        for key in ('m2_profile','m2_view','m2_vision_pool_factor','m2_pool_view'):
            if key in saved:config[key]=saved[key]
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                      PYOPENGL_PLATFORM='egl', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    run = create_run(config, 'M2', 'environment-memory')
    print('run_dir='+str(run), flush=True)
    record = dict(status='running', rows=[], model_loads=0)
    def save():
        _write_json_atomically(run/'summary.json', record)
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch, _ = base._check_cuda_and_native(config)
        torch.use_deterministic_algorithms(True)
        base._patch_egl_cleanup()
        record['render_cleanup'] = install_render_buffer_cleanup(shared_egl=True)
        _, cls = base._libero_api()
        adapter=model=None
        def sample(stage, cycle):
            gc.collect()
            ctypes.CDLL(None).malloc_trim(0)
            row = memory_record(torch, stage)
            row['cycle'] = cycle
            record['rows'].append(row)
            save()
            print(json.dumps(row), flush=True)
            if row['available_mib'] < 900:
                raise RuntimeError('RAM below900MiB')
        env = None
        try:
            sample('start', -1)
            if args.resident_source:
                from perceptbudget.libero_input import load_libero_norm_stats
                stats=load_libero_norm_stats(config['norm_stats_path'])
                model=base._load_and_check_model(config,torch)
                from flash_rt.subgraphs.pi05.context_action import enable
                enable(model)
                record['model_loads']=1
                sample('model_loaded',-1)
            for cycle in range(10 if args.spatial10 else 9):
                task = base._load_tasks([cycle if args.spatial10 else 0],
                    ['libero_spatial' if args.spatial10 else SUITES[cycle % 3]])[0]
                base._seed_process(7, np, torch)
                env = base._make_env(cls, task)
                env._pb_lock_fixtures = True
                sim_model=env.env.sim.model._model
                print('render_geometry='+json.dumps(dict(task=cycle,offwidth=sim_model.vis.global_.offwidth,
                    offheight=sim_model.vis.global_.offheight,offsamples=sim_model.vis.quality.offsamples,
                    texture_bytes=sim_model.tex_data.nbytes,narena=sim_model.narena)),flush=True)
                del sim_model
                sample('constructed', cycle)
                observation, _ = calibrate(env, task, np)
                sample('calibrated', cycle)
                observation, _ = base._prepare_env_episode(env, task, np)
                sample('episode_reset', cycle)
                if model is not None:
                    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
                    from perceptbudget.rtc_memory import release_unused_native_expert
                    from perceptbudget.rtc_source_step_graph import h50_state_prefix_lengths
                    prepared=base._make_snapshot(observation,stats,True)
                    if adapter is None:
                        base._infer_actions(model,prepared,task['instruction'],50,{})
                        release_unused_native_expert(model,torch)
                        adapter=OriginalOpenPiRTCAdapter(model,torch,config)
                        request={};adapter.bootstrap(prepared,task['instruction'],request)
                        lengths=h50_state_prefix_lengths(request['rtc_guidance']['valid_prefix_length'],
                            task['instruction'],pool_factor=config.get('m2_vision_pool_factor',1),
                            pooled_view=config.get('m2_pool_view','both'))
                        root_expert=adapter.expert
                        record['first_graph_audit']=adapter.prepare_step_bank(lengths)
                        adapter.step_bank=None;adapter.expert=root_expert
                        torch.cuda.synchronize();torch.cuda.empty_cache()
                    else:
                        adapter.bootstrap(prepared,task['instruction'],{})
                    prepared=None
                    sample('resident_context_ready',cycle)
                env.close()
                env = observation = None
                sample('closed', cycle)
            record['status'] = 'completed'
        except BaseException as exc:
            record.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if env is not None:
                env.close()
            if adapter is not None:
                adapter.release_graph()
            save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
