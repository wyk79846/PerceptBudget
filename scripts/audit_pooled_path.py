#!/usr/bin/env python3
"""Replay recorded pooling inputs through native construction and shared switching.

Two sequential guarded workers, one model load each, no environment or clocks.
The production autotune policy is retained; differences are reported, not hidden
by changing precision, noise, inputs, inference steps or matrix algorithms.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def selected_requests(requests):
    fast=[r for r in requests if r.get('context_refresh',{}).get('mode')=='pool2']
    if not fast:
        raise ValueError('Require recorded fresh pool2 alternation')
    indices=sorted(set((0,len(fast)//2,len(fast)-1)))
    return [fast[i] for i in indices]


def compare_arrays(left,right):
    import numpy as np
    if left.shape!=right.shape:
        return dict(shape_matches=False,left_shape=list(left.shape),right_shape=list(right.shape),exact=False)
    delta=np.abs(left.astype(np.float64)-right.astype(np.float64))
    return dict(shape_matches=True,exact=bool(np.array_equal(left,right)),
        max_abs=float(delta.max()),mean_abs=float(delta.mean()),
        different_elements=int(np.count_nonzero(delta)),elements=int(delta.size))


def worker(config,source,run,leg,*,fixed_gemm=False):
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    from scripts import eval_q as base
    from scripts.diagnose_pool_parity import pool_check
    from perceptbudget.experiment import _write_json_atomically
    from perceptbudget.h50_checkpoint import image_input,normalize_state
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import release_unused_native_expert,memory_record
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.rtc_visual_alternation import FreshViewRTCAdapter
    from perceptbudget.shared_visual_context import wrist_frontend

    cfg=dict(config,m2_profile=True,m2_view='full',rtc_audit_inputs=False,
             m2_visual_fast_mode='pool2',m2_visual_fast_enabled=True)
    if leg=='native':
        cfg['m2_vision_pool_factor']=2
    if fixed_gemm:
        cfg['native_gemm_policy']='heuristic-first'
    source_cfg=json.loads((source/'run.json').read_text())['config_snapshot']
    for key in ('H','N','checkpoint_provenance_revision','state_enabled','state_prompt_capacity','native_gemm_policy'):
        if key=='native_gemm_policy' and fixed_gemm:
            continue
        if cfg[key]!=source_cfg[key]:
            raise ValueError('Source mismatch: '+key)
    requests=selected_requests(json.loads((source/'rtc/q5_requests.json').read_text()))
    frames=np.load(source/'rtc/q5_frames.npz')['frames']
    stats=load_libero_norm_stats(cfg['norm_stats_path'])
    task=base._load_tasks([0],[source_cfg['rtc_suite']])[0]
    def snapshot(r):
        frame=frames[r['observation_step']]
        state=np.asarray(r['observation_raw_state'],dtype=np.float32)
        normalized=normalize_state(state,stats)
        if not np.array_equal(normalized,np.asarray(r['model_state'],dtype=np.float32)):
            raise RuntimeError('Recorded/current state normalization differs')
        return dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],
                    raw_state=state,state=normalized)
    torch,_=base._check_cuda_and_native(cfg)
    torch.use_deterministic_algorithms(True)
    result=dict(status='running',leg=leg,cases=[],memory=[],model_loads=1,
        source=str(source),gemm_policy=cfg['native_gemm_policy'],
        source_gemm_policy=source_cfg['native_gemm_policy'],diagnostic_fixed_gemm=fixed_gemm,
        scope='Recorded fixed-input numerical audit, not latency/energy or closed-loop evaluation')
    def mem(stage):
        r=memory_record(torch,stage)
        result['memory'].append(r)
        if r['available_mib']<900:
            raise RuntimeError('Pool path audit RAM below900MiB')
    try:
        initial=snapshot(requests[0])
        model=base._load_and_check_model(cfg,torch)
        from flash_rt.subgraphs.pi05.context_action import enable
        enable(model)
        base._infer_actions(model,initial,task['instruction'],50,{})
        if leg=='shared':
            from flash_rt.api import VLAModel
            fast=VLAModel(wrist_frontend(model._pipe,torch,fast_kind='pool2'),'torch')
            enable(fast)
            base._infer_actions(fast,initial,task['instruction'],50,{})
            release_unused_native_expert(fast,torch)
        release_unused_native_expert(model,torch)
        adapter=(FreshViewRTCAdapter if leg=='shared' else OriginalOpenPiRTCAdapter)(model,torch,cfg)
        if leg=='shared':
            adapter.attach_wrist(fast)
        gate=json.loads((Path(cfg['verified_reference_run'])/'summary.json').read_text())
        if not gate.get('passed') or adapter.expert.provenance!=gate['original_source_hashes']:
            raise RuntimeError('Frozen expert provenance gate mismatch')
        adapter.bootstrap(initial,task['instruction'],{})
        lengths=sorted({r['rtc_guidance']['valid_prefix_length'] for r in requests})
        if leg=='shared':
            lengths=sorted(set(lengths)|{n+384 for n in lengths})
        result['graph_audit']=adapter.prepare_step_bank(lengths)
        mem('prepared')
        def download(pl,name,count):
            data=np.empty(count,dtype=np.uint16)
            pl.bufs[name].download(data)
            return torch.from_numpy(data).view(torch.bfloat16).float().numpy()
        for index,r in enumerate(requests+requests[::-1]):
            inputs=snapshot(r)
            solver=r['rtc_solver_inputs']
            previous=np.asarray(solver['previous_raw'],dtype=np.float32)
            noise=torch.tensor(solver['noise'],device='cuda',dtype=torch.float32)
            def fixed_noise(*shape,**kw):
                if tuple(shape)!=(50,32):
                    raise RuntimeError('Unexpected random draw in replay')
                return noise.clone()
            if leg=='shared':
                # Perturb the other geometry before every pooled replay.
                cfg['m2_visual_fast_enabled']=False
                with patch.object(torch,'randn',side_effect=fixed_noise):
                    adapter.predict(inputs,task['instruction'],previous,delay=solver['delay'],
                                    executed=solver['executed'],request={})
                cfg['m2_visual_fast_enabled']=True
                adapter._guided_requests=0
            request={}
            with patch.object(torch,'randn',side_effect=fixed_noise):
                actions=adapter.predict(inputs,task['instruction'],previous,delay=solver['delay'],
                                        executed=solver['executed'],request=request)
            pl=adapter.pipeline
            valid=request['rtc_guidance']['valid_prefix_length']
            if valid!=r['rtc_guidance']['valid_prefix_length'] or pl.vision_seq_enc!=128:
                raise RuntimeError('Actual pooled prefix geometry mismatch')
            layout=pool_check(pl,torch,np,'both')
            if layout['mismatched_elements'] or request['rtc_guidance']['variable_length_fallback']:
                raise RuntimeError('Pooling layout/fallback audit failed')
            values=dict(pooled=download(pl,'vision_x_pooled',128*1152),
                keys=adapter.expert.keys.detach().cpu().float().numpy(),
                values=adapter.expert.values.detach().cpu().float().numpy(),
                encoder_rope=download(pl,'encoder_rope_weights',pl.encoder_seq_len*256)[:valid*256],
                decoder_rope=download(pl,'decoder_rope_weights',50*256),
                raw=np.asarray(request['actions_raw'],dtype=np.float32),actions=actions)
            if not all(np.isfinite(v).all() for v in values.values()):
                raise RuntimeError('Nonfinite audit arrays')
            np.savez_compressed(run/f'{leg}_{index}.npz',**values)
            result['cases'].append(dict(index=index,prediction_id=r['prediction_id'],
                observation_step=r['observation_step'],valid=valid,layout=layout,
                input_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in inputs['images']],
                raw_vs_recorded=compare_arrays(values['raw'],np.asarray(r['actions_raw'],dtype=np.float32))))
            mem('case_'+str(index))
            _write_json_atomically(run/f'{leg}.json',result)
        result['status']='completed'
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(run/f'{leg}.json',result)


def main():
    from perceptbudget.experiment import load_config,create_run,_write_json_atomically
    from perceptbudget.baseline import validate
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--worker',choices=('native','shared'))
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--fixed-gemm',action='store_true',help='Diagnostic only: first heuristic in BOTH legs; production config unchanged')
    args=parser.parse_args()
    cfg=load_config(repo_root=ROOT)
    validate(cfg)
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent!=(Path(cfg['output_root'])/'M2').resolve():
            parser.error('Require M2 worker directory')
        worker(cfg,args.source_run,args.run_dir,args.worker,fixed_gemm=args.fixed_gemm)
        return
    import numpy as np
    with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=create_run(cfg,'M2','pool-path-audit',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        summary=dict(status='running',source=str(args.source_run),comparisons=[],model_loads=0,
                     diagnostic_fixed_gemm=args.fixed_gemm)
        try:
            for leg in ('native','shared'):
                cmd=[cfg['python_executable'],str(ROOT/'scripts/guarded_run.py'),'--',
                     cfg['python_executable'],'-u',str(Path(__file__).resolve()),
                     '--source-run',str(args.source_run),'--worker',leg,'--run-dir',str(run)]
                if args.fixed_gemm:
                    cmd.append('--fixed-gemm')
                summary['model_loads']+=1
                with (run/f'{leg}.log').open('w') as stream:
                    completed=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f'{leg} worker exited{completed.returncode}; inspect its log')
            left=json.loads((run/'native.json').read_text())
            right=json.loads((run/'shared.json').read_text())
            if len(left['cases'])!=len(right['cases']) or left['status']!=right['status'] or left['status']!='completed':
                raise RuntimeError('Incomplete paired audit')
            for a,b in zip(left['cases'],right['cases']):
                if (a['prediction_id'],a['input_hashes'],a['valid'])!=(b['prediction_id'],b['input_hashes'],b['valid']):
                    raise RuntimeError('Audit inputs differ across legs')
                with np.load(run/f"native_{a['index']}.npz") as x,np.load(run/f"shared_{b['index']}.npz") as y:
                    fields={k:compare_arrays(x[k],y[k]) for k in x.files}
                summary['comparisons'].append(dict(index=a['index'],prediction_id=a['prediction_id'],fields=fields))
            summary.update(status='completed',all_exact=all(r['exact'] for c in summary['comparisons'] for r in c['fields'].values()),
                note='Nonzero differences require localization; autotune can select different numerical kernels across workers')
        except BaseException as exc:
            summary.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            _write_json_atomically(run/'summary.json',summary)
            meta=json.loads((run/'run.json').read_text());meta['status']=summary['status']
            _write_json_atomically(run/'run.json',meta)
            print('POOL_PATH_AUDIT_STATUS='+json.dumps(dict(status=summary['status'],all_exact=summary.get('all_exact'),error=summary.get('error'))),flush=True)


if __name__=='__main__':
    main()
