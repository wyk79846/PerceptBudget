#!/usr/bin/env python3
"""One-camera SigLIP recomputation with full VLM/state/RTC: fixed-input probe."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--observation-step', type=int, default=33)
    parser.add_argument('--clock-costs', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    import numpy as np
    from scripts import eval_q as base
    from scripts.diagnose_fast_fixed_input import recorded_frame_index
    from scripts.profile_selective_fast import TuneOnce, clock_costs
    from perceptbudget.experiment import create_run, _write_json_atomically as write
    from perceptbudget.h50_checkpoint import image_input, normalize_state
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.trace_frames import read_recorded_frame
    from perceptbudget.context_preparation import prepare_context_only
    from perceptbudget.camera_feature_context import CameraFeatureContext
    from perceptbudget.shared_visual_context import wrist_frontend
    from perceptbudget.rtc_memory import memory_record, release_unused_native_expert
    from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
    from perceptbudget.rtc_author_source import sample_author_source

    cfg = json.loads((args.source_run/'run.json').read_text())['config_snapshot']
    cfg.update(mapped_prompt_embedding=True, rtc_audit_inputs=False,
               rtc_audit_attention_same_input=False, rtc_audit_fused_norm=False)
    requests = json.loads((args.source_run/'rtc/q5_requests.json').read_text())
    matches = [r for r in requests if r['observation_step'] == args.observation_step]
    if len(matches) != 1 or matches[0]['rtc_solver_inputs']['previous_raw'] is None:
        raise ValueError('Require one recorded guided request')
    request = matches[0]
    summary = json.loads((args.source_run/'summary.json').read_text())
    stats = load_libero_norm_stats(cfg['norm_stats_path'])
    def snapshot(r):
        index = recorded_frame_index(summary, r['observation_step'])
        frame = read_recorded_frame(args.source_run/'rtc/q5_frames_spool.npy', index)
        state = normalize_state(r['observation_raw_state'], stats)
        if not np.array_equal(state, np.asarray(r['model_state'], np.float32)):
            raise ValueError('Recorded state preprocessing differs')
        result = dict(images=[image_input(frame[:, :256]), image_input(frame[:, 256:])], state=state)
        import hashlib
        if [hashlib.sha256(x.tobytes()).hexdigest() for x in result['images']] != r['context_refresh']['image_input_sha256']:
            raise ValueError('Recorded image preprocessing differs')
        return result
    fresh, old = snapshot(request), snapshot(requests[0])
    run = create_run(cfg, 'M4', 'camera-feature-fast', repo_root=ROOT)
    shutil.copy2(__file__, run/'probe.py')
    shutil.copy2(ROOT/'perceptbudget/camera_feature_context.py', run/'context_source.py')
    result = dict(status='running', model_loads=0, source_run=str(args.source_run),
                  observation_step=args.observation_step, cases={},
                  scope='Fixed input and real cost only; no closed-loop success claim')
    print('run_dir='+str(run), flush=True)
    def save(): write(run/'summary.json', result)
    def memory(stage):
        row = memory_record(torch, stage)
        result.setdefault('memory', []).append(row); save()
        if row['available_mib'] < 900:
            raise RuntimeError('RAM below900MiB; no reload or lower protection')
    adapter = None
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            torch, _ = base._check_cuda_and_native(cfg)
            torch.use_deterministic_algorithms(True)
            task = base._load_tasks([cfg['screen']['task_ids'][0]], [cfg['screen']['suites'][0]])[0]
            prompt = task['instruction']
            from flash_rt.api import VLAModel
            from flash_rt.subgraphs.pi05.context_action import enable
            model = base._load_and_check_model(cfg, torch); result['model_loads'] = 1
            model._pipe.gemm = TuneOnce(model._pipe.gemm)
            enable(model)
            result['full_preparation'] = prepare_context_only(model, fresh, prompt, torch)
            single = VLAModel(wrist_frontend(model._pipe, torch), 'torch')
            enable(single)
            result['single_preparation'] = prepare_context_only(single,
                dict(fresh, images=[fresh['images'][0]]), prompt, torch)
            context = CameraFeatureContext(model, single, torch)
            for m in (model, single): release_unused_native_expert(m, torch)
            memory('native_released')
            adapter = OriginalOpenPiRTCAdapter(model, torch, cfg)
            adapter.bootstrap(fresh, prompt, {})
            valid = context.full(fresh, prompt, step=args.observation_step)
            adapter.prepare_step_bank([valid])
            memory('source_graph_ready')
            solver = request['rtc_solver_inputs']
            noise = torch.tensor(solver['noise'], device='cuda', dtype=torch.float32)
            previous = torch.tensor(solver['previous_raw'], device='cuda', dtype=torch.float32)
            def raw_actions(valid):
                expert, graph = adapter.step_bank.select(valid)
                if graph is None: raise ValueError('Unprepared actual prefix')
                expert.refresh_context(context.pl.attn.enc_K, context.pl.attn.enc_V, valid)
                raw, _ = graph.replay(noise, previous, solver['delay'], solver['executed'])
                return raw.detach().cpu().numpy().copy()
            valid = context.full(fresh, prompt, step=args.observation_step)
            baseline = raw_actions(valid)
            result['full_vs_recorded_max_abs'] = float(np.max(np.abs(baseline-np.asarray(request['actions_raw']))))
            if result['full_vs_recorded_max_abs'] != 0:
                raise ValueError('Full output changed versus original recorded source')
            full_features = context.features()
            for camera in (0, 1):
                context.full(fresh, prompt, step=args.observation_step)
                valid, metadata = context.fast(fresh, prompt, step=args.observation_step, camera=camera)
                out = raw_actions(valid)
                diff = context.features()-full_features
                result['cases']['same_input_camera'+str(camera)] = dict(metadata,
                    feature_max_abs=float(diff.abs().max()),
                    cached_peer_feature_max_abs=float(diff[1-camera].abs().max()),
                    raw_max_abs=float(np.max(np.abs(out-baseline))))
                if result['cases']['same_input_camera'+str(camera)]['cached_peer_feature_max_abs'] != 0:
                    raise ValueError('Fast modified cached peer features')
            context.full(old, prompt, step=requests[0]['observation_step'])
            valid, metadata = context.fast(fresh, prompt, step=args.observation_step)
            fast = raw_actions(valid)
            start = solver['delay']; diff = fast[start:start+10, :7]-baseline[start:start+10, :7]
            result['cases']['cached_old_full'] = dict(metadata,
                next10_action7_rmse=float(np.sqrt(np.mean(diff**2))),
                next10_arm6_rmse=float(np.sqrt(np.mean(diff[:, :6]**2))),
                next10_gripper_max_abs=float(np.max(np.abs(diff[:, 6]))))
            expert, _ = adapter.step_bank.select(valid)
            author, _, _, sha = sample_author_source(expert, noise, previous,
                solver['delay'], solver['executed'], beta=cfg['rtc_beta'],
                source=Path(cfg['output_root'])/'rtc-author-reference/src/model.py')
            result['author_sampler_check'] = dict(max_abs=float(np.max(np.abs(fast-author.detach().cpu().numpy()))), sha256=sha)
            if result['author_sampler_check']['max_abs'] > 1e-4:
                raise ValueError('Fast RTC graph differs from author sampling loop')
            valid = context.full(fresh, prompt, step=args.observation_step)
            result['full_repeat_max_abs'] = float(np.max(np.abs(raw_actions(valid)-baseline)))
            if result['full_repeat_max_abs'] != 0:
                raise ValueError('Fast interfered with Full output')
            np.savez_compressed(run/'raw_actions.npz', full=baseline, fast=fast)
            save(); print('CAMERA_FEATURE_CHECK='+json.dumps(result['cases']), flush=True)
            if args.clock_costs:
                def infer(label, record):
                    if label == 'Full': valid = context.full(fresh, prompt, step=args.observation_step)
                    else: valid, _ = context.fast(fresh, prompt, step=args.observation_step)
                    return raw_actions(valid)
                result['clock_costs'] = clock_costs(infer, [None]*3, torch, run, write,
                    input_contract=dict(visual_mode='camera_feature_cache', visual_tokens=512,
                        state_enabled=True, encoded_views=1, cached_views=1, kv_recomputed=True,
                        position_policy='unchanged_full_geometry'))
            memory('finished')
            result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if adapter is not None:
            adapter.release_graph()
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
