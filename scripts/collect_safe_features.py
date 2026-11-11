"""Extract final expert features from saved inputs, one resident model.

Replay saved requests without simulator execution or frequency changes. An
existing extraction can be reused while adding only failed saved rollouts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.analyze_gripper_timing import INDEX, save
from perceptbudget.failure_features import FEATURE_LAYOUTS


def prepare_saved(source_manifest, layout):
    manifest=json.loads(source_manifest.read_text())
    root=source_manifest.parent.parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-safe-'+layout+'-no-q10')
    root.mkdir()
    manifest.update(dataset_label='safe_'+layout+'_no_q10',source_manifest=str(source_manifest),
        feature_layout=layout,feature_names=FEATURE_LAYOUTS[layout],reused_features=0,reuse_root=None,
        pilot_reference_root=manifest.get('reuse_root'),
        intervention=f"Re-extract first/last action positions and first/last denoising steps; same saved{len(manifest['episodes'])} rollouts, no q10")
    if 'collection_budget' in manifest:
        manifest['source_selection_budget']=manifest.pop('collection_budget')
    manifest.pop('feature_source',None)
    save(root/'manifest.json',manifest)
    pointer=dict(run=str(root),progress=str(root/'progress.md'))
    save('/tmp/perceptbudget-safe-'+layout+'-current.json',pointer)
    save('/tmp/perceptbudget-safe-current.json',pointer)
    return root


def prepare(reuse_root=None, failures_index=None):
    if (reuse_root is None)!=(failures_index is None):
        raise ValueError('Use --reuse-run and --append-failures together')
    label='old200' if reuse_root is None else 'old200-plus-failures'
    root = INDEX.parent.parent / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-safe-probe-'+label)
    root.mkdir()
    episodes, items = [], []
    if reuse_root is None:
        original = json.loads(INDEX.read_text())['rows']
    else:
        previous=json.loads((reuse_root/'manifest.json').read_text())
        episodes,items=previous['episodes'],previous['items']
        for episode in episodes:
            episode.update(source_group='old200',method='mixed',request_file='q5_requests.json',frame_file='q5_frames.npz')
        original=[row for row in json.loads(failures_index.read_text())['rows'] if row['outcome']!='success']
    reused=len(items)
    for row in original:
        number=len(episodes)
        run = Path(row['run'])
        request_path=next((run/'rtc').glob('q*_requests.json'))
        requests = json.loads(request_path.read_text())
        frame_file=request_path.name.replace('_requests.json','_frames.npz')
        split = 'train' if row['task_id'] < 6 else 'validation' if row['task_id'] < 8 else 'test'
        episodes.append(dict(episode=number, seed=row['seed'], suite=row['suite'], task_id=row['task_id'],
            outcome=row['outcome'], failure=row['outcome']!='success', steps=row['steps'],
            split=split, source_run=str(run),method=row.get('method','mixed'),
            source_group='old200' if reuse_root is None else 'new_failed_only',
            request_file=request_path.name,frame_file=frame_file))
        for index, request in enumerate(requests):
            items.append(dict(index=len(items), episode=number, request_index=index,
                prediction_id=request['prediction_id'], observation_step=request['observation_step'],
                ready_step=request.get('ready_observed_step'),
                mode=(request.get('context_refresh') or {}).get('mode','full'),
                valid=request['rtc_guidance']['valid_prefix_length']))
    manifest = dict(source_index=str(INDEX), episodes=episodes, items=items, total=len(items),
        dataset_label=label,reuse_root=str(reuse_root) if reuse_root else None,reused_features=reused,
        failures_index=str(failures_index) if failures_index else None,
        feature_names=['last_denoise_mean_actions','mean_denoise_mean_actions'], feature_width=1024,
        split='Within every suite: task0..5 train, task6..7 validation, task8..9 test; all seeds of a task stay together',
        source_scope='Only original old200 development data; new25seed remains untouched' if reuse_root is None else
            'User-authorized old200 plus failed rollouts from the 2000 campaign; failure-enriched development data',
        policy='Frozen H50/N5/BF16, saved Full/Fast state and original positions, saved RTC inputs')
    save(root/'manifest.json', manifest)
    pointer=dict(run=str(root),progress=str(root/'progress.md'))
    save('/tmp/perceptbudget-safe-'+label+'-current.json',pointer)
    save('/tmp/perceptbudget-safe-current.json',pointer)
    return root


def worker(root):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', MALLOC_ARENA_MAX='2', OMP_NUM_THREADS='4')
    import numpy as np
    from scripts import eval_q as base
    from perceptbudget.context_preparation import prepare_context_only
    from perceptbudget.h50_checkpoint import image_input
    from perceptbudget.rtc_memory import release_unused_native_expert
    from perceptbudget.openpi_reference import OriginalOpenPiExpert, sample_original_rtc
    from perceptbudget.shared_visual_context import wrist_frontend
    from perceptbudget.selective_pipeline import align_expert_positions
    from perceptbudget.rtc_exact_flash_probe import enable_exact_flash_probe
    from perceptbudget.rtc_fused_norm import enable_fused_norm
    from perceptbudget.rtc_condition_cache import enable_condition_cache
    from perceptbudget.rtc_source_step_graph import SourceStepBank
    from perceptbudget.failure_features import ExpertFeatureTap
    plan = json.loads((root/'manifest.json').read_text())
    episodes, items = plan['episodes'], plan['items']
    cfg = json.loads((Path(episodes[0]['source_run'])/'run.json').read_text())['config_snapshot']
    cfg.update(rtc_audit_inputs=False, rtc_audit_attention_same_input=False, rtc_audit_fused_norm=False,
        m4_validate_setup=False, m4_skip_eval_calibration=True)
    started = time.monotonic()
    features_path, done_path = root/'features.npy', root/'completed.npy'
    resume = features_path.exists()
    feature_shape=(len(plan['feature_names']),plan['feature_width'])
    features = np.lib.format.open_memmap(features_path, mode='r+' if resume else 'w+', dtype=np.float32,
        shape=(len(items), *feature_shape))
    done = np.lib.format.open_memmap(done_path, mode='r+' if resume else 'w+', dtype=np.bool_, shape=(len(items),))
    if not resume:
        features[:] = np.nan
        done[:] = False
        if plan.get('reuse_root'):
            previous=Path(plan['reuse_root'])
            count=plan['reused_features']
            features[:count]=np.load(previous/'features.npy',mmap_mode='r')
            done[:count]=np.load(previous/'completed.npy',mmap_mode='r')
            features.flush();done.flush()
    metadata_path = root/'extraction_rows.jsonl'
    status = dict(status='loading', completed=int(done.sum()), total=len(items), model_loads=0,
        reused_features=plan.get('reused_features',0),episodes=len(episodes),
        source_scope=plan['source_scope'], clock_settings_changed=False, checkpoint_frozen=True)
    def progress(stage, **extra):
        status.update(stage=stage, completed=int(done.sum()), elapsed_seconds=time.monotonic()-started, **extra)
        save(root/'progress.json', status)
        per = Counter(episodes[item['episode']]['split'] for item in items if done[item['index']])
        (root/'progress.md').write_text('# SAFE小探针：特征采集\n\n'
            f"状态{status['status']}；阶段{stage}；推理记录{status['completed']}/{len(items)}；模型加载{status['model_loads']}次。\n\n"
            f"复用已有{status['reused_features']}条；本次新增{status['completed']-status['reused_features']}/{len(items)-status['reused_features']}条。\n\n"
            f"已采集：训练{per['train']}、验证{per['validation']}、测试{per['test']}条；按任务划分。\n\n"
            '只重放保存输入，不跑仿真；正式v4保持不变。\n')
        print('FEATURE_PROGRESS='+json.dumps({k:status[k] for k in ('status','stage','completed','total','elapsed_seconds')},ensure_ascii=False),flush=True)
    progress('starting')
    bank = tap = None
    @lru_cache(maxsize=4)
    def source_data(number):
        episode=episodes[number]
        run = Path(episode['source_run'])
        requests = json.loads((run/'rtc'/episode.get('request_file','q5_requests.json')).read_text())
        with np.load(run/'rtc'/episode.get('frame_file','q5_frames.npz'), allow_pickle=False) as archive:
            frames, steps = archive['frames'], archive['frame_steps']
        return requests, frames, {int(step):i for i,step in enumerate(steps)}
    def inputs(item):
        requests, frames, positions = source_data(item['episode'])
        request = requests[item['request_index']]
        frame = frames[positions[request['observation_step']]]
        snapshot = dict(images=[image_input(frame[:,:256]),image_input(frame[:,256:])],
            state=np.asarray(request['model_state'],dtype=np.float32),
            raw_state=np.asarray(request['observation_raw_state'],dtype=np.float32))
        return request, snapshot
    try:
        torch, _ = base._check_cuda_and_native(cfg)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        tasks = base._load_tasks(list(range(40)), ['libero_spatial','libero_object','libero_goal','libero_10'])
        prompts = {(t['suite'],t['suite_task_id']):t['instruction'] for t in tasks}
        first = next(item for item in items if item['mode']=='full')
        request, snapshot = inputs(first)
        ep = episodes[first['episode']]
        prompt = prompts[ep['suite'],ep['task_id']]
        model = base._load_and_check_model(cfg,torch)
        status['model_loads']=1
        progress('full_weights_loaded')
        from flash_rt.subgraphs.pi05.context_action import enable
        enable(model)
        prepare_context_only(model,snapshot,prompt,torch)
        from flash_rt.api import VLAModel
        fast = VLAModel(wrist_frontend(model._pipe,torch,fast_kind='pool2_agent',no_state=False,
            preserve_positions=True,external_selector='divprune',external_motion_tokens=16,
            state_prompt_capacity=104),'torch')
        enable(fast)
        fast_snapshot = dict(snapshot,state=snapshot['state'][:8])
        prepare_context_only(fast,fast_snapshot,prompt,torch)
        release_unused_native_expert(model,torch)
        release_unused_native_expert(fast,torch)
        expert = OriginalOpenPiExpert(cfg,50,5)
        expert.refresh_context(model._pipe.pipeline.attn.enc_K, model._pipe.pipeline.attn.enc_V,
            int(model._pipe.pipeline.attn.dec_devpos.item()))
        align_expert_positions(expert,320,104,fast_prefixes={item['valid'] for item in items if item['mode']!='full'})
        enable_exact_flash_probe(expert)
        enable_fused_norm(expert,cfg['output_root'])
        enable_condition_cache(expert,graph_live=True)
        tap = ExpertFeatureTap(expert,torch,layout=plan.get('feature_layout','pooled'))
        ordered = sorted((item for item in items if not done[item['index']]),
            key=lambda item:(item['mode']!='full',item['valid'],item['episode'],item['prediction_id']))
        # Observe both geometry paths early; these are real dataset requests,
        # not extra rollout/calibration samples and never repeated afterward.
        early = [next((item for item in ordered if item['mode']==mode),None) for mode in ('full','pool2_agent')]
        early.append(next((item for item in ordered if episodes[item['episode']].get('method')=='q10'),None))
        early = list({item['index']:item for item in early if item is not None}.values())
        ordered = early + [item for item in ordered if item not in early]
        progress('collecting', status='running')
        with metadata_path.open('a') as metadata:
            for position,item in enumerate(ordered,1):
                memory = {k:v for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
                if int(memory['MemAvailable'].split()[0]) < 600*1024:
                    raise RuntimeError('Available RAM below600MiB; completed features retained for resume')
                request,snapshot = inputs(item)
                ep = episodes[item['episode']]
                prompt = prompts[ep['suite'],ep['task_id']]
                active = model if item['mode']=='full' else fast
                if item['mode']!='full':
                    snapshot['state']=snapshot['state'][:8]
                    active._pipe.pipeline.set_motion_indices(request['context_refresh'].get('motion_token_indices',[]))
                pipe,pl = active._pipe,active._pipe.pipeline
                with torch.cuda.stream(pipe._graph_torch_stream):
                    active.set_prompt(prompt,state=snapshot['state'])
                    pipe._fill_img_buf(snapshot)
                    pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf,pl.input_images_buf,pipe._graph_torch_stream.cuda_stream)
                    pl._context_graph.replay(pl._graph_stream)
                pipe._graph_torch_stream.synchronize()
                valid = int(pl.attn.dec_devpos.item())
                if valid!=item['valid']:
                    raise RuntimeError(f"Saved context length {item['valid']} differs from replay {valid}")
                expert.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                solver = request['rtc_solver_inputs']
                noise = torch.as_tensor(np.asarray(solver['noise'],dtype=np.float32),device='cuda')
                previous = None if solver['previous_raw'] is None else torch.as_tensor(
                    np.asarray(solver['previous_raw'],dtype=np.float32),device='cuda')
                if previous is not None:
                    if bank is None:
                        bank=SourceStepBank(expert,[valid],noise,beta=5.,resident=True,max_lengths=1,
                            cache_limit=4,verify=False)
                    elif valid not in bank.active_lengths:
                        bank.prepare_resident([valid],noise,beta=5.)
                    bound,graph=bank.select(valid)
                    bound.refresh_context(pl.attn.enc_K,pl.attn.enc_V,valid)
                tap.reset()
                t=time.monotonic()
                if previous is None:
                    raw,_=sample_original_rtc(expert,noise,None,0,0,beta=5.,author_order=True)
                else:
                    raw,_=graph.replay(noise,previous,solver['delay'],solver['executed'])
                result=tap.read(5)
                if result.shape!=feature_shape or not np.isfinite(result).all():
                    raise RuntimeError('Invalid extracted feature')
                error=float(np.max(np.abs(raw.cpu().numpy()-np.asarray(request['actions_raw'],dtype=np.float32))))
                features[item['index']]=result
                done[item['index']]=True
                metadata.write(json.dumps(dict(index=item['index'],mode=item['mode'],valid=valid,
                    expert_and_read_seconds=time.monotonic()-t,raw_max_abs_difference=error),ensure_ascii=False)+'\n')
                if position%20==0 or position==1 or position==len(ordered):
                    features.flush();done.flush();metadata.flush();progress('collecting')
        progress('features_completed',status='completed')
        save(root/'extraction_summary.json',dict(status,feature_shape=list(features.shape),
            feature_bytes=int(features.nbytes),episodes=len(episodes)))
    except BaseException as exc:
        features.flush();done.flush()
        progress('stopped',status='invalid',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if bank is not None:bank.release()
        if tap is not None:tap.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--reuse-run',type=Path)
    parser.add_argument('--append-failures',type=Path)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--feature-layout',choices=tuple(FEATURE_LAYOUTS),default='pooled')
    args=parser.parse_args()
    if args.prepare:
        if args.source_manifest:
            print(prepare_saved(args.source_manifest,args.feature_layout))
        else:
            print(prepare(args.reuse_run,args.append_failures))
        return
    if args.run_dir is None:parser.error('--prepare or --run-dir is required')
    with (INDEX.parent.parent.parent/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker(args.run_dir)


if __name__=='__main__':
    main()
