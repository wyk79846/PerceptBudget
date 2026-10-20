"""Pinned upstream LeRobot PI05/RTC/ActionQueue, with a LIBERO-only harness.

Not the official real-robot CLI or a full upstream LIBERO async reproduction.
No FlashRT imports. No training or modifications to upstream model/RTC code.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import create_run,load_config
from scripts import eval_q as base


def prediction_ready(future,step,origin,controlled):
    """In controlled mode neither early commit nor extra simulation steps are allowed."""
    if future is None:
        return False
    if controlled:
        if step-origin<5:
            return False
        future.result()  # wait at exactly five simulated control steps
    return future.done()


def load_policy(config):
    """Strict streamed assignment avoids randomly initializing multiple GiB."""
    import torch
    from safetensors import safe_open
    from lerobot.configs.types import FeatureType,PolicyFeature,RTCAttentionSchedule
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,))}
    features.update({f'observation.images.{name}':PolicyFeature(type=FeatureType.VISUAL,shape=(3,224,224))
                     for name in ('base_0_rgb','left_wrist_0_rgb')})
    cfg=PI05Config(device='cpu',dtype='bfloat16',chunk_size=25,n_action_steps=5,
        num_inference_steps=5,max_state_dim=8,input_features=features,
        output_features={'action':PolicyFeature(type=FeatureType.ACTION,
            shape=(32 if config.get('controlled_delay_steps')==5 else 7,))},
        rtc_config=RTCConfig(enabled=True,max_guidance_weight=5.,execution_horizon=20,
                            prefix_attention_schedule=RTCAttentionSchedule.EXP))
    cfg.device='meta'  # construction only; no initialized random-weight allocation
    with torch.device('meta'):
        policy=PI05Policy(cfg)
    expected=policy.state_dict()
    loaded={}
    with safe_open(config['model_path'],framework='pt',device='cpu') as checkpoint:
        for key in checkpoint.keys():
            fixed=policy._fix_pytorch_state_dict_keys({key:checkpoint.get_tensor(key)},cfg)
            for name,value in fixed.items():
                name=name if name.startswith('model.') else 'model.'+name
                if name not in expected:
                    raise RuntimeError('unexpected checkpoint key: '+name)
                if value.shape!=expected[name].shape:
                    raise RuntimeError('checkpoint shape mismatch: '+name)
                loaded[name]=value.to(device='cuda',dtype=expected[name].dtype)
    # safetensors omits the shared embedding; restore the model's declared tie.
    prefix='model.paligemma_with_expert.paligemma.'
    embedding=prefix+'model.language_model.embed_tokens.weight'
    head=prefix+'lm_head.weight'
    if embedding not in loaded and head in loaded:
        loaded[embedding]=loaded[head]
    policy.load_state_dict(loaded,strict=True,assign=True)
    del loaded
    for module in policy.modules():
        for name,buffer in list(module.named_buffers(recurse=False)):
            if not buffer.is_meta:
                continue
            if name=='position_ids':
                replacement=torch.arange(buffer.shape[-1],device='cuda').expand(buffer.shape)
            elif name=='inv_freq' and hasattr(module,'rope_init_fn'):
                replacement,scale=module.rope_init_fn(module.config,device=torch.device('cuda'))
                module.attention_scaling=scale
                module.original_inv_freq=replacement
            else:
                raise RuntimeError('unhandled meta buffer: '+name)
            module.register_buffer(name,replacement,persistent=name not in module._non_persistent_buffers_set)
    cfg.device='cuda'
    policy.eval().requires_grad_(False)
    if any(t.is_meta for t in list(policy.parameters())+list(policy.buffers())):
        raise RuntimeError('unmaterialized model tensor')
    torch.cuda.synchronize()
    return policy,dict(strict=True,keys=len(expected),tied_embedding_from=head,
                       method='meta construction, original key fixer, strict streamed CUDA assignment')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--controlled-delay-five',action='store_true',
                        help='Quality-only: five old control steps then wait, full raw32 RTC target')
    args=parser.parse_args()
    config=load_config(repo_root=ROOT)
    config.update(H=25,N=5,state_enabled=True,state_prompt_capacity=80,candidate_q=[5],
                  execution_mode='lerobot_native_rtc_libero_harness',python_executable=sys.executable)
    config['screen'].update(suites=['libero_spatial'],task_ids=[2],seed=7,init_state_index=0,trials_per_task=1)
    if args.controlled_delay_five:
        config.update(controlled_delay_steps=5,execution_mode='lerobot_controlled_d5_quality',
                      rtc_raw_action_dim=32)
    os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run=create_run(config,'M1','lerobot-rtc',repo_root=ROOT)
        print('run_dir='+str(run),flush=True)
        shutil.copy2(__file__,run/'eval_lerobot_rtc.py')
        result=dict(status='running',scope='unchanged upstream model/RTC/queue; custom LIBERO harness')
        env=None
        try:
            import numpy as np
            import torch
            import sentencepiece as spm
            import lerobot
            import transformers
            from lerobot.policies.rtc.action_queue import ActionQueue
            from lerobot.policies.rtc.latency_tracker import LatencyTracker
            from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
            from lerobot.processor.core import TransitionKey
            from perceptbudget.libero_input import load_libero_norm_stats
            source_root=Path(lerobot.__file__).resolve().parents[2]
            result['lerobot_revision']=subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
            result['versions']=dict(torch=torch.__version__,transformers=transformers.__version__)
            result['source_hashes']={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (source_root/'src/lerobot/policies/pi05/modeling_pi05.py',
                          source_root/'src/lerobot/policies/rtc/modeling_rtc.py',
                          source_root/'src/lerobot/policies/rtc/action_queue.py')}
            torch.use_deterministic_algorithms(True)
            base._seed_process(7,np,torch)
            policy,result['weight_load']=load_policy(config)
            print('model_loaded='+json.dumps(result['weight_load']),flush=True)
            stats=load_libero_norm_stats(config['norm_stats_path'])
            tokenizer=spm.SentencePieceProcessor(model_file='/home/wyk/.cache/flash_rt/paligemma_tokenizer.model')
            prepare_text=Pi05PrepareStateTokenizerProcessorStep(max_state_dim=8)
            task=base._load_tasks([2],['libero_spatial'])[0]
            def batch(observation):
                snap=base._make_snapshot(observation,stats,True)
                transition={TransitionKey.OBSERVATION:{'observation.state':torch.tensor(snap['state'])[None]},
                            TransitionKey.COMPLEMENTARY_DATA:{'task':[task['instruction']]}}
                text=prepare_text(transition)[TransitionKey.COMPLEMENTARY_DATA]['task'][0]
                tokens=tokenizer.encode(text,add_bos=True)
                if len(tokens)>80:raise RuntimeError('state prompt exceeds unchanged capacity80')
                padded=tokens+[0]*(80-len(tokens))
                data={'observation.language.tokens':torch.tensor([padded],device='cuda'),
                      'observation.language.attention_mask':torch.tensor([[True]*len(tokens)+[False]*(80-len(tokens))],device='cuda')}
                for name,image in zip(('base_0_rgb','left_wrist_0_rgb'),snap['images']):
                    data['observation.images.'+name]=torch.from_numpy(image.copy()).permute(2,0,1)[None].to('cuda').float()/255.
                return data
            def infer(data,previous,delay,audit=None):
                start=time.monotonic()
                kwargs={}
                if args.controlled_delay_five:
                    kwargs['noise']=torch.randn(1,25,32,device='cuda',dtype=torch.float32)
                actions=policy.predict_action_chunk(data,prev_chunk_left_over=previous,inference_delay=delay,**kwargs)
                torch.cuda.synchronize()
                if not torch.isfinite(actions).all():raise RuntimeError('nonfinite LeRobot actions')
                actions=actions[0].detach()
                q01=torch.tensor(stats['actions']['q01'][:7],device='cuda')
                q99=torch.tensor(stats['actions']['q99'][:7],device='cuda')
                # Keep the existing experiment's LIBERO action adapter, including clipping.
                physical=(actions[:,:7].clamp(-1,1)+1)*.5*(q99-q01+1e-6)+q01
                if audit is not None:
                    audit.update(noise=kwargs['noise'][0].cpu().tolist(),
                        previous_raw=None if previous is None else previous.detach().cpu().tolist(),
                        delay=delay,actions_raw=actions.cpu().tolist(),
                        language_tokens=data['observation.language.tokens'][0].cpu().tolist(),
                        language_mask=data['observation.language.attention_mask'][0].cpu().tolist())
                # The control queue must be host-resident: .cpu() during a
                # background CUDA forward would serialize control behind it.
                return actions,physical.cpu(),time.monotonic()-start
            base._patch_egl_cleanup()
            _,cls=base._libero_api()
            env=base._make_env(cls,task);env._pb_lock_fixtures=True
            observation,_=base._prepare_env_episode(env,task,np)
            data=batch(observation)
            bootstrap,_,unguided=infer(data,None,0)
            _,_,guided=infer(data,bootstrap[5:],5)
            result['preflight_seconds']=dict(unguided=unguided,guided=guided)
            print('preflight='+json.dumps(result['preflight_seconds']),flush=True)
            if args.preflight_only:
                result.update(status='completed',episode_started=False)
                return
            observation,_=base._prepare_env_episode(env,task,np)
            base._seed_process(7,np,torch)
            queue=ActionQueue(policy.config.rtc_config)
            latency=LatencyTracker()
            latency.add(guided)
            bootstrap_audit={} if args.controlled_delay_five else None
            raw,physical,_=infer(batch(observation),None,0,bootstrap_audit)
            if bootstrap_audit is not None:base._write_json_atomic(run/'bootstrap.json',bootstrap_audit)
            queue.merge(raw,physical,0,None)
            from scripts.probe_delay_buffer import snapshot_scene
            rows=[];requests=[];frames=[];pending=None;underflows=[]
            started=time.monotonic();status='timeout'
            with ThreadPoolExecutor(max_workers=1) as executor:
                for step in range(220):
                    tick=time.monotonic()
                    frames.append(np.concatenate((observation['agentview_image'],observation['robot0_eye_in_hand_image']),axis=1))
                    if prediction_ready(pending,step,request_step if pending is not None else 0,args.controlled_delay_five):
                        raw,physical,seconds=pending.result()
                        nominal=5 if args.controlled_delay_five else math.ceil((time.monotonic()-request_start)/.05)
                        consumed=queue.get_action_index()-request_index
                        if args.controlled_delay_five and consumed!=5:
                            raise RuntimeError('controlled handoff did not consume exactly five old actions')
                        queue.merge(raw,physical,nominal,request_index)
                        latency.add(time.monotonic()-request_start)
                        requests.append(dict(observation_step=request_step,commit_step=step,
                            inferred_delay=request_delay,actual_consumed=consumed,nominal_delay=nominal,
                            inference_seconds=seconds,remaining=queue.qsize(),solver_inputs=request_audit))
                        pending=None
                    if pending is None and queue.qsize()<=20:
                        request_start=time.monotonic();request_index=queue.get_action_index()
                        previous=queue.get_left_over();request_step=step
                        request_delay=5 if args.controlled_delay_five else math.ceil(latency.max()/.05)
                        request_audit={} if args.controlled_delay_five else None
                        pending=executor.submit(infer,batch(observation),previous,request_delay,request_audit)
                    action=queue.get()
                    if action is None:
                        underflows.append(step)
                        command=np.zeros(7,dtype=np.float32)
                        command[6]=rows[-1]['action'][6] if rows else -1.
                    else:command=action.cpu().numpy().astype(np.float32)
                    before=snapshot_scene(env,observation)
                    observation,reward,done,_=env.step(command)
                    success=bool(env.check_success())
                    rows.append(dict(step=step,action=command.tolist(),before=before,
                        after=snapshot_scene(env,observation),success=success,monotonic=time.monotonic()))
                    if success:status='success';break
                    if done:status='environment_done';break
                    time.sleep(max(0.,.05-(time.monotonic()-tick)))
                control_finished=time.monotonic()
            elapsed=control_finished-started
            np.savez_compressed(run/'frames.npz',frames=np.stack(frames))
            base._write_json_atomic(run/'controls.json',rows)
            base._write_json_atomic(run/'requests.json',requests)
            result.update(status='completed',episode=dict(status=status,steps=len(rows),real_seconds=elapsed,
                control_hz=len(rows)/elapsed,underflows=underflows),
                median_inference_ms=statistics.median(x['inference_seconds']*1000 for x in requests) if requests else None,
                controlled_delay_steps=5 if args.controlled_delay_five else None,
                limitations=['one task; no success-rate estimate','LIBERO harness, not upstream hardware CLI',
                    ('fixed simulated d5 with waiting, not true async; output feature32 preserves full raw RTC target'
                     if args.controlled_delay_five else
                     'official example nominal wall-time delay and output7 retained; actual consumption logged'),
                    'state8 and local SentencePiece80 match project; not LeRobot default padded state32',
                    'same project clipped quantile action adapter; not full official processor pipeline',
                    'full LeRobot encoder differs from FlashRT; speed and algorithm not isolated'])
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if env is not None:env.close()
            base._write_json_atomic(run/'summary.json',result)
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
