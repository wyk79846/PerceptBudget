"""Isolated capacity200/64/200 traces with fixed inputs/noise and poison gates.

Diagnostic only: hooks record eager native-layer buffers after graph warmup.
No edits to FlashRT, model weights, production inference, or task execution.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import create_run,load_config
from scripts import eval_q as base


def worker(run,capacity,label):
    import fcntl
    config=load_config(repo_root=ROOT)
    config.update(H=20,N=5,state_enabled=True,state_prompt_capacity=capacity)
    config['screen'].update(seed=7,init_state_index=0,suites=['libero_spatial'])
    directory=run/label
    directory.mkdir(exist_ok=True)
    result={'status':'loading','capacity':capacity,'label':label,'H':20,'N':5,
            'state_enabled':True,'flashrt_revision':config['flashrt_revision']}
    with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',
                          PYOPENGL_PLATFORM='egl',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        base._patch_egl_cleanup()
        torch,_=base._check_cuda_and_native(config)
        torch.use_deterministic_algorithms(True)
        import numpy as np
        from perceptbudget.libero_input import load_libero_norm_stats
        from perceptbudget.rtc_expert import DifferentiableExpert
        from perceptbudget.rtc_guidance import soft_mask_tensor
        from flash_rt.subgraphs.pi05.context_action import enable
        from flash_rt.subgraphs.pi05.rtc_action_vjp import CapturedRtcChunk
        fixture=run/'fixture.npz'
        if not fixture.exists():
            task=base._load_tasks([0],['libero_spatial'])[0]
            _,env_type=base._libero_api()
            env=base._make_env(env_type,task)
            try:
                obs,_=base._prepare_env_episode(env,task,np)
                snapshot=base._make_snapshot(obs,load_libero_norm_stats(config['norm_stats_path']),True)
                noise=np.random.default_rng(7).standard_normal((20,32)).astype(np.float32)
                np.savez(fixture,image0=snapshot['images'][0],image1=snapshot['images'][1],
                         state=snapshot['state'],noise=noise)
                (run/'fixture.json').write_text(json.dumps({'prompt':task['instruction']}))
            finally: env.close()
        with np.load(fixture) as f:
            snapshot={'images':[f['image0'].copy(),f['image1'].copy()],'state':f['state'].copy()}
            fixed_noise=f['noise'].copy()
        prompt=json.loads((run/'fixture.json').read_text())['prompt']
        model=base._load_and_check_model(config,torch)
        enable(model)
        try:
            base._infer_actions(model,snapshot,prompt,20,{})
            pipe,pl=model._pipe,model._pipe.pipeline
            valid=int(pl.attn.dec_devpos.item())
            tensors={}
            phase={'kind':None,'layer':None,'gemm':0,'rms':0}
            def host_bits(pointer,shape):
                torch.cuda.synchronize()
                value=np.empty(shape,dtype=np.uint16)
                code=pipe._cudart.cudaMemcpy(ctypes.c_void_p(value.ctypes.data),
                    ctypes.c_void_p(int(pointer)),value.nbytes,2)
                if code: raise RuntimeError(f'cudaMemcpy failed {code}')
                return value
            def buffer(name,width,rows=None):
                return host_bits(pl.bufs[name].ptr.value,(rows or valid,width))
            def kv():
                torch.cuda.synchronize()
                return tuple(t[:,:valid].contiguous().view(torch.uint16).cpu().numpy().copy()
                             for t in (pl.attn.enc_K,pl.attn.enc_V))
            initial_kv=kv()
            try:
                from openpi.models.tokenizer import PaligemmaTokenizer
                ids,mask=PaligemmaTokenizer(max_len=200).tokenize(prompt,state=snapshot['state'])
                tensors['00_token_ids']=np.asarray(ids)[np.asarray(mask,dtype=bool)]
                result['tokenizer']='openpi'
            except (ImportError,FileNotFoundError,OSError,RuntimeError):
                from flash_rt.utils.paligemma_tokenizer import load_paligemma_sentencepiece
                from flash_rt.core.utils.pi05_prompt import format_pi05_prompt
                tensors['00_token_ids']=np.asarray(load_paligemma_sentencepiece().Encode(
                    format_pi05_prompt(prompt,snapshot['state']),add_bos=True))
                result['tokenizer']='sentencepiece_native_fallback'
            language=host_bits(pl._lang_embeds_buf.ptr.value,(capacity,2048))
            tensors['01_language_embeddings']=language[:valid-512]
            result['language_padding_nonzero']=int(np.count_nonzero(language[valid-512:]))
            tensors['02_encoder_rope']=buffer('encoder_rope_weights',256)
            tensors['03_decoder_rope']=buffer('decoder_rope_weights',256,20)
            result.update(status='tracing',valid_prefix=valid,token_count=valid-512,
                enc_seqused=pl.attn.enc_seqused.cpu().tolist(),
                dec_seqused=pl.attn.dec_seqused.cpu().tolist(),
                decoder_position=pl.attn.dec_devpos.cpu().tolist(),fixed_shape=pl._fixed_shape,
                fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
            original_vision,original_encoder=pl._vision_layer,pl._encoder_layer
            original_gemm,original_fvk,original_attn=pl.gemm,pl.fvk,pl.attn.run
            class GemmTrace:
                def __getattr__(self,name): return getattr(original_gemm,name)
                def bf16_nn(self,a,b,d,m,n,k,**kwargs):
                    first=phase['layer']==0 and (phase['kind']=='encoder' or
                          (phase['kind']=='decoder' and phase.get('step')==0))
                    index=phase['gemm']
                    stem='20_encoder0' if phase['kind']=='encoder' else '40_native0'
                    if first:
                        tensors[f'{stem}_gemm{index}_input']=host_bits(a,(min(valid,m),k))
                    value=original_gemm.bf16_nn(a,b,d,m,n,k,**kwargs)
                    if first:
                        tensors[f'{stem}_gemm{index}_output']=host_bits(d,(min(valid,m),n))
                        phase['gemm']+=1
                    return value
            class FvkTrace:
                def __getattr__(self,name):
                    native=getattr(original_fvk,name)
                    if name!='rms_norm': return native
                    def wrapped(*args,**kwargs):
                        value=native(*args,**kwargs)
                        if phase['kind']=='encoder' and phase['layer']==0:
                            tensors[f'20_encoder0_rms{phase["rms"]}']=host_bits(args[2],(valid,args[4]))
                            phase['rms']+=1
                        return value
                    return wrapped
            def vision(*args,**kwargs):
                phase.update(kind='vision',layer=args[0])
                value=original_vision(*args,**kwargs)
                tensors[f'10_vision_layer{args[0]:02d}']=buffer('vision_x',1152,512)
                return value
            def encoder(*args,**kwargs):
                phase.update(kind='encoder',layer=args[0])
                if args[0]==0: tensors['19_encoder_input']=buffer('encoder_x',2048)
                value=original_encoder(*args,**kwargs)
                tensors[f'21_encoder_layer{args[0]:02d}_key']=pl.attn.enc_K[args[0],:valid].contiguous().view(torch.uint16).cpu().numpy().copy()
                tensors[f'21_encoder_layer{args[0]:02d}_value']=pl.attn.enc_V[args[0],:valid].contiguous().view(torch.uint16).cpu().numpy().copy()
                if args[0]<17: tensors[f'21_encoder_layer{args[0]:02d}_output']=buffer('encoder_x',2048)
                return value
            def attention(*args,**kwargs):
                value=original_attn(*args,**kwargs)
                if phase['kind']=='encoder' and phase['layer']==0:
                    tensors['20_encoder0_attention']=host_bits(value,(valid,2048))
                if phase['kind']=='decoder' and phase['layer']==0 and phase.get('step')==0:
                    tensors['40_native0_action_key']=pl.attn.enc_K[0,valid:valid+20].contiguous().view(torch.uint16).cpu().numpy().copy()
                    tensors['40_native0_action_value']=pl.attn.enc_V[0,valid:valid+20].contiguous().view(torch.uint16).cpu().numpy().copy()
                    tensors['40_native0_attention']=host_bits(value,(20,2048))
                return value
            pl._vision_layer,pl._encoder_layer=vision,encoder
            pl.gemm,pl.fvk,pl.attn.run=GemmTrace(),FvkTrace(),attention
            try:
                with torch.cuda.stream(pipe._graph_torch_stream):
                    model.set_prompt(prompt,state=snapshot['state'])
                    pipe._fill_img_buf({'images':snapshot['images']})
                    pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf,pl.input_images_buf,
                                                            pipe._graph_torch_stream.cuda_stream)
                    tensors['04_normalized_images']=buffer('observation_images_normalized',224*3,2*224)
                    pl._copy_lang_embeds_to_encoder_x(stream=pipe._graph_torch_stream.cuda_stream)
                    pl.vision_encoder(pipe._graph_torch_stream.cuda_stream)
                    pl.transformer_encoder(pipe._graph_torch_stream.cuda_stream)
                torch.cuda.synchronize()
            finally:
                pl._vision_layer,pl._encoder_layer=original_vision,original_encoder
                pl.gemm,pl.fvk,pl.attn.run=original_gemm,original_fvk,original_attn
            traced_kv=kv()
            result['traced_eager_equals_warm_graph_kv']=all(np.array_equal(a,b) for a,b in zip(initial_kv,traced_kv))
            # Poison masked encoder rows, preserving every valid token.
            poisoned=language.copy()
            poisoned[valid-512:]=np.uint16(np.asarray(100.,dtype=np.float32).view(np.uint32)>>16)
            pl._lang_embeds_buf.upload(poisoned)
            with torch.cuda.stream(pipe._graph_torch_stream): pl._context_graph.replay(pl._graph_stream)
            poisoned_kv=kv()
            result['encoder_padding_poison_invariant']=all(np.array_equal(a,b) for a,b in zip(initial_kv,poisoned_kv))
            pl._lang_embeds_buf.upload(language)
            with torch.cuda.stream(pipe._graph_torch_stream): pl._context_graph.replay(pl._graph_stream)
            torch.cuda.synchronize()
            expert=DifferentiableExpert(pipe,attention_backend='flash-varlen').snapshot_context(fixed_prefix_capacity=True)
            expert.enable_fused_attention_ops(config['output_root'])
            tensors['30_decoder_styles_attn']=expert._context['attn'].view(torch.uint16).cpu().numpy().copy()
            tensors['31_decoder_styles_ffn']=expert._context['ffn'].view(torch.uint16).cpu().numpy().copy()
            noise=torch.as_tensor(fixed_noise,device='cuda').bfloat16().float()
            tensors['32_fixed_noise']=noise.cpu().numpy()
            target=torch.zeros_like(noise); target[:15]=noise[5:]
            weights=soft_mask_tensor(20,5,5,device=noise.device)
            graph=CapturedRtcChunk(expert.step_delta,noise,num_steps=5)
            raw,norms,_=(v.detach().clone() for v in graph.replay(noise,target,weights))
            tensors['33_rtc_raw']=raw.cpu().numpy()
            tensors['34_rtc_vjp_norm']=norms.cpu().numpy()
            with torch.no_grad():
                expert._context['key'][:,valid:]=11.
                expert._context['value'][:,valid:]=-13.
            poisoned_raw=graph.replay(noise,target,weights)[0].detach().clone()
            result['expert_padding_poison_invariant']=bool(torch.equal(raw,poisoned_raw))
            tensors['35_rtc_raw_poisoned_padding']=poisoned_raw.cpu().numpy()
            # Native bootstrap is a separate decoder from the RTC expert.
            # Reproduce its seeded BF16 noise, and trace the original kernels.
            base._seed_process(7,np,torch)
            bootstrap_noise=torch.empty_like(pipe._noise_buf)
            with torch.cuda.stream(pipe._graph_torch_stream): bootstrap_noise.normal_()
            torch.cuda.synchronize()
            tensors['39_native_seed7_noise']=bootstrap_noise.float().cpu().numpy()
            original_decoder=pl._decoder_layer
            def decoder(*args,**kwargs):
                layer,step=args[0],args[1]
                phase.update(kind='decoder',layer=layer,step=step)
                if step==0 and layer==0:
                    tensors['40_native0_hidden_input']=buffer('decoder_x',1024,20)
                value=original_decoder(*args,**kwargs)
                tensors[f'41_native_step{step:02d}_layer{layer:02d}']=buffer('decoder_x',1024,20)
                return value
            pl._decoder_layer,pl.gemm,pl.attn.run=decoder,GemmTrace(),attention
            phase['gemm']=0
            try:
                with torch.cuda.stream(pipe._graph_torch_stream):
                    pipe._copy_tensor_to_pipeline_buf_stream(bootstrap_noise,pl.input_noise_buf,
                                                            pipe._graph_torch_stream.cuda_stream)
                    pl.transformer_decoder(stream=pipe._graph_torch_stream.cuda_stream)
                tensors['42_native_raw']=buffer('diffusion_noise',32,20)
            finally:
                pl._decoder_layer,pl.gemm,pl.attn.run=original_decoder,original_gemm,original_attn
            original_split=pl.attn._call_fvk_fa2_seqused_splitkv
            def unsplit(q,k,v,o,lse,seqused,**kwargs):
                return pl.attn._call_fvk_fa2_seqused(q,k,v,o,lse,seqused,
                    stream=kwargs.get('stream',0),softmax_scale=kwargs.get('softmax_scale'))
            pl.attn._call_fvk_fa2_seqused_splitkv=unsplit
            try:
                with torch.cuda.stream(pipe._graph_torch_stream):
                    pipe._copy_tensor_to_pipeline_buf_stream(bootstrap_noise,pl.input_noise_buf,
                                                            pipe._graph_torch_stream.cuda_stream)
                    pl.transformer_decoder(stream=pipe._graph_torch_stream.cuda_stream)
                tensors['43_native_raw_unsplit']=buffer('diffusion_noise',32,20)
            finally: pl.attn._call_fvk_fa2_seqused_splitkv=original_split
            np.savez_compressed(directory/'trace.npz',**tensors)
            result.update(status='completed',stage_order=list(tensors),weights_frozen=all(
                not w.requires_grad and w.grad is None for w in expert.weights.values()))
        except BaseException as exc:
            result.update(status='invalid',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            base._write_json_atomic(directory/'summary.json',result)
            print(json.dumps(result),flush=True)


def compare(run):
    import numpy as np
    result={'status':'completed','comparisons':{}}
    def as_float(x):
        return (x.astype(np.uint32)<<16).view(np.float32) if x.dtype==np.uint16 else x.astype(np.float32)
    for a,b,label in (('capacity200a','capacity64','capacity_change'),('capacity200a','capacity200b','same_capacity_repeat')):
        meta=json.loads((run/a/'summary.json').read_text())
        rows=[]
        with np.load(run/a/'trace.npz') as left,np.load(run/b/'trace.npz') as right:
            for key in meta['stage_order']:
                x,y=left[key],right[key]
                if x.shape!=y.shape: raise ValueError(f'shape mismatch {key}')
                xf,yf=as_float(x),as_float(y)
                diff=xf-yf
                rows.append({'stage':key,'exact':bool(np.array_equal(x,y)),
                    'max_abs':float(np.max(np.abs(diff))),
                    'relative_l2':float(np.linalg.norm(diff.ravel())/max(np.linalg.norm(xf.ravel()),1e-12)),
                    'changed_fraction':float(np.count_nonzero(x!=y)/x.size)})
        result['comparisons'][label]={'first_difference':next((r for r in rows if not r['exact']),None),'stages':rows}
    result['worker_checks']={name:json.loads((run/name/'summary.json').read_text())
                             for name in ('capacity200a','capacity64','capacity200b')}
    base._write_json_atomic(run/'summary.json',result)
    print(json.dumps({k:v['first_difference'] for k,v in result['comparisons'].items()}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker-run',type=Path)
    parser.add_argument('--capacity',type=int)
    parser.add_argument('--label')
    parser.add_argument('--fixture-from',type=Path)
    args=parser.parse_args()
    if args.worker_run: return worker(args.worker_run,args.capacity,args.label)
    config=load_config(repo_root=ROOT)
    config.update(H=20,N=5,state_enabled=True,diagnostic_capacities=[200,64,200])
    run=create_run(config,'M1','capacity-numeric-diagnosis',repo_root=ROOT)
    if args.fixture_from:
        for name in ('fixture.npz','fixture.json'): shutil.copy2(args.fixture_from/name,run/name)
    shutil.copy2(__file__,run/'diagnose_state_capacity.py')
    print('run_dir='+str(run),flush=True)
    for capacity,label in ((200,'capacity200a'),(64,'capacity64'),(200,'capacity200b')):
        command=[config['python_executable'],'-u',str(ROOT/'scripts/guarded_run.py'),'--',
            config['python_executable'],'-u',str(Path(__file__).resolve()),'--worker-run',str(run),
            '--capacity',str(capacity),'--label',label]
        print(label+' start',flush=True)
        with (run/(label+'.log')).open('w') as stream:
            proc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if proc.returncode:
            base._write_json_atomic(run/'summary.json',{'status':'stopped_invalid','worker':label,'exit_code':proc.returncode})
            raise RuntimeError(label+' failed')
        print(label+' completed',flush=True)
    compare(run)


if __name__=='__main__': main()
