"""Execute unmodified local LeRobot RTC class without package dependencies.

This is NOT full LeRobot policy or environment execution. AST extraction only
avoids unrelated package imports; class bodies remain unchanged.
"""
import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import create_run,load_config


def main():
    import torch
    config=load_config(repo_root=ROOT)
    run=create_run(config,'M1','lerobot-rtc-source-audit',repo_root=ROOT)
    print('run_dir='+str(run),flush=True)
    source=Path('/home/wyk/workspace/lerobot/src/lerobot')
    scope=dict(torch=torch,Tensor=torch.Tensor,dataclass=dataclass,Enum=Enum,
               math=math,logging=logging,Tracker=None,__name__=__name__)
    hashes={}
    for relative,name in (('configs/types.py','RTCAttentionSchedule'),
                          ('policies/rtc/configuration_rtc.py','RTCConfig'),
                          ('policies/rtc/modeling_rtc.py','RTCProcessor')):
        path=source/relative
        tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
        hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        destination=run/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    rtc=scope['RTCProcessor'](scope['RTCConfig'](prefix_attention_schedule=scope['RTCAttentionSchedule'].EXP,
        max_guidance_weight=5.,execution_horizon=15))
    captured={}
    rtc.track=lambda **kwargs:captured.update(kwargs)
    noise=torch.ones(1,20,2)
    previous=torch.zeros(1,15,2)
    output=rtc.denoise_step(noise,previous,5,.6,lambda x:2*x,execution_horizon=15)
    weights=rtc.get_prefix_weights(5,15,20)[None,:,None]
    error=(previous.new_zeros(1,20,2)-(noise-.6*2*noise))*weights
    true_correction=error*(1-.6*2)
    correction=captured['correction']
    result=dict(scope='unmodified local LeRobot RTC class, CPU toy; not full policy/LIBERO',
        source_hashes=hashes,device='cpu',
        correction_vs_identity_error_max_abs=float((correction-error).abs().max()),
        correction_vs_full_denoiser_vjp_max_abs=float((correction-true_correction).abs().max()),
        first_correction=float(correction[0,0,0]),
        expected_full_vjp_first=float(true_correction[0,0,0]),
        velocity_input_requires_grad=None,
        note='Local RTCProcessor computes velocity before enabling requires_grad on detached x; source behavior, not patched here')
    observed=[]
    rtc.denoise_step(noise,previous,5,.6,lambda x:(observed.append(x.requires_grad) or 2*x),execution_horizon=15)
    result['velocity_input_requires_grad']=observed
    (run/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
