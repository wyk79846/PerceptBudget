"""Explicit adapter for the pinned LeRobot H50 LIBERO release; no global switch."""
import json
from pathlib import Path

DIRECTORY=Path('/home/wyk/workspace/openpi_cache/openpi-assets/checkpoints/lerobot_pi05_libero_h50_quantiles_v044')
PROFILE='lerobot_h50_quantiles_v044'


def configure(config):
    from safetensors import safe_open
    manifest=json.loads((DIRECTORY/'download_manifest.json').read_text())
    if manifest['status']!='download_verified':
        raise RuntimeError('H50 checkpoint download/checksum not complete')
    stats={'_profile':PROFILE}
    for filename,feature,name in (
        ('policy_preprocessor_step_2_normalizer_processor.safetensors','observation.state','state'),
        ('policy_postprocessor_step_0_unnormalizer_processor.safetensors','action','actions')):
        with safe_open(DIRECTORY/filename,framework='np') as source:
            stats[name]={key:source.get_tensor(feature+'.'+key).tolist() for key in ('q01','q99')}
    path=DIRECTORY/'perceptbudget_norm_stats.json'
    if not path.exists():
        path.write_text(json.dumps(stats,indent=2))
    elif json.loads(path.read_text())!=stats:
        raise RuntimeError('derived statistics differ; refusing overwrite')
    config.update(model_id=PROFILE,checkpoint_path=str(DIRECTORY),
                  model_path=str(DIRECTORY/'model.safetensors'),norm_stats_path=str(path),
                  state_prompt_capacity=200,checkpoint_key_prefix='model.',checkpoint_profile=PROFILE,
                  checkpoint_provenance_revision=manifest['revision'],
                  checkpoint_validation='published structural mapping; live graph/eager gate; not old-weight author gate')


def normalize_state(state,stats):
    import numpy as np
    low=np.asarray(stats['state']['q01'],dtype=np.float32)
    high=np.asarray(stats['state']['q99'],dtype=np.float32)
    denominator=high-low
    denominator=np.where(denominator==0,np.float32(1e-8),denominator)
    normalized=2*(np.asarray(state,dtype=np.float32)-low)/denominator-1
    return np.pad(normalized,(0,32-len(normalized))).astype(np.float32)


def decode_actions(raw,stats):
    import numpy as np
    x=np.asarray(raw,dtype=np.float32)[...,:7]
    low=np.asarray(stats['actions']['q01'],dtype=np.float32)
    high=np.asarray(stats['actions']['q99'],dtype=np.float32)
    denominator=high-low
    denominator=np.where(denominator==0,np.float32(1e-8),denominator)
    return (x+1)*denominator/2+low


def image_input(image):
    """LeRobot float bilinear resize, represented in FlashRT's 0..255 units.

    Keep fractional pixel values; do not round back to uint8 after resizing.
    Missing third camera has a false mask upstream, so omit masked tokens here.
    """
    import numpy as np
    import torch
    x=torch.from_numpy(np.ascontiguousarray(image[::-1,::-1])).permute(2,0,1)[None].float()/255.
    if x.shape[-2:]!=(224,224):
        x=torch.nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
    return (x[0].permute(1,2,0)*255).numpy()
