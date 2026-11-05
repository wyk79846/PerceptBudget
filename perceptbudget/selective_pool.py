"""Experimental redundancy pooling; no object/importance prediction."""

def compact(vision, merged_blocks):
    """GPU/CPU torch [2,16,16,D] -> camera-major row-ordered compact tokens.

    Merge the least dispersed normalized-feature 2x2 blocks of camera0.
    Camera1 is copied unchanged. Equal scores use stable block-index ordering.
    """
    import torch
    if vision.ndim!=4 or vision.shape[:3]!=(2,16,16):raise ValueError('Require two16x16 feature grids')
    if merged_blocks not in (0,8,32,64):raise ValueError('Use0,8,32,64 merged blocks')
    if merged_blocks==0:return vision.reshape(512,-1),None
    x=vision[0].float()
    blocks=x.reshape(8,2,8,2,-1).permute(0,2,1,3,4).reshape(64,4,-1)
    unit=torch.nn.functional.normalize(blocks,dim=-1,eps=1e-8)
    dispersion=(unit-unit.mean(dim=1,keepdim=True)).square().sum(dim=-1).mean(dim=1)
    order=torch.argsort(dispersion,stable=True)
    # Inverse permutation avoids scalar indexed assignment, which is not
    # CUDA-graph capturable on this device/PyTorch build.
    selected=torch.argsort(order)<merged_blocks
    # Match native pooling's FP32 addition order and one BF16 cast.
    means=((blocks[:,0]+blocks[:,1]+blocks[:,2]+blocks[:,3])/4).to(vision.dtype)
    index=torch.arange(256,device=x.device)
    row=index//16;col=index%16;block=(row//2)*8+col//2
    keep=~selected[block] | ((row%2==0)&(col%2==0))
    # Fixed output length, no GPU nonzero/variable-size host roundtrip.
    keys=torch.where(keep,index,index+256)
    indices=torch.argsort(keys)[:256-3*merged_blocks]
    external=vision[0].reshape(256,-1)[indices]
    external=torch.where(selected[block[indices],None],means[block[indices]],external)
    return torch.cat((external,vision[1].reshape(256,-1))),selected


def compact_both(vision,merged_blocks=32):
    """Merge redundant2x2 blocks independently in each camera.

    Scores describe local feature redundancy, not semantic task importance.
    Camera and spatial order are preserved; each view selects independently.
    The default32 blocks keeps160 tokens;8 blocks keeps232 per camera.
    """
    import torch
    external, first = compact(vision, merged_blocks)
    wrist, second = compact(vision.flip(0), merged_blocks)
    per_camera=256-3*merged_blocks
    return torch.cat((external[:per_camera], wrist[:per_camera])), torch.stack((first, second))
