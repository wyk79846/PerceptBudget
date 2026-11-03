"""Experimental redundancy pooling; no object/importance prediction."""

def compact(vision, merged_blocks):
    """GPU/CPU torch [2,16,16,D] -> camera-major row-ordered compact tokens.

    Merge the least dispersed normalized-feature 2x2 blocks of camera0.
    Camera1 is copied unchanged. Equal scores use stable block-index ordering.
    """
    import torch
    if vision.ndim!=4 or vision.shape[:3]!=(2,16,16):raise ValueError('Require two16x16 feature grids')
    if merged_blocks not in (0,32,64):raise ValueError('Use0,32,64 merged blocks')
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


def compact_both(vision):
    """Keep160 tokens per camera by merging32 redundant2x2 blocks in each.

    Scores describe local feature redundancy, not semantic task importance.
    Camera and spatial order are preserved; each view selects independently.
    """
    import torch
    external, first = compact(vision, 32)
    wrist, second = compact(vision.flip(0), 32)
    return torch.cat((external[:160], wrist[:160])), torch.stack((first, second))
