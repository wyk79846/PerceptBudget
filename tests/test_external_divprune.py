import numpy as np
import pytest
from perceptbudget.external_divprune import select_indices


def test_motion_seeds_follow_original_patch_coordinates_and_zero_change_falls_back():
    from perceptbudget.external_divprune import motion_seeds
    before=np.zeros((224,224,3),dtype=np.uint8)
    after=before.copy()
    expected=list(range(17,33))
    for i in expected:
        y,x=divmod(i,16)
        after[y*14:(y+1)*14,x*14:(x+1)*14]=100
    indices,scores=motion_seeds(after,before)
    assert indices.tolist()==expected
    assert np.array_equal(np.flatnonzero(scores),expected)
    assert motion_seeds(before,before)[0].size==0


def test_motion_seeds_remain_selected_while_divprune_fills_other48_slots():
    seeds=list(range(240,256))
    selected=select_indices(np.eye(256,dtype=np.float32),64,seeds=seeds)
    assert selected.tolist()==list(range(48))+seeds
    assert len(set(selected))==64


def test_motion_pair_changes_only_selection_and_keeps_both_fast_policies():
    from perceptbudget.device_quality import configure_frequency_variant
    for view,count in [('ordinary_full',0),('ordinary_fast',16)]:
        cfg=dict(m3_ordinary_fast_pair=True,m3_external_motion_pair=True,
            m3_external_motion_tokens=16,m3_fast_own_budget=True,m3_fast_critical=True)
        configure_frequency_variant(cfg,view)
        assert cfg['m3_ordinary_fast'] and cfg['m3_fast_own_budget'] and cfg['m3_fast_critical']
        assert cfg['m3_external_motion_tokens']==count


def test_motion_cuda_graph_replay_preserves_seeds_wrist_and_original_rope():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from perceptbudget.external_divprune import load
    lib=load();gen=torch.Generator(device='cuda').manual_seed(123)
    features=torch.randn((256,2048),generator=gen,device='cuda',dtype=torch.bfloat16)
    source=torch.randn((512,1152),generator=gen,device='cuda',dtype=torch.bfloat16)
    inverse=torch.empty(256,device='cuda');distances=torch.empty((256,256),device='cuda')
    mapping=torch.empty(320,device='cuda',dtype=torch.int32)
    output=torch.empty((320,1152),device='cuda',dtype=torch.bfloat16)
    seeds=torch.zeros(17,device='cuda',dtype=torch.int32)
    rope=torch.randn((600,256),generator=gen,device='cuda',dtype=torch.bfloat16)
    gathered=torch.empty((352,256),device='cuda',dtype=torch.bfloat16)
    def launch():
        stream=torch.cuda.current_stream().cuda_stream
        assert lib.external_divprune_seeded(features.data_ptr(),inverse.data_ptr(),distances.data_ptr(),
            mapping.data_ptr(),source.data_ptr(),output.data_ptr(),seeds.data_ptr(),stream)==0
        assert lib.external_divprune_rope(rope.data_ptr(),mapping.data_ptr(),gathered.data_ptr(),352,stream)==0
    launch();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):launch()
    for chosen in ([],list(range(240,256)),list(range(16))):
        packed=torch.zeros(17,dtype=torch.int32);packed[0]=len(chosen)
        if chosen:packed[1:17]=torch.tensor(chosen,dtype=torch.int32)
        seeds.copy_(packed);graph.replay();torch.cuda.synchronize()
        indices=mapping.cpu().long()
        expected=select_indices(features.float().cpu().numpy(),64,seeds=chosen)
        assert indices[:64].tolist()==expected.tolist()
        assert set(chosen)<=set(indices[:64].tolist())
        assert indices[64:].tolist()==list(range(256,512))
        assert torch.equal(output,source[indices.to('cuda')])
        positions=torch.cat((indices,torch.arange(512,544))).to('cuda')
        assert torch.equal(gathered,rope[positions])


def test_max_min_selects_representatives_and_restores_spatial_order():
    # Four repeated visual contents: take one of each before any repetition.
    features=np.repeat(np.eye(4,dtype=np.float32),4,axis=0)
    assert select_indices(features,4).tolist()==[0,4,8,12]
    assert select_indices(np.eye(256,dtype=np.float32),64).tolist()==list(range(64))


def test_external_selector_pair_keeps_both_full_fast_policies():
    from perceptbudget.device_quality import configure_frequency_variant
    for label,selector in [('ordinary_full','avgpool'),('ordinary_fast','divprune')]:
        config=dict(m3_ordinary_fast_pair=True,m3_external_selector_pair=True)
        configure_frequency_variant(config,label)
        assert config['m3_ordinary_fast'] is True
        assert config['m3_external_token_selector']==selector


def test_cuda_selection_gathers_values_and_original_positions_under_graph_replay():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from perceptbudget.external_divprune import load
    lib=load()
    generator=torch.Generator(device='cuda').manual_seed(123)
    projected=torch.randn((256,2048),generator=generator,device='cuda',dtype=torch.bfloat16)
    original=torch.randn((512,1152),generator=generator,device='cuda',dtype=torch.bfloat16)
    inverse=torch.empty(256,device='cuda');distances=torch.empty((256,256),device='cuda')
    mapping=torch.empty(320,device='cuda',dtype=torch.int32)
    packed=torch.empty((320,1152),device='cuda',dtype=torch.bfloat16)
    rope=torch.randn((600,256),generator=generator,device='cuda',dtype=torch.bfloat16)
    gathered=torch.empty((352,256),device='cuda',dtype=torch.bfloat16)
    def launch():
        stream=torch.cuda.current_stream().cuda_stream
        assert lib.external_divprune(projected.data_ptr(),inverse.data_ptr(),distances.data_ptr(),
            mapping.data_ptr(),original.data_ptr(),packed.data_ptr(),stream)==0
        assert lib.external_divprune_rope(rope.data_ptr(),mapping.data_ptr(),gathered.data_ptr(),352,stream)==0
    launch()
    def check():
        indices=mapping.cpu().long()
        expected=select_indices(projected.float().cpu().numpy(),64)
        assert indices[:64].tolist()==expected.tolist()
        assert indices[64:].tolist()==list(range(256,512))
        assert torch.equal(packed,original[indices.to('cuda')])
        positions=torch.cat((indices,torch.arange(512,544))).to('cuda')
        assert torch.equal(gathered,rope[positions])
        assert torch.equal(packed[64:],original[256:])
        return indices[:64].tolist()
    before=check()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):launch()
    projected.copy_(torch.randn(projected.shape,generator=generator,device='cuda',dtype=torch.bfloat16))
    graph.replay()
    after=check()
    assert before!=after
