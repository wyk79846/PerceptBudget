import sys
import json
from types import SimpleNamespace
import pytest
from perceptbudget.shared_visual_context import wrist_frontend


def test_shared_weight_tensors_but_separate_runtime_context(monkeypatch):
    created=[]
    def backend(**kwargs):
        created.append(kwargs)
        return object()
    monkeypatch.setitem(sys.modules, 'flash_rt.frontends.torch.pi05_rtx',
                        SimpleNamespace(RtxFlashAttnBackend=backend, ENC_L=18))
    tensor=object()
    full=SimpleNamespace(num_views=2, _vision_pool_factor=1,
        _ckpt_bf16={'weight':tensor}, _img_buf=['external','wrist'],
        max_prompt_len=200, chunk_size=50, pipeline=object(),
        _fixed_pipeline=object(), _prompt_pipeline_cache={1:object()},
        _flashrt_subgraph_frontend_hooks=[object()])
    clone=wrist_frontend(full,None)
    assert clone._ckpt_bf16 is not full._ckpt_bf16
    assert clone._ckpt_bf16['weight'] is tensor
    assert clone._img_buf==['wrist'] and clone.num_views==1
    assert full.num_views==2 and full.pipeline is not None
    assert clone.pipeline is None and clone._prompt_pipeline_cache=={}
    assert clone._flashrt_subgraph_frontend_hooks is not full._flashrt_subgraph_frontend_hooks
    assert created[0]['encoder_seq_max']==456


def test_shared_context_refuses_released_or_wrong_geometry():
    with pytest.raises(ValueError):
        wrist_frontend(SimpleNamespace(num_views=1),None)
    with pytest.raises(ValueError,match='before releasing'):
        wrist_frontend(SimpleNamespace(num_views=2,_vision_pool_factor=1,_pb_context_only=True),None)


def test_pooled_frontend_keeps_both_images_and_shares_weights(monkeypatch):
    created=[]
    monkeypatch.setitem(sys.modules,'flash_rt.frontends.torch.pi05_rtx',SimpleNamespace(
        RtxFlashAttnBackend=lambda **kw: created.append(kw) or object(),ENC_L=18))
    full=SimpleNamespace(num_views=2,_vision_pool_factor=1,_ckpt_bf16={'w':object()},
        _img_buf=['external','wrist'],max_prompt_len=200,chunk_size=50)
    pooled=wrist_frontend(full,None,fast_kind='pool2')
    assert pooled.num_views==2 and pooled._vision_pool_factor==2
    assert pooled._img_buf is full._img_buf
    assert pooled._ckpt_bf16['w'] is full._ckpt_bf16['w']
    assert full._vision_pool_factor==1 and created[0]['encoder_seq_max']==712


def test_historical_visual_prewarm_is_shapes_only_with_provenance_gate(tmp_path,monkeypatch):
    from perceptbudget.shared_visual_context import historical_visual_prefixes
    monkeypatch.setattr('perceptbudget.shared_visual_context.VISUAL_PREFIX_REFERENCES',{'suite':('full','wrist')})
    config=dict(output_root=str(tmp_path),checkpoint_provenance_revision='rev',H=50,N=5,
                state_enabled=True,state_prompt_capacity=200)
    for name,view,lengths in (('full','full',[664,669]),('wrist','wrist',[408,412])):
        run=tmp_path/'M2'/name
        (run/'rtc').mkdir(parents=True)
        (run/'run.json').write_text(json.dumps({'config_snapshot':dict(config,m2_view=view)}))
        (run/'rtc/q5_requests.json').write_text(json.dumps([
            {'rtc_guidance':{'valid_prefix_length':n}} for n in lengths]))
    lengths,record=historical_visual_prefixes(config,'suite',0,665,409)
    assert lengths==list(range(664,670))+list(range(408,414))
    assert record['kind']=='historical_shape_envelope_not_exhaustive'
    with pytest.raises(ValueError,match='mismatch'):
        historical_visual_prefixes(dict(config,N=10),'suite',0,665,409)
