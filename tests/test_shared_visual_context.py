import sys
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
