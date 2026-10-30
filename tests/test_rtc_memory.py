from types import SimpleNamespace
from unittest.mock import patch


def test_context_only_cleanup_preserves_encoder_and_blocks_native_replay():
    import torch
    import pytest
    from perceptbudget.rtc_memory import release_unused_native_expert
    context,full,decoder=object(),object(),object()
    enc=torch.ones(2)
    pl=SimpleNamespace(_context_graph=context,_graph=full,_decoder_only_graph=decoder,
                       weights={'encoder_x':1,'decoder_action_in_proj_w':2})
    pipe=SimpleNamespace(pipeline=pl,_fixed_pipeline=pl,_force_bf16=True,
        _ckpt_bf16={'encoder_x':enc,'decoder_action_in_proj_w':torch.ones(3)},
        _fp8_weights={},_int8_weights={})
    model=SimpleNamespace(_pipe=pipe)
    with patch('torch.cuda.synchronize'),patch('torch.cuda.empty_cache'),patch(
            'perceptbudget.rtc_memory.memory_record',side_effect=lambda t,s:dict(stage=s)):
        result=release_unused_native_expert(model,torch)
    assert pipe._ckpt_bf16 == {'encoder_x':enc}
    assert pl._context_graph is context
    assert pl._graph.retained_handle is full
    assert pl._decoder_only_graph.retained_handle is decoder
    assert pl.weights == {'encoder_x':1}
    assert result['released_weight_mib'] == 12/2**20
    for call in (pipe.infer,pipe._build_pipeline_weights,pl.forward,pl._graph.replay):
        with pytest.raises(RuntimeError,match='native expert released'):
            call()
    with pytest.raises(RuntimeError,match='already released'):
        release_unused_native_expert(model,torch)


def test_context_only_cleanup_releases_decoder_buffers_and_precomputed_styles():
    import torch
    from perceptbudget.rtc_memory import release_unused_native_expert

    class Buffer:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    context = object()
    enc = torch.ones(2)
    precomputed = {'attn': Buffer(32), 'ffn': Buffer(48)}
    pl = SimpleNamespace(
        _context_graph=context,
        _graph=object(),
        _decoder_only_graph=object(),
        weights={'encoder_x': 1, 'decoder_action_in_proj_w': 2,
                 'precomputed': precomputed},
        bufs={'vision_x': Buffer(128), 'decoder_rope_weights': Buffer(32),
              'decoder_style_attn': Buffer(256),
              'decoder_style_ffn': Buffer(512), 'diffusion_noise': Buffer(64),
              'x_normed_buf': Buffer(16), 'gate_buf': Buffer(24),
              'encoder_gate_buf': Buffer(40)},
    )
    pipe = SimpleNamespace(pipeline=pl, _fixed_pipeline=pl, _force_bf16=True,
        _ckpt_bf16={'encoder_x': enc, 'decoder_action_in_proj_w': torch.ones(3)},
        _fp8_weights={}, _int8_weights={}, _precomputed_styles=precomputed)
    model = SimpleNamespace(_pipe=pipe)

    with patch('torch.cuda.synchronize'), patch('torch.cuda.empty_cache'), patch(
            'perceptbudget.rtc_memory.memory_record', side_effect=lambda t, s: dict(stage=s)):
        result = release_unused_native_expert(model, torch)

    assert pl._context_graph is context
    assert set(pl.bufs) == {'vision_x', 'encoder_gate_buf', 'decoder_rope_weights'}
    assert 'precomputed' not in pl.weights
    assert pipe._precomputed_styles == {}
    assert result['released_buffer_names'] == [
        'decoder_style_attn', 'decoder_style_ffn', 'diffusion_noise',
        'x_normed_buf', 'gate_buf']
    assert result['released_buffer_mib'] == (256 + 512 + 64 + 16 + 24) / 2**20
    assert result['released_precomputed_reference_mib'] == (32 + 48) / 2**20


def test_cleanup_refuses_unwarmed_or_unfixed_pipeline():
    import pytest
    from perceptbudget.rtc_memory import release_unused_native_expert
    pipe=SimpleNamespace(pipeline=SimpleNamespace(),_force_bf16=True)
    with pytest.raises(RuntimeError,match='prewarmed fixed BF16'):
        release_unused_native_expert(SimpleNamespace(_pipe=pipe),None)
