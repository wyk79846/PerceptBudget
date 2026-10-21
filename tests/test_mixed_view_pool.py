import pytest
from perceptbudget.mixed_view_pool import visual_tokens,pipeline_type


@pytest.mark.parametrize('factor',[2,4])
@pytest.mark.parametrize('view',['agent','wrist'])
def test_mixed_pipeline_allocates_shortened_encoder_before_rope(factor,view):
    class Base:
        def _allocate_buffers(self):
            return {'seen_encoder_length':self.encoder_seq_len}
    cls=pipeline_type(Base,view)
    pl=cls()
    pl.num_views=2;pl.vision_pool_factor=factor;pl.max_prompt_len=200;pl.chunk_size=50
    count=visual_tokens(dict(m2_vision_pool_factor=factor,m2_pool_view=view))
    assert pl._allocate_buffers()['seen_encoder_length']==count+200
    assert pl.total_kv==count+250
