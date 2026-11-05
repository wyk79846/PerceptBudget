from types import SimpleNamespace

import pytest

from perceptbudget.expanded_context import backend_type,ExpandedQKV


def test_full_kv_validity_is_independent_of_compact_query_count():
    class Backend:
        def __init__(self,**kwargs):pass
        def set_fixed_valid_len(self,n):self.valid=n
    backend=backend_type(Backend,320)(encoder_seq_max=712)
    backend.set_fixed_valid_len(320+144)
    assert backend.valid==512+144
    backend.set_fixed_valid_len(320+13)
    assert backend.valid==512+13
    assert backend._pb_expanded_kv_capacity==712


def test_native_rope_gets_original_full_query_and_kv_rows():
    events=[]
    class Ptr:
        def __init__(self,value):self.value=value
        def data_ptr(self):return self.value
    lib=SimpleNamespace(
        expand_selected_rows=lambda *a:events.append(('expand',a)) or 0,
        gather_selected_rows=lambda *a:events.append(('gather',a)) or 0)
    native=SimpleNamespace(qkv_split_rope=lambda *a,**kw:events.append(('rope',a)))
    pl=SimpleNamespace(_selector_lib=lib,vision_seq_enc=320,max_prompt_len=200,encoder_seq_len=520,
        _expanded_qkv=Ptr(11),_expanded_queries=Ptr(12),_selector_mask=Ptr(13),_selector_mapping=Ptr(14),
        _selector_original_rope=SimpleNamespace(ptr=SimpleNamespace(value=15)))
    proxy=ExpandedQKV(native,pl)
    proxy.qkv_split_rope(1,2,3,4,5,520,2048,256,256,256,stream=17)
    assert events[0]==('expand',(1,13,14,11,320,200,2560,17))
    assert events[1]==('rope',(11,15,3,4,5,712,2048,256,256,256))
    assert len(events)==2
    with pytest.raises(ValueError):proxy.qkv_split_rope(1,2,3,4,5,50,2048,256,256,256,stream=17)


def test_full_attention_precedes_compact_output_projection():
    calls=[]
    class Backend:
        def __init__(self,**kwargs):
            self._fixed_shape=True;self._fa2_sites={'encoder':True}
        def encoder_attn(self,layer,seq,stream=0):
            calls.append((layer,seq,stream));return 99
    backend=backend_type(Backend,320)(encoder_seq_max=712)
    pointer=lambda n:SimpleNamespace(data_ptr=lambda:n)
    pool=[]
    backend._pb_pool_pipeline=SimpleNamespace(encoder_seq_len=520,max_prompt_len=200,
        _compact_attention_output=pointer(10),_selector_mask=pointer(11),_selector_mapping=pointer(12),
        _selector_lib=SimpleNamespace(pool_selected_rows=lambda *a:pool.append(a) or 0))
    assert backend.encoder_attn(3,520,stream=5)==10
    assert calls==[(3,712,5)]
    assert pool==[(99,11,12,10,320,200,2048,5)]
