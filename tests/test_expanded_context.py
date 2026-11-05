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


def test_native_rope_gets_expanded_rows_and_attention_gets_compact_queries():
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
    assert events[1]==('rope',(11,15,12,4,5,712,2048,256,256,256))
    assert events[2]==('gather',(12,14,3,320,200,2048,17))
    with pytest.raises(ValueError):proxy.qkv_split_rope(1,2,3,4,5,50,2048,256,256,256,stream=17)
