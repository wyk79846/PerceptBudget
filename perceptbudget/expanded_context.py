"""Compact encoder projections/MLPs, original-position full attention.

All four positions of a merged block remain queries and attention references.
Native RoPE and FlashAttention still do the arithmetic; outputs are averaged
back to compact rows before the output projection and MLP.
The action expert sees the original prefix shape, including complete state.
"""


def backend_type(base_type, visual_tokens):
    if visual_tokens not in (320,464):raise ValueError('Unsupported compact query geometry')
    class ExpandedBackend(base_type):
        def __init__(self,*args,**kwargs):
            self._pb_expanded_kv_capacity=kwargs['encoder_seq_max']
            self._pb_compact_visual_tokens=visual_tokens
            super().__init__(*args,**kwargs)

        def set_fixed_valid_len(self,compact_valid):
            return super().set_fixed_valid_len(compact_valid+512-visual_tokens)

        def encoder_attn(self,layer_idx,seq,stream=0):
            if not self._fixed_shape or not self._fa2_sites['encoder']:
                raise ValueError('Expanded context requires fixed BF16 encoder FA2')
            pl=self._pb_pool_pipeline
            if seq!=pl.encoder_seq_len:raise ValueError('Compact encoder query capacity changed')
            full=super().encoder_attn(layer_idx,self._pb_expanded_kv_capacity,stream=stream)
            output=pl._compact_attention_output.data_ptr()
            code=pl._selector_lib.pool_selected_rows(full,pl._selector_mask.data_ptr(),pl._selector_mapping.data_ptr(),
                output,visual_tokens,pl.max_prompt_len,2048,stream)
            if code:raise RuntimeError('Pool complete attention outputs failed:'+str(code))
            return output
    return ExpandedBackend


class ExpandedQKV:
    def __init__(self,native,pipeline):self.native,self.pipeline=native,pipeline
    def __getattr__(self,name):return getattr(self.native,name)

    def qkv_split_rope(self,*args,stream=0):
        pl=self.pipeline
        if len(args)!=10 or tuple(args[6:])!=(2048,256,256,256) or args[5]!=pl.encoder_seq_len:
            raise ValueError('Expanded context intercepts encoder QKV only')
        source,rope,query,key,value,seq,*dimensions=args
        lib=pl._selector_lib;visual=pl.vision_seq_enc;prompt=pl.max_prompt_len
        raw=pl._expanded_qkv.data_ptr()
        code=lib.expand_selected_rows(source,pl._selector_mask.data_ptr(),pl._selector_mapping.data_ptr(),
            raw,visual,prompt,2560,stream)
        if code:raise RuntimeError('Expand encoder QKV failed:'+str(code))
        self.native.qkv_split_rope(raw,pl._selector_original_rope.ptr.value,query,key,value,
            512+prompt,*dimensions,stream=stream)
