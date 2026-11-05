"""Compact encoder queries/MLPs, original-position full visual keys/values.

All four positions of a merged block remain attention references. Native RoPE
and FlashAttention still do the arithmetic; project kernels only copy rows.
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
            q=self.enc_Q[:seq].unsqueeze(0)
            k=self.enc_K[layer_idx,:self._pb_expanded_kv_capacity].unsqueeze(0)
            v=self.enc_V[layer_idx,:self._pb_expanded_kv_capacity].unsqueeze(0)
            o=self._enc_O[:,:seq].contiguous()
            self._call_fvk_fa2_seqused(q,k,v,o,self._enc_lse,self.enc_seqused,stream=stream)
            return o.data_ptr()
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
        raw=pl._expanded_qkv.data_ptr();full_q=pl._expanded_queries.data_ptr()
        code=lib.expand_selected_rows(source,pl._selector_mask.data_ptr(),pl._selector_mapping.data_ptr(),
            raw,visual,prompt,2560,stream)
        if code:raise RuntimeError('Expand encoder QKV failed:'+str(code))
        self.native.qkv_split_rope(raw,pl._selector_original_rope.ptr.value,full_q,key,value,
            512+prompt,*dimensions,stream=stream)
        code=lib.gather_selected_rows(full_q,pl._selector_mapping.data_ptr(),query,visual,prompt,2048,stream)
        if code:raise RuntimeError('Gather compact encoder queries failed:'+str(code))
