"""Read frozen prompt rows from the checkpoint instead of pinning its vocabulary.

The public lookup still returns CUDA BF16 rows. FlashRT's tokenizer, embedding
scale, context kernels and RTC remain unchanged; only table storage changes.
"""
import gc
from pathlib import Path


class MappedCudaEmbedding:
    def __init__(self, table, device, dtype, record, *, owner=None):
        import torch
        if table.device.type!='cpu' or table.ndim!=2 or table.requires_grad:
            raise ValueError('Require a frozen CPU checkpoint matrix')
        self.table,self.device,self.dtype=table,torch.device(device),dtype
        self.shape=table.shape
        self.record,self.owner=record,owner

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        import torch
        from torch.nn import functional as F
        if func is not F.embedding:return NotImplemented
        ids,weight=args[:2];options=kwargs or {}
        if (options.get('padding_idx') is not None or options.get('max_norm') is not None
                or options.get('scale_grad_by_freq',False) or options.get('sparse',False)):
            raise ValueError('Mapped prompt lookup supports frozen plain embedding only')
        with torch.no_grad():
            rows=F.embedding(ids.detach().to(device='cpu'),weight.table)
            result=rows.to(device=weight.device,dtype=weight.dtype)
        weight.record['lookup_calls']+=1
        weight.record['lookup_rows']+=ids.numel()
        return result


def install_mapped_prompt_embeddings(frontends, checkpoint_path, *, checkpoint_prefix='', verify_all=False):
    import torch
    from safetensors import safe_open
    if not frontends:raise ValueError('Require at least one frontend')
    old=frontends[0].embedding_weight
    if isinstance(old,MappedCudaEmbedding):return old.record
    if old.device.type!='cuda' or old.dtype!=torch.bfloat16 or old.requires_grad:
        raise ValueError('Mapped prompt storage requires frozen CUDA BF16 source')
    if any(p.embedding_weight is not old or p._ckpt_bf16['embedding_weight'] is not old for p in frontends):
        raise ValueError('Frontends must share the same original embedding tensor')
    key=checkpoint_prefix+'paligemma_with_expert.paligemma.lm_head.weight'
    owner=safe_open(str(Path(checkpoint_path)),framework='pt',device='cpu')
    table=owner.get_tensor(key)  # File-backed view, not a full anonymous CPU copy.
    if table.shape!=old.shape:raise ValueError('Checkpoint embedding dimensions differ')
    record=dict(storage='checkpoint_mmap_cpu_rows_to_cuda',source_key=key,
        shape=list(old.shape),source_dtype=str(table.dtype),output_dtype=str(old.dtype),
        output_device=str(old.device),released_gpu_table_mib=old.numel()*old.element_size()/2**20,
        verified_all_rows=False,lookup_calls=0,lookup_rows=0)
    if verify_all:
        for start in range(0,len(table),4096):
            cpu=table[start:start+4096].to(dtype=old.dtype).view(torch.uint16)
            native=old[start:start+4096].detach().cpu().view(torch.uint16)
            if not torch.equal(cpu,native):raise RuntimeError('Mapped BF16 table differs from native conversion')
        del cpu,native
        record['verified_all_rows']=True
    mapped=MappedCudaEmbedding(table,old.device,old.dtype,record,owner=owner)
    torch.cuda.synchronize()
    for p in frontends:
        p.embedding_weight=p._ckpt_bf16['embedding_weight']=mapped
        p._pb_mapped_embedding_record=record
    del old
    gc.collect();torch.cuda.empty_cache()
    return record
