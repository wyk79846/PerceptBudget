"""Original OpenPI suffix modules, isolated from installed Transformers.

The source methods are executed unchanged: no rewritten transformer, folded
projection, custom backward or weight gradients. Prefix KV is supplied by the
caller; this is not an independent validation of the FlashRT encoder.
"""
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace


def _source_class(path, name, methods, scope):
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    original.bases = [ast.Attribute(value=ast.Name(id='nn', ctx=ast.Load()), attr='Module', ctx=ast.Load())]
    original.decorator_list = []
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (
        'get_safe_dtype', 'create_sinusoidal_pos_embedding', 'make_att_2d_masks')]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *functions, original], type_ignores=[]))
    exec(compile(code, str(path), 'exec'), scope)
    return scope[name]


def load_original_suffix(config, horizon, device='cuda'):
    """Read only suffix weights; avoid allocating a second VLM."""
    import torch
    from safetensors import safe_open
    from transformers import GemmaConfig

    root = Path(config['openpi_root'])/'src/openpi/models_pytorch'
    gemma_file = root/'transformers_replace/models/gemma/modeling_gemma.py'
    spec = importlib.util.spec_from_file_location('transformers.models.gemma.pb_original_suffix', gemma_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = GemmaConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=18,
                      num_attention_heads=8, num_key_value_heads=1, head_dim=256,
                      hidden_activation='gelu_pytorch_tanh', vocab_size=1,
                      use_adarms=True, adarms_cond_dim=1024)
    cfg._attn_implementation = 'eager'
    with torch.device('meta'):
        gemma = module.GemmaModel(cfg)
    gemma.embed_tokens = None
    gemma.rotary_emb = module.GemmaRotaryEmbedding(cfg, device=device)
    scope = dict(torch=torch, Tensor=torch.Tensor, nn=torch.nn, F=torch.nn.functional, math=math)
    bridge_type = _source_class(root/'gemma_pytorch.py', 'PaliGemmaWithExpertModel', {'forward'}, scope)
    suffix_type = _source_class(root/'pi0_pytorch.py', 'PI0Pytorch',
        {'embed_suffix', 'denoise_step', '_apply_checkpoint', '_prepare_attention_masks_4d'}, scope)
    suffix = suffix_type()
    suffix.pi05 = True
    suffix.config = SimpleNamespace(action_horizon=horizon)
    suffix.gradient_checkpointing_enabled = False
    bridge = bridge_type()
    bridge.gemma_expert = torch.nn.Module()
    bridge.gemma_expert.model = gemma
    suffix.paligemma_with_expert = bridge
    with safe_open(config['model_path'], framework='pt', device='cpu') as checkpoint:
        prefix = 'paligemma_with_expert.gemma_expert.model.'
        weights = {k[len(prefix):]: checkpoint.get_tensor(k).to(device=device,
            dtype=torch.float32 if 'layernorm' in k or '.model.norm.' in k else torch.bfloat16)
            for k in checkpoint.keys() if k.startswith(prefix) and 'embed_tokens' not in k}
        gemma.load_state_dict(weights, assign=True, strict=True)
        del weights
        for name in ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out'):
            weight, bias = checkpoint.get_tensor(name+'.weight'), checkpoint.get_tensor(name+'.bias')
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0], device='meta')
            linear.load_state_dict(dict(weight=weight.to(device=device, dtype=torch.float32),
                                        bias=bias.to(device=device, dtype=torch.float32)), assign=True)
            setattr(suffix, name, linear)
    suffix.eval().requires_grad_(False)
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (gemma_file, root/'gemma_pytorch.py', root/'pi0_pytorch.py')}
    return suffix, provenance


def original_prefix(keys, values, valid):
    """FlashRT adjacent-pair K -> original split-half RoPE coordinates."""
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 18 or keys.shape[2:] != (1, 256):
        raise ValueError('expected 18 x prefix x 1 x 256 KV')
    if not 0 < valid <= keys.shape[1]:
        raise ValueError('invalid prefix length')
    k = keys[:, :valid].detach().reshape(18, valid, 1, 128, 2)
    k = k.transpose(-1, -2).flatten(-2).permute(0, 2, 1, 3).unsqueeze(1).contiguous()
    v = values[:, :valid].detach().permute(0, 2, 1, 3).unsqueeze(1).contiguous()
    return k, v


class OriginalOpenPiExpert:
    def __init__(self, config, horizon, num_steps, device='cuda'):
        self.suffix, self.provenance = load_original_suffix(config, horizon, device)
        self.horizon, self.num_steps, self.device = horizon, num_steps, device
        self.keys = self.values = None

    def refresh_context(self, keys, values, valid):
        import torch
        self.keys, self.values = original_prefix(keys, values, valid)
        self.prefix_mask = torch.ones(1, valid, dtype=torch.bool, device=self.device)

    def velocity(self, x, step):
        import torch
        from transformers.cache_utils import DynamicCache
        if self.keys is None or x.shape != (self.horizon, 32) or not 0 <= step < self.num_steps:
            raise ValueError('invalid original expert context/input/step')
        cache = DynamicCache()
        for i in range(18):
            cache.update(self.keys[i], self.values[i], i)
        t = torch.tensor([1-step/self.num_steps], device=x.device, dtype=torch.float32)
        return self.suffix.denoise_step(None, self.prefix_mask, cache, x.unsqueeze(0), t).squeeze(0)


def sample_original_rtc(expert, noise, previous, delay, executed, beta=5.):
    """Author's training-free Euler/VJP algorithm in the original flow time.

    OpenPI v decreases time 1->0; author's velocity is its negative. The old
    prefix is a soft guidance target, never hard-overwritten after a step.
    """
    import torch
    horizon, steps = expert.horizon, expert.num_steps
    if not 0 <= executed <= horizon or not 0 <= delay <= horizon-executed:
        raise ValueError('delay exceeds available old-action overlap')
    if previous is not None and previous.shape != (horizon-executed, 32):
        raise ValueError('old action target length mismatch')
    x = noise.detach().float()
    if x.shape != (horizon, 32) or steps < 1 or not math.isfinite(beta) or beta <= 0:
        raise ValueError('invalid noise/steps/beta')
    if previous is None:
        with torch.no_grad():
            for step in range(steps):
                x = x-expert.velocity(x, step)/steps
        return x, []
    target = torch.zeros_like(x)
    target[:len(previous)] = previous.detach().float()
    i = torch.arange(horizon, device=x.device, dtype=torch.float32)
    end = horizon-executed
    weights = ((delay-1-i)/(end-delay+1)+1).clamp(0, 1)
    weights = torch.where(i >= end, 0, weights*torch.expm1(weights)/(math.e-1))[:, None]
    records = []
    for step in range(steps):
        tau = step/steps
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            velocity = -expert.velocity(x, step)
            clean = x+(1-tau)*velocity
            residual = ((target-clean)*weights).detach()
            correction, = torch.autograd.grad(clean, x, grad_outputs=residual)
            gain = beta if step == 0 else min(beta, (tau*tau+(1-tau)**2)/(tau*(1-tau)))
            x = (x+(1/steps)*(velocity+gain*correction)).detach()
        if not torch.isfinite(x).all():
            raise RuntimeError('nonfinite original RTC sample')
        records.append(float(correction.norm()))
    return x, records
