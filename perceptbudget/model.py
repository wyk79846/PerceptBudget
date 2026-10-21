"""Thin adapter to FlashRT with recorded loading and tokenizer-cache patches."""

import importlib
import os


def load_model(config):
    """Use the existing constructor's chunk_size; inference is unchanged."""
    if config["precision"] != "bfloat16":
        raise ValueError("This M1 adapter implements only the requested BF16 baseline")
    if config.get('native_gemm_policy', 'autotune') not in ('autotune', 'heuristic-first'):
        raise ValueError('Unknown native GEMM policy')
    from perceptbudget.flashrt_import import select_flashrt

    select_flashrt(config["flashrt_root"], config["openpi_root"])
    # Fail before allocating checkpoint copies if the clean checkout isn't built.
    for name in ("flash_rt.flash_rt_kernels", "flash_rt.flash_rt_fa2"):
        importlib.import_module(name)
    # Process-local settings only: no change to the board or shared environment.
    os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
    os.environ["FVK_RTX_FA2"] = "1"
    os.environ.pop("FVK_RTX_FA2_SITES", None)
    for key in ("FVK_PI05_RTX_FORCE_INT8", "FVK_PI05_RTX_INT8_ENCODER_ONLY",
                "FVK_PI05_RTX_INT8_VISION"):
        os.environ[key] = "0"
    from flash_rt.api import VLAModel
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    if config.get('state_prompt_capacity') is not None:
        from perceptbudget.compact_state import frontend_type
        Pi05TorchFrontendRtx=frontend_type(Pi05TorchFrontendRtx,config['state_prompt_capacity'])

    pipe = Pi05TorchFrontendRtx(
        config["checkpoint_path"], num_views=2, chunk_size=config["H"],
        num_steps=config["N"], use_fp8=False, hardware="rtx_sm87",
        vision_pool_factor=1, vision_num_layers=27, cache_frames=1,
        state_prompt_mode="fixed",
    )
    from perceptbudget.gemm_policy import apply_policy
    apply_policy(pipe, config.get('native_gemm_policy', 'autotune'))
    if not pipe._force_bf16 or any((pipe._fp8_weights, pipe._int8_weights,
                                  pipe._use_int8_encoder, pipe._use_int8_vision)):
        raise RuntimeError("Requested full BF16 path did not take effect")
    if config.get('checkpoint_profile')=='lerobot_h50_quantiles_v044':
        # Mixed F32/BF16 source conversion can leave freed host arenas resident.
        # Release only already-unused allocator pages, never weights or GPU data.
        import gc
        import ctypes
        gc.collect()
        trimmed=ctypes.CDLL(None).malloc_trim(0)
        print('h50_host_allocator_trim='+str(trimmed),flush=True)
    return VLAModel(pipe, "torch")


def validate_actions(actions, horizon):
    import numpy as np

    actions = np.asarray(actions)
    if actions.shape != (horizon, 7) or not np.isfinite(actions).all():
        raise RuntimeError("Invalid action output: shape=" + str(actions.shape))
    return actions.copy()
