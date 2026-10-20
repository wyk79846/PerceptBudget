"""Characterize pinned upstream behavior, not a replacement implementation."""
import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec('lerobot') is None,reason='separate LeRobot comparison environment')
def test_pinned_lerobot_correction_omits_velocity_input_derivative():
    import torch
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.configs.types import RTCAttentionSchedule
    rtc=RTCProcessor(RTCConfig(enabled=True,debug=True,max_guidance_weight=5.,
        execution_horizon=2,prefix_attention_schedule=RTCAttentionSchedule.EXP))
    x=torch.arange(1,7,dtype=torch.float32).reshape(3,2)
    rtc.denoise_step(x,torch.zeros(2,2),1,.5,lambda z:.1*z.square(),execution_horizon=2)
    record=rtc.get_all_debug_steps()[0]
    expected=record.err*(1-.1*x)  # derivative of x - 0.5 * 0.1*x^2
    torch.testing.assert_close(record.correction,record.err,rtol=0,atol=0)
    assert (record.correction-expected).abs().max()>.05
