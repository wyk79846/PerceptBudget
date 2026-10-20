"""Small LIBERO-to-FlashRT input adapter; imports image/numeric libraries lazily."""

import json
from pathlib import Path


def load_libero_norm_stats(path):
    """Read the checkpoint's LIBERO ``norm_stats.json`` payload."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("无法读取 LIBERO norm_stats：{} ({})".format(path, exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("norm_stats 顶层必须是对象")
    stats = payload.get("norm_stats", payload)
    if not isinstance(stats, dict):
        raise ValueError("norm_stats 字段必须是对象")
    return stats


def raw_libero_state(observation):
    """Return LIBERO's 8 values: EEF xyz, quaternion axis-angle, gripper qpos."""
    import numpy as np

    try:
        position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).reshape(-1)
        gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("LIBERO observation 缺少有效的末端位姿或夹爪 state") from exc
    if position.shape != (3,) or quaternion.shape != (4,) or gripper.shape != (2,):
        raise ValueError(
            "LIBERO state 维度错误：pos={}, quat={}, gripper={}".format(
                position.shape, quaternion.shape, gripper.shape
            )
        )
    if not (np.isfinite(position).all() and np.isfinite(quaternion).all()
            and np.isfinite(gripper).all()):
        raise ValueError("LIBERO raw state 含非有限值")

    # Match robosuite's quaternion convention and quat2axisangle formula (x,y,z,w).
    w = float(np.clip(quaternion[3], -1.0, 1.0))
    denominator = np.sqrt(max(0.0, 1.0 - w * w))
    if denominator < 1e-8:
        axis_angle = np.zeros(3, dtype=np.float32)
    else:
        angle = 2.0 * np.arccos(w)
        axis_angle = (quaternion[:3] * (angle / denominator)).astype(np.float32)
    state = np.concatenate((position, axis_angle, gripper)).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO raw state 必须是 8 维有限向量")
    return state


def quantile_normalize_state(raw_state, norm_stats):
    """Apply OpenPI's q01/q99 formula without padding or truncating dimensions."""
    import numpy as np

    state = np.asarray(raw_state, dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("待归一化 state 必须是 8 维有限向量")
    try:
        state_stats = norm_stats["state"]
        q01 = np.asarray(state_stats["q01"], dtype=np.float32)
        q99 = np.asarray(state_stats["q99"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("LIBERO norm_stats 必须含 state.q01 和 state.q99") from exc
    if q01.shape != (8,) or q99.shape != (8,):
        raise ValueError(
            "checkpoint state q01/q99 必须恰为 8 维，不自动补维或截断"
        )
    if not (np.isfinite(q01).all() and np.isfinite(q99).all()):
        raise ValueError("checkpoint state q01/q99 含非有限值")

    # Same quantile normalization used by OpenPI: (x-q01)/(q99-q01+1e-6)*2-1.
    normalized = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    if not np.isfinite(normalized).all():
        raise ValueError("归一化后的 LIBERO state 含非有限值")
    return normalized.astype(np.float32, copy=False)


def _rotate_resize(image):
    import numpy as np
    import cv2

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(
            "LIBERO camera image 必须是 HxWx3 uint8，实际为 {} {}".format(
                array.shape, array.dtype
            )
        )
    rotated = np.ascontiguousarray(array[::-1, ::-1])
    height, width = rotated.shape[:2]
    scale = min(224 / height, 224 / width)
    resized_height = int(height * scale)
    resized_width = int(width * scale)
    resized = cv2.resize(
        rotated,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_top = (224 - resized_height) // 2
    pad_left = (224 - resized_width) // 2
    result = np.zeros((224, 224, 3), dtype=array.dtype)
    result[
        pad_top:pad_top + resized_height,
        pad_left:pad_left + resized_width,
    ] = resized
    return result


def prepare_libero_observation(observation, norm_stats, *, state_enabled):
    """Build FlashRT's two-view input and retain raw proprioception for tracing.

    When enabled, the normalized state is passed to FlashRT's ``state`` argument.
    The configured Pi0.5 frontend turns that value into extra prompt tokens; it is
    not OpenPI pi05_libero's official default state path (discrete_state_input is
    false and pi05 has no continuous state projection).
    """
    raw_state = raw_libero_state(observation)
    if norm_stats.get('_profile')=='lerobot_h50_quantiles_v044':
        from perceptbudget.h50_checkpoint import image_input,normalize_state
        result={'images':[image_input(observation[k]) for k in ('agentview_image','robot0_eye_in_hand_image')],
                'raw_state':raw_state}
        if state_enabled:
            result['state']=normalize_state(raw_state,norm_stats)
        return result
    try:
        images = [
            _rotate_resize(observation["agentview_image"]),
            _rotate_resize(observation["robot0_eye_in_hand_image"]),
        ]
    except KeyError as exc:
        raise ValueError("LIBERO observation 缺少双相机图像：" + str(exc)) from exc

    prepared = {"images": images, "raw_state": raw_state}
    if state_enabled:
        prepared["state"] = quantile_normalize_state(raw_state, norm_stats)
    return prepared
