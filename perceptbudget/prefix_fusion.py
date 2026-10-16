"""RTC-inspired two-source fusion in normalized raw32 space (not full RTC).

Only the old queue and latest committed tail participate; no chunk history is
retained. The caller must supply an after queue containing only the unlocked
tail, then unnormalize returned raw rows to action7 and update its own queue.
"""

import math

import numpy as np


def round_to_bf16(values):
    """Return BF16 round-to-nearest-even values stored as NumPy float32.

    Shape is preserved. Nonfinite inputs and finite values that would overflow
    BF16 are rejected. No torch or device operations are involved.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(values, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("BF16 input must be finite float32")
    bits = array.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
               & np.uint32(0xFFFF0000)).view(np.float32)
    if not np.isfinite(rounded).all():
        raise ValueError("BF16 rounding overflow")
    return rounded


def _raw(item):
    raw = np.array(item[4], copy=True)
    if raw.shape != (32,):
        raise ValueError("queue raw32 must have shape (32,)")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError("queue raw32 must be real numeric values")
    if not np.isfinite(raw).all():
        raise ValueError("queue raw32 must be finite")
    return raw.astype(np.float64)


def fuse_prefix_tail(queue_before, queue_after, *, decay=0.1):
    """Return ``{target: {'raw': ndarray32, 'metadata': dict}}`` without mutation.

    Targets are exactly those in queue_after (already mapped/cropped by
    commit_prefix_hold). Items begin with (action7, id, origin, offset, raw32);
    optional trailing audit fields are ignored. The first six normalized raw
    dimensions use weights exp(-decay*abs(old_offset-new_offset)) and 1.
    Only fused arm values are BF16-rounded; latest raw[6:] is copied exactly.
    Latest model rows should already be BF16-representable for whole-row exact
    prefix locking. Single-source rows are copied without rounding.
    """
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("decay must be finite and nonnegative")
    result = {}
    for target, new in queue_after.items():
        new_raw = _raw(new)
        old = queue_before.get(target)
        old_raw = _raw(old) if old is not None else None
        weight = 0.0
        raw = new_raw.copy()
        if old is not None:
            distance = abs(float(old[3]) - float(new[3]))
            if not math.isfinite(distance):
                raise ValueError("offset distance must be finite")
            weight = math.exp(-decay * distance)
            # Divide weights first to avoid overflowing a finite convex mean.
            arm = old_raw[:6] * (weight / (1 + weight))
            arm += new_raw[:6] * (1 / (1 + weight))
            raw[:6] = round_to_bf16(arm)
        metadata = {
            "method": "RTC-inspired two-source", "source_count": 2 if old is not None else 1,
            "arm_dimensions": 6, "gripper_policy": "latest_prediction",
            "old_raw": None if old_raw is None else old_raw.tolist(),
            "new_raw": new_raw.tolist(), "weight_old": weight, "weight_new": 1.0,
            "old_id": None if old is None else old[1], "new_id": new[1],
            "old_origin": None if old is None else old[2], "new_origin": new[2],
            "old_offset": None if old is None else old[3], "new_offset": new[3],
            "decay": decay,
        }
        result[target] = {"raw": raw, "metadata": metadata}
    return result
