"""CPU-only action blending from React When You Need To: Event-Triggered Asynchronous Inference for VLA Policies, Eq. 8."""

import math


def _positive_finite_beta(beta):
    if isinstance(beta, (bool, str, bytes)):
        raise ValueError("beta must be a positive finite number")
    try:
        beta = float(beta)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("beta must be a positive finite number") from None
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite number")
    return beta


def _copy_actions(actions, name):
    if isinstance(actions, (str, bytes)):
        raise ValueError(f"{name} must be a two-dimensional action sequence")
    try:
        rows = list(actions)
    except TypeError:
        raise ValueError(f"{name} must be a two-dimensional action sequence") from None

    copied = []
    width = None
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ValueError(f"{name} must contain finite numeric vectors")
        try:
            values = list(row)
        except TypeError:
            raise ValueError(f"{name} must contain finite numeric vectors") from None
        if not values:
            raise ValueError(f"{name} vectors must have a nonzero width")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError(f"{name} vectors must have a consistent width")

        vector = []
        for value in values:
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{name} values must be finite numbers")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name} values must be finite numbers") from None
            if not math.isfinite(value):
                raise ValueError(f"{name} values must be finite numbers")
            vector.append(value)
        copied.append(vector)
    return copied, width


def _exprel(value):
    """Return expm1(value) / value, including its continuous value at zero."""
    if value == 0.0:
        return 1.0
    return math.expm1(value) / value


def blend_actions(old, new, beta=1.2):
    """Blend equally shaped action chunks and return ``(merged, old_weights)``.

    For N >= 2, each step uses the paper's Eq. 8 old-action weight
    ``(exp(-beta*k/(N-1)) - exp(-beta)) / (1 - exp(-beta))``. The same weight
    is applied to every action dimension, including the gripper; values are
    not binarized. Since the paper's schedule is undefined for N == 1, that
    case explicitly returns a copy of ``new`` with weight zero. Two empty
    chunks are accepted and return ``([], [])``.
    """
    beta = _positive_finite_beta(beta)
    old_actions, old_width = _copy_actions(old, "old")
    new_actions, new_width = _copy_actions(new, "new")

    if len(old_actions) != len(new_actions):
        raise ValueError("old and new action chunks must have equal lengths")
    if old_width is not None and new_width is not None and old_width != new_width:
        raise ValueError("old and new action vectors must have equal widths")

    count = len(old_actions)
    if count == 0:
        return [], []
    if count == 1:
        return new_actions, [0.0]

    weights = []
    merged = []
    denominator = _exprel(-beta)
    last = count - 1
    for index, (old_vector, new_vector) in enumerate(zip(old_actions, new_actions)):
        if index == 0:
            weight = 1.0
        elif index == last:
            weight = 0.0
        else:
            fraction = index / last
            remaining_fraction = 1.0 - fraction
            # Algebraically equivalent to Eq. 8, with expm1 and the exprel
            # ratio avoiding cancellation for small beta and stable decay
            # when beta is large.
            weight = (
                math.exp(-beta * fraction)
                * remaining_fraction
                * _exprel(-beta * remaining_fraction)
                / denominator
            )
        weights.append(weight)
        merged.append([
            weight * old_value + (1.0 - weight) * new_value
            for old_value, new_value in zip(old_vector, new_vector)
        ])
    return merged, weights
