"""CPU-only timing estimates; a feasible choice is not an action-quality guarantee."""

import math
import operator
from collections import deque


def _positive_finite(value):
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _nonnegative_integer(value):
    if isinstance(value, bool):
        return None
    try:
        value = operator.index(value)
    except TypeError:
        return None
    return value if value >= 0 else None


class PrefixTiming:
    """Select a configured prefix using recent inference and control timings."""

    _HISTORY_SIZE = 16

    def __init__(self, latency_seconds, nominal_period=0.05,
                 lengths=(5, 6, 7, 8), margin=0.005):
        nominal_period = _positive_finite(nominal_period)
        if nominal_period is None:
            raise ValueError("nominal_period must be a positive finite number")
        if isinstance(margin, (bool, str, bytes)):
            raise ValueError("margin must be a finite nonnegative number")
        try:
            margin = float(margin)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("margin must be a finite nonnegative number") from None
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("margin must be a finite nonnegative number")

        try:
            lengths = tuple(lengths)
            if any(isinstance(length, bool) for length in lengths):
                raise ValueError
            lengths = tuple(operator.index(length) for length in lengths)
        except (TypeError, ValueError):
            raise ValueError("lengths must contain positive integers") from None
        if not lengths or any(isinstance(length, bool) or length <= 0
                              for length in lengths):
            raise ValueError("lengths must contain positive integers")

        self.nominal_period = nominal_period
        self.lengths = tuple(sorted(set(lengths)))
        self.margin = margin
        self._latencies = deque(maxlen=self._HISTORY_SIZE)
        self._ticks = deque(maxlen=self._HISTORY_SIZE)
        self.observe_latency(latency_seconds)

    def observe_latency(self, seconds):
        """Record a positive finite inference latency; ignore invalid readings."""
        seconds = _positive_finite(seconds)
        if seconds is None:
            return False
        self._latencies.append(seconds)
        return True

    def observe_tick(self, seconds):
        """Record a positive finite tick, floored at the nominal period."""
        seconds = _positive_finite(seconds)
        if seconds is None:
            return False
        self._ticks.append(max(seconds, self.nominal_period))
        return True

    def choose(self, q, buffer_actions):
        """Return a JSON-compatible decision and its timing estimates."""
        latency_estimate = None
        required_steps = None
        reason = None
        if self._latencies:
            ordered = sorted(self._latencies)
            rank = (4 * len(ordered) + 4) // 5
            latency_estimate = ordered[rank - 1]

        # Use the faster end: slower past steps must not imply guaranteed slack.
        ordered_ticks = sorted(self._ticks)
        tick_estimate = (ordered_ticks[(len(ordered_ticks) + 4) // 5 - 1]
                         if ordered_ticks else self.nominal_period)
        if latency_estimate is None:
            reason = "missing_latency_samples"
        else:
            required_steps = math.ceil(
                (latency_estimate + self.margin) / tick_estimate)

        q_limit = _nonnegative_integer(q)
        buffer_limit = _nonnegative_integer(buffer_actions)
        selected = None
        if reason is None and q_limit is not None and buffer_limit is not None:
            selected = next(
                (length for length in self.lengths
                 if length >= required_steps
                 and length <= q_limit
                 and length <= buffer_limit),
                None,
            )
        if selected is None and reason is None:
            reason = "no_candidate_fits"

        return {
            "selected": selected,
            "infeasible": selected is None,
            "reason": reason,
            "required_steps": required_steps,
            "latency_estimate_s": latency_estimate,
            "tick_estimate_s": tick_estimate,
            "tick_estimator": "recent16_nearest_rank_p20",
            "margin_s": self.margin,
            "latency_sample_count": len(self._latencies),
            "tick_sample_count": len(self._ticks),
        }
