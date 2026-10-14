"""Small dependency-free controller for asynchronously predicted actions.

The caller owns observation capture, the worker/executor, and the fixed-rate
environment loop.  ``submit(snapshot, observation_step)`` must return a
Future-like object whose result is an iterable of actions for consecutive
control steps starting at ``observation_step``.
"""

import math
from statistics import median


SUPPORTED_Q = (5, 8, 10, 12, 15)
DEFAULT_HORIZON = 20


class AsyncActionController:
    """Keep a step-aligned action queue while at most one inference is active.

    Call :meth:`bootstrap` once with the first observation.  Thereafter call
    :meth:`request` with each new observation before asking for that step's
    action.  It submits only at q-step boundaries relative to bootstrap; a
    boundary reached while inference is active is recorded and skipped.

    ``pop`` consumes exactly the action for ``control_step`` and advances the
    step even when no action is ready.  Thus the caller can keep its control
    period fixed and handle ``None`` as an explicit wait/hold step.
    """

    def __init__(
        self,
        submit,
        *,
        q=5,
        control_period_s,
        horizon=DEFAULT_HORIZON,
        start_step=0,
    ):
        if not callable(submit):
            raise TypeError("submit must be callable")
        if isinstance(q, bool) or not isinstance(q, int) or q not in SUPPORTED_Q:
            raise ValueError("q must be one of {}".format(SUPPORTED_Q))
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if q > horizon:
            raise ValueError("q cannot exceed the prediction horizon")
        if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
            raise ValueError("start_step must be a non-negative integer")
        if not isinstance(control_period_s, (int, float)) or isinstance(control_period_s, bool):
            raise ValueError("control_period_s must be a positive number")
        if not math.isfinite(control_period_s) or control_period_s <= 0:
            raise ValueError("control_period_s must be a positive finite number")

        self._submit = submit
        self.q = q
        self.horizon = horizon
        self.control_period_s = float(control_period_s)
        self._next_step = start_step
        self._buffer = {}
        self._inflight = None
        self._inflight_step = None
        self._schedule_origin = None
        self._last_trigger_step = None
        self._bootstrapped = False

        self.submitted_requests = 0
        self.failed_requests = 0
        self.skipped_requests = 0
        self.skipped_trigger_steps = []
        self.submitted_actions = 0
        self.dropped_actions = 0
        self.expired_actions = 0
        self.overwritten_actions = 0
        self.underflow_steps = []
        self._buffer_samples = []

    @property
    def control_step(self):
        """Absolute index of the next control step to execute."""
        return self._next_step

    @property
    def inflight(self):
        return self._inflight is not None

    @property
    def action(self):
        """The next action, or ``None`` if this control step is not covered."""
        return self._buffer.get(self._next_step)

    @property
    def buffer(self):
        """Number of consecutive executable actions starting at control_step."""
        step = self._next_step
        while step in self._buffer:
            step += 1
        return step - self._next_step

    @property
    def queued_actions(self):
        """Number of future actions held, including any beyond a gap."""
        return len(self._buffer)

    @property
    def buffer_time_s(self):
        return self.buffer * self.control_period_s

    @property
    def buffer_samples(self):
        """Samples of ``(control_step, consecutive_action_count)`` from pop()."""
        return tuple(self._buffer_samples)

    @property
    def buffer_summary(self):
        """Min/median/max valid buffered actions sampled at control steps."""
        values = [count for _, count in self._buffer_samples]
        if not values:
            return {"count": 0, "min": None, "median": None, "max": None}
        return {
            "count": len(values),
            "min": min(values),
            "median": median(values),
            "max": max(values),
        }

    @property
    def buffer_time_summary_s(self):
        """The buffer summary converted using the fixed control period."""
        summary = self.buffer_summary
        return {
            key: (value * self.control_period_s if key != "count" and value is not None else value)
            for key, value in summary.items()
        }

    def bootstrap(self, snapshot, *, observation_step=None):
        """Submit the first observation immediately; the main agent calls this."""
        if self._bootstrapped:
            raise RuntimeError("bootstrap may only be called once")
        step = self._coerce_current_step(observation_step)
        self._start_request(snapshot, step)
        self._schedule_origin = step
        self._last_trigger_step = step
        self._bootstrapped = True
        return True

    def request(self, snapshot, *, observation_step=None):
        """Poll completion and submit at q-step boundaries; return if submitted.

        The main loop should call this once for each fresh observation.  The
        observation is anchored to the next action's absolute control step.
        Non-boundary calls are cheap no-ops after polling the current Future.
        """
        if not self._bootstrapped:
            raise RuntimeError("call bootstrap before request")
        self.poll()
        step = self._coerce_current_step(observation_step)
        if (step - self._schedule_origin) <= 0 or (step - self._schedule_origin) % self.q:
            return False
        if step == self._last_trigger_step:
            return False

        # A trigger is consumed even when busy: never queue stale observations.
        self._last_trigger_step = step
        if self.inflight:
            self.skipped_requests += 1
            self.skipped_trigger_steps.append(step)
            return False
        self._start_request(snapshot, step)
        return True

    def poll(self):
        """Harvest a completed Future; inference errors propagate to the caller."""
        future = self._inflight
        if future is None or not future.done():
            return False

        start_step = self._inflight_step
        # Release the slot even when result() or result validation raises.
        self._inflight = None
        self._inflight_step = None
        try:
            actions = tuple(future.result())
            if len(actions) > self.horizon:
                raise ValueError(
                    "prediction returned {} actions; horizon is {}".format(
                        len(actions), self.horizon
                    )
                )
            if any(action is None for action in actions):
                raise ValueError("prediction actions must not contain None")
        except BaseException:
            self.failed_requests += 1
            raise

        self.submitted_actions += len(actions)
        current_step = self._next_step
        for offset, action in enumerate(actions):
            action_step = start_step + offset
            if action_step < current_step:
                # The observation became stale while its prediction was running.
                self.dropped_actions += 1
                self.expired_actions += 1
                continue
            if action_step in self._buffer:
                # A newer prediction supersedes the old action at this step.
                self.dropped_actions += 1
                self.overwritten_actions += 1
            self._buffer[action_step] = action
        return True

    def pop(self):
        """Consume this step's action and advance one fixed control period.

        Returns ``None`` and records an underflow when no action covers the
        current step.  It never replays an action from an earlier step.
        """
        self.poll()
        step = self._next_step
        self._buffer_samples.append((step, self.buffer))
        action = self._buffer.pop(step, None)
        if action is None:
            self.underflow_steps.append(step)
        self._next_step += 1
        return action

    def _coerce_current_step(self, observation_step):
        step = self._next_step if observation_step is None else observation_step
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("observation_step must be a non-negative integer")
        if step != self._next_step:
            raise ValueError(
                "observation_step {} does not match next control step {}".format(
                    step, self._next_step
                )
            )
        return step

    def _start_request(self, snapshot, observation_step):
        future = self._submit(snapshot, observation_step)
        if not callable(getattr(future, "done", None)) or not callable(
            getattr(future, "result", None)
        ):
            raise TypeError("submit must return a Future-like object with done() and result()")
        self._inflight = future
        self._inflight_step = observation_step
        self.submitted_requests += 1


class SynchronousActionController(AsyncActionController):
    """Pause control until prediction completes; execute its prefix from step 0.

    For quality measurement only: inference waiting is real elapsed time, not
    simulated robot motion. Replanning still happens at the configured q steps.
    """

    def pop(self):
        if self._inflight is not None:
            try:
                self._inflight.result()
            except BaseException:
                self.poll()  # Harvest failures through the shared bookkeeping.
                raise
        return super().pop()
