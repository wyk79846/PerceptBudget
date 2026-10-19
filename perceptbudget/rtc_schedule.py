"""Algorithm 1 delay queue; controller steps, not wall-time conversions."""
from collections import deque


class DelayQueue:
    def __init__(self, initial_delay, size=10):
        if size < 1 or initial_delay < 0:
            raise ValueError('invalid delay queue')
        self.values = deque([int(initial_delay)], maxlen=size)

    @property
    def estimate(self):
        return max(self.values)

    def observe(self, delay):
        if delay < 0:
            raise ValueError('negative delay')
        self.values.append(int(delay))
