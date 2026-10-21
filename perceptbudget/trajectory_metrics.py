"""Trajectory resampling shared by the H50 analysis."""
import math

def distance(a, b):
    return math.dist(a, b)

def resample(points, count=101):
    # Normalized accumulated path length: descriptive shape comparison only.
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + distance(a, b))
    if not cumulative[-1]:
        return [points[0]] * count
    result = []
    index = 0
    for i in range(count):
        target = cumulative[-1] * i / (count - 1)
        while index + 1 < len(points) - 1 and cumulative[index + 1] < target:
            index += 1
        span = cumulative[index + 1] - cumulative[index]
        alpha = (target - cumulative[index]) / span if span else 0.0
        result.append([a + alpha * (b - a)
                       for a, b in zip(points[index], points[index + 1])])
    return result
