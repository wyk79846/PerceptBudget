"""Shared synchronized model timing."""
import statistics
import time

def timed(torch, fn, count=5):
    values = []
    for _ in range(count):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    return {'trials_ms': values, 'median_ms': statistics.median(values)}
