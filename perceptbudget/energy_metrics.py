"""Energy integration and interval helpers; no historical experiment CLI."""
import math
from perceptbudget.power import PowerReading, _integrate_report

MAX_GAP_SECONDS = 0.3

def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None

def _merge(intervals, start, stop):
    clipped = sorted((max(start, a), min(stop, b)) for a, b in intervals if b > a)
    merged = []
    for a, b in clipped:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged

def _integral(readings, intervals):
    duration = sum(b - a for a, b in intervals)
    if duration <= 0:
        return {"duration_seconds": 0.0, "coverage_seconds": 0.0,
                "coverage_fraction": None, "covered_energy_j": 0.0,
                "total_energy_j": 0.0, "mean_power_w": None,
                "full_coverage": True, "missing_count": 0}
    reports = [_integrate_report(readings, a, b, MAX_GAP_SECONDS)
               for a, b in intervals]
    covered = sum(report.coverage_seconds for report in reports)
    partial_mj = sum(report.energy_mj or 0.0 for report in reports)
    full = all(report.total_energy_mj is not None for report in reports)
    total_mj = partial_mj if full else None
    return {
        "duration_seconds": duration,
        "coverage_seconds": covered,
        "coverage_fraction": covered / duration,
        "covered_energy_j": partial_mj / 1000.0 if covered else None,
        "total_energy_j": total_mj / 1000.0 if total_mj is not None else None,
        "mean_power_w": total_mj / duration / 1000.0
        if total_mj is not None else None,
        "full_coverage": full,
        "missing_count": sum(report.missing_count for report in reports),
    }
