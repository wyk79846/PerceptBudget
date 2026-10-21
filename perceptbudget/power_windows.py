"""Matched idle/busy power window measurement."""


def block_schedule():
    return ['idle', 'busy'] * 3 + ['idle']

def measurement_summary(readings, blocks):
    import numpy as np
    from perceptbudget.energy_metrics import _integral
    measured = []
    for block in blocks:
        # Keep transitions visible in raw data, use middle of long windows for averages.
        start, stop = block['start_monotonic'] + 1., block['stop_monotonic'] - .5
        if stop <= start:
            raise ValueError('Power windows too short')
        report = _integral(readings, [(start, stop)])
        measured.append(dict(block, measured_start=start, measured_stop=stop, power=report))
    grouped = {}
    for mode in ('idle', 'busy'):
        rows = [b['power'] for b in measured if b['mode'] == mode]
        duration = sum(b['duration_seconds'] for b in rows)
        complete = all(b['full_coverage'] for b in rows)
        energy = sum(b['total_energy_j'] for b in rows) if complete else None
        grouped[mode] = dict(windows=len(rows), duration_seconds=duration,
                             mean_power_w=energy/duration if energy is not None else None,
                             energy_j=energy, full_coverage=complete)
    on, idle = grouped['busy']['mean_power_w'], grouped['idle']['mean_power_w']
    valid = [r for r in readings if r.power_mw is not None]
    changes = [b.timestamp for a, b in zip(valid, valid[1:]) if a.power_mw != b.power_mw]
    intervals = np.diff(changes).tolist()
    return dict(blocks=measured, grouped=grouped, P_on_w=on, P_idle_w=idle,
                r=idle/on if idle is not None and on else None,
                samples=len(readings), missing_samples=sum(r.power_mw is None for r in readings),
                observed_change_interval_median_s=float(np.median(intervals)) if intervals else None,
                change_interval_note='Observed value changes, not an exact hardware update period')
