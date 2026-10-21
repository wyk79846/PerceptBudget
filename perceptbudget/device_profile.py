"""M2 discovery and synthetic sensor diagnostics, not a model energy benchmark."""
import ast
import hashlib
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from perceptbudget.device_control import inventory
from perceptbudget.power import PowerMonitor


def inspect_fast_interface(flashrt_root):
    root = Path(flashrt_root)
    paths = ['flash_rt/frontends/torch/pi05_rtx.py', 'flash_rt/models/pi05/pipeline_rtx.py']
    evidence = []
    view_argument_found = False
    for name in paths:
        path = root / name
        source = path.read_text()
        tree = ast.parse(source)
        if 'frontends/' in name:
            view_argument_found = any(isinstance(n, ast.FunctionDef) and n.name == '__init__'
                and 'num_views' in [a.arg for a in n.args.args + n.args.kwonlyargs]
                for n in ast.walk(tree))
        evidence.append(dict(path=str(path), sha256=hashlib.sha256(source.encode()).hexdigest(),
            relevant_lines=[dict(line=i, source=line.strip()) for i, line in enumerate(source.splitlines(), 1)
                            if any(key in line for key in ('num_views:', 'self.num_views =',
                                'self.vision_seq =', 'self.vision_seq_enc =',
                                'self.encoder_seq_len =', 'img_list[:self.num_views]'))]))
    return dict(scope='source inspection only; no live Fast speed/quality validation',
                project_loader_current_views=2, upstream_num_views_interface=view_argument_found,
                expected_visual_tokens=dict(full=512, single_view=256),
                expected_encoder_capacity_with_200_text_tokens=dict(full=712, single_view=456),
                flashrt_modified=False, source=evidence)


def sensor_steps(*, block_seconds=2., pairs=3, poll_interval=.02):
    """Matched synthetic GPU busy/idle steps to inspect sensor response only.

    Does not change frequencies, load a checkpoint, run LIBERO, or assert railgate.
    GPU lock must be held by the calling process.
    """
    if block_seconds < 1 or pairs < 1:
        raise ValueError('sensor windows require >=1 second and >=1 pair')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(memory['MemAvailable'].split()[0]) < 768 * 1024:
        raise RuntimeError('Available RAM below 768 MiB; sensor load not started')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; sensor load steps not started')
    torch.manual_seed(7)
    a = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    monitor = PowerMonitor(poll_interval=poll_interval)
    blocks = []
    monitor.start()
    try:
        for mode in ['idle', 'busy'] * pairs + ['idle']:
            start = time.monotonic()
            iterations = 0
            while time.monotonic() - start < block_seconds:
                if mode == 'busy':
                    torch.mm(a, b, out=out)
                    torch.cuda.synchronize()
                    iterations += 1
                else:
                    time.sleep(min(.01, max(0, block_seconds - (time.monotonic()-start))))
            blocks.append(dict(mode=mode, start_monotonic=start,
                               stop_monotonic=time.monotonic(), iterations=iterations))
    finally:
        torch.cuda.synchronize()
        monitor.stop()
    readings = monitor.readings
    valid = [r for r in readings if r.power_mw is not None]
    changes = [r.timestamp for a, r in zip(valid, valid[1:]) if a.power_mw != r.power_mw]
    intervals = [b-a for a, b in zip(changes, changes[1:])]
    windows = []
    for block in blocks:
        middle = [r.power_mw/1000 for r in valid
                  if block['start_monotonic']+.5 <= r.timestamp < block['stop_monotonic']-.2]
        windows.append(dict(block, middle_mean_power_w=statistics.mean(middle) if middle else None))
    return dict(scope='synthetic matmul sensor-response diagnostic, not Full/Fast/Continue service power',
                power_boundary='INA3221 VDD_IN module input', poll_interval_seconds=poll_interval,
                blocks=windows, samples=[asdict(r) for r in readings],
                missing_samples=len(readings)-len(valid), observed_value_changes=len(changes),
                median_observed_change_interval_seconds=statistics.median(intervals) if intervals else None,
                warning='Observed value changes do not establish hardware update period or response latency',
                hardware_writes=False)


def discovery(config):
    return dict(status='discovery_completed', device=inventory(),
                fast_interface=inspect_fast_interface(config['flashrt_root']),
                baseline_id=config['baseline_id'], full_service_measurements_started=False,
                clock_settings_changed=False)
