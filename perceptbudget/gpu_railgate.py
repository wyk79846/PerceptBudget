"""Read GPU power state; only the separate root lease changes railgate_enable."""
import time
from pathlib import Path

GPU=Path('/sys/devices/platform/bus@0/17000000.gpu')


def state(root=GPU):
    values=dict(monotonic=time.monotonic())
    for name in ('railgate_enable','railgate_delay','is_railgated','gpu_powered_on',
                 'power/runtime_status','power/runtime_suspended_time'):
        try:values[name]=(root/name).read_text().strip()
        except OSError as exc:values[name]=type(exc).__name__
    return values


def confirmed_off(sample):
    return sample.get('is_railgated')=='yes' and sample.get('gpu_powered_on')=='off'


def set_enabled(value, root=GPU):
    if value not in ('0','1'):
        raise ValueError('Only railgate_enable0/1 is supported')
    path=root/'railgate_enable'
    if path.read_text().strip()!=value:
        path.write_text(value)
    if path.read_text().strip()!=value:
        raise RuntimeError('railgate_enable readback differs')
