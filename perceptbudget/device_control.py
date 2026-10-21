"""Read-only device discovery for M2. No clock or power-mode writes."""
import os
import platform
import subprocess
import time
from pathlib import Path


def read_node(path):
    path = Path(path)
    try:
        value = path.read_text().replace('\x00', '').strip()
        return dict(path=str(path), value=value, error=None,
                    writable=os.access(path, os.W_OK))
    except (OSError, UnicodeError) as exc:
        return dict(path=str(path), value=None, error=str(exc), writable=False)


def inventory(sys_root=Path('/sys')):
    sys_root = Path(sys_root)
    start = time.monotonic()
    domains = {}
    for path in sorted((sys_root / 'class/devfreq').glob('*')):
        if 'gpu' in path.name:
            domains['gpu'] = {key: read_node(path / key) for key in (
                'available_frequencies', 'cur_freq', 'min_freq', 'max_freq', 'governor')}
    for path in sorted((sys_root / 'devices/system/cpu/cpufreq').glob('policy*')):
        domains['cpu_' + path.name] = {key: read_node(path / key) for key in (
            'related_cpus', 'scaling_available_frequencies', 'scaling_cur_freq',
            'scaling_min_freq', 'scaling_max_freq', 'scaling_governor')}
    emc = sys_root / 'kernel/debug/bpmp/debug/clk/emc'
    domains['emc'] = {key: read_node(emc / key) for key in (
        'rate', 'min_rate', 'max_rate', 'possible_rates', 'mrq_rate_locked')}
    gpu_power = sys_root / 'devices/platform/bus@0/17000000.gpu/power'
    runtime = {key: read_node(gpu_power / key) for key in (
        'control', 'runtime_status', 'runtime_active_time', 'runtime_suspended_time')}
    thermal = []
    for path in sorted((sys_root / 'class/thermal').glob('thermal_zone*')):
        thermal.append(dict(kind=read_node(path / 'type'), temperature=read_node(path / 'temp')))
    try:
        mode = subprocess.run(['/usr/sbin/nvpmodel', '-q'], text=True,
                              capture_output=True, check=False, timeout=10)
        power_mode = dict(exit_code=mode.returncode, stdout=mode.stdout, stderr=mode.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        power_mode = dict(exit_code=None, error=str(exc))
    return dict(monotonic=start, elapsed_seconds=time.monotonic()-start,
                kernel=platform.release(), board=read_node('/proc/device-tree/model'),
                jetson_release=read_node('/etc/nv_tegra_release'),
                power_mode=power_mode, domains=domains, gpu_runtime=runtime,
                thermal=thermal, hardware_writes=False,
                units=dict(gpu='Hz', emc='Hz', cpu='kHz', thermal='millidegrees C'),
                note='writable is an access check, not proof of safe control; no railgate inference')
