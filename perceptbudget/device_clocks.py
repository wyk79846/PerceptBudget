"""Bounded clock transactions; no nvpmodel, governor, voltage or thermal writes."""
from pathlib import Path


# Screening candidates, NOT the three final M2 operating points. Never exceed
# the currently frozen baseline caps, even if the hardware advertises more.
PROFILES = {
    'baseline': (1173000000, 3199000000, 1497600),
    'gpu816': (816000000, 3199000000, 1497600),
    'emc2133': (1173000000, 2133000000, 1497600),
    'cpu1190': (1173000000, 3199000000, 1190400),
    'gpu_emc': (816000000, 2133000000, 1497600),
    'gpu_emc_cpu': (816000000, 2133000000, 1190400),
    # 2750MHz was advertised but two hardware preflights stayed at3199MHz.
    # Keep known-working EMC and screen an intermediate GPU rate instead.
    'balanced': (1020000000, 3199000000, 1497600),
}

SCREEN_PROFILES = ('baseline', 'gpu816', 'emc2133', 'cpu1190')
MATRIX_PROFILES = ('baseline', 'balanced', 'gpu_emc')


def observed_switch(control, target, current):
    import time
    command = time.monotonic()
    if target != current:
        control.apply(target)
    settings = time.monotonic()
    gpu, emc, cpu = PROFILES[target]
    while True:
        actual = control.actual()
        observed = time.monotonic()
        expected = {name: gpu if name.endswith('/cur_freq') else
                    emc if name.endswith('/rate') else cpu for name in actual}
        if actual == expected:
            return dict(command_monotonic=command, settings_readback_monotonic=settings,
                        observed_match_monotonic=observed, actual=actual,
                        change_required=target != current)
        if observed >= settings+.5:
            raise RuntimeError('Frequency did not reach requested value within0.5s')
        time.sleep(.002)


def cycle_trace(control, profile, emit, *, cycles=5, phase_seconds=3):
    """Bounded static transitions. Readback time is not GPU wake/service time."""
    import time
    current = 'baseline'
    for cycle in range(cycles):
        for phase, target in (('low', profile), ('high', 'baseline')):
            record = observed_switch(control, target, current)
            emit(dict(status='PHASE', cycle=cycle, phase=phase, profile=target, **record))
            end = record['observed_match_monotonic']+phase_seconds
            while time.monotonic() < end:
                emit(dict(status='SAMPLE', monotonic=time.monotonic(), actual=control.actual()))
                time.sleep(min(.02, max(0, end-time.monotonic())))
            current = target
        emit(dict(status='CYCLE_END', cycle=cycle, monotonic=time.monotonic()))


def recovery_trace(control, profile, idle_seconds, emit, fd, *, serve_profile='baseline', paired_idle=False):
    """Five finite RUN/DONE exchanges; no arbitrary commands or model code."""
    import os
    import select
    import time
    allowed_idle = (.05, .1, .15, .2, 1., 2.) if paired_idle else (0., .2, 1.)
    if profile not in MATRIX_PROFILES or serve_profile not in MATRIX_PROFILES or idle_seconds not in allowed_idle:
        raise ValueError('Recovery scenario outside whitelist')
    def receive(expected, index):
        deadline = time.monotonic()+(6 if paired_idle else 3)
        data = b''
        while time.monotonic() < deadline:
            emit(dict(status='SAMPLE', monotonic=time.monotonic(), actual=control.actual(),
                      request_index=index))
            ready, _, _ = select.select([fd], [], [], min(.02, max(0, deadline-time.monotonic())))
            if ready:
                part = os.read(fd, 32)
                if not part:
                    raise RuntimeError('Recovery worker pipe closed')
                data += part
                if len(data) > 16:
                    raise ValueError('Oversize recovery protocol token')
                if b'\n' in data:
                    if data != expected+b'\n':
                        raise ValueError('Unexpected recovery protocol token')
                    return
        raise TimeoutError('Recovery worker handshake exceeded'+('6s' if paired_idle else '3s'))
    for index in range(5):
        receive(b'RUN', index)
        down = observed_switch(control, profile, serve_profile)
        # Paired trials share the command origin, including down-switch cost.
        idle_end = down['command_monotonic' if paired_idle else 'observed_match_monotonic']+idle_seconds
        while time.monotonic() < idle_end:
            emit(dict(status='SAMPLE', monotonic=time.monotonic(), actual=control.actual(),
                      request_index=index))
            time.sleep(min(.02, max(0, idle_end-time.monotonic())))
        up = observed_switch(control, serve_profile, profile)
        emit(dict(status='SERVE', request_index=index, down=down, up=up,
                  requested_idle_seconds=idle_seconds, profile=profile, serve_profile=serve_profile,
                  paired_idle=paired_idle, requested_serve_monotonic=idle_end))
        receive(b'DONE', index)
        emit(dict(status='REQUEST_ACK', request_index=index, monotonic=time.monotonic()))


class ClockControl:
    def __init__(self, sys_root='/sys'):
        self.root = Path(sys_root)
        self.gpu = self.root/'class/devfreq/17000000.gpu'
        self.emc = self.root/'kernel/debug/bpmp/debug/clk/emc'
        self.cpus = sorted((self.root/'devices/system/cpu/cpufreq').glob('policy*'))
        if not self.cpus:
            raise RuntimeError('No CPU policies found')

    @staticmethod
    def read(path):
        return int(path.read_text().strip())

    @staticmethod
    def write(path, value):
        # Open existing nodes only; never create missing control files.
        with path.open('r+') as stream:
            stream.write(str(int(value)))
            stream.flush()

    def paths(self):
        paths = [self.gpu/'min_freq', self.gpu/'max_freq',
                 self.emc/'rate', self.emc/'mrq_rate_locked']
        for cpu in self.cpus:
            paths += [cpu/'scaling_min_freq', cpu/'scaling_max_freq']
        return paths

    def snapshot(self):
        return {str(p.relative_to(self.root)): self.read(p) for p in self.paths()}

    def validate(self, name):
        gpu, emc, cpu = PROFILES[name]
        if gpu not in map(int, (self.gpu/'available_frequencies').read_text().split()):
            raise ValueError('GPU rate not advertised')
        for policy in self.cpus:
            if cpu not in map(int, (policy/'scaling_available_frequencies').read_text().split()):
                raise ValueError('CPU rate not advertised')
        table = (self.emc/'dvfs_table').read_text().splitlines()
        rates = {int(line.split()[1]) for line in table
                 if line.strip().startswith('vdd_core ') and len(line.split()) >= 3}
        if emc not in rates or not self.read(self.emc/'min_rate') <= emc <= self.read(self.emc/'max_rate'):
            raise ValueError('EMC rate outside advertised range')
        return gpu, emc, cpu

    def bounds(self, low, high, minimum, maximum):
        if minimum > maximum:
            raise ValueError('Reversed clock limits')
        # Avoid transient min > max in either transition direction.
        if minimum > self.read(high):
            self.write(high, maximum)
            self.write(low, minimum)
        else:
            self.write(low, minimum)
            self.write(high, maximum)
        if (self.read(low), self.read(high)) != (minimum, maximum):
            raise RuntimeError('Clock bounds readback mismatch')

    def apply(self, name):
        gpu, emc, cpu = self.validate(name)  # all validation before first write
        self.bounds(self.gpu/'min_freq', self.gpu/'max_freq', gpu, gpu)
        # Match installed NVIDIA jetson_clocks: lock BEFORE setting rate.
        # Unlocking creates an automatic-DVFS race between write and relock.
        self.write(self.emc/'mrq_rate_locked', 1)
        self.write(self.emc/'rate', emc)
        self.wait_emc(emc)
        for policy in self.cpus:
            self.bounds(policy/'scaling_min_freq', policy/'scaling_max_freq', cpu, cpu)

    def wait_emc(self, requested):
        import time
        deadline = time.monotonic()+.5
        while True:
            actual = self.read(self.emc/'rate')
            locked = self.read(self.emc/'mrq_rate_locked')
            if actual == requested and locked == 1:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'EMC readback mismatch after0.5s: requested={requested}, '
                                   f'actual={actual}, locked={locked}; no rounding tolerance applied')
            time.sleep(.002)

    def restore(self, saved):
        expected = {str(p.relative_to(self.root)) for p in self.paths()}
        if set(saved) != expected or any(type(v) is not int for v in saved.values()):
            raise ValueError('Snapshot node set or types mismatch')
        value = lambda p: saved[str(p.relative_to(self.root))]
        errors = []
        operations = [(self.bounds, (self.gpu/'min_freq', self.gpu/'max_freq',
                       value(self.gpu/'min_freq'), value(self.gpu/'max_freq')))]
        operations += [(self.write, (self.emc/'mrq_rate_locked', 1)),
                       (self.write, (self.emc/'rate', value(self.emc/'rate')))]
        if value(self.emc/'mrq_rate_locked') == 1:
            operations.append((self.wait_emc, (value(self.emc/'rate'),)))
        operations.append((self.write, (self.emc/'mrq_rate_locked', value(self.emc/'mrq_rate_locked'))))
        for p in self.cpus:
            operations.append((self.bounds, (p/'scaling_min_freq', p/'scaling_max_freq',
                               value(p/'scaling_min_freq'), value(p/'scaling_max_freq'))))
        for fn, args in operations:
            try:
                fn(*args)
            except Exception as exc:
                errors.append(str(exc))
        # Unlocked EMC may immediately change rate; verify lock, not that rate.
        current = self.snapshot()
        for key, wanted in saved.items():
            if key.endswith('/rate') and saved[str((self.emc/'mrq_rate_locked').relative_to(self.root))] == 0:
                continue
            if current[key] != wanted:
                errors.append(f'Restore mismatch: {key}')
        if errors:
            raise RuntimeError('; '.join(errors))

    def actual(self):
        paths = [self.gpu/'cur_freq', self.emc/'rate']
        paths += [p/'scaling_cur_freq' for p in self.cpus]
        return {str(p.relative_to(self.root)): self.read(p) for p in paths}
