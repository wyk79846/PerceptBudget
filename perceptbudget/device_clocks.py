"""Bounded clock transactions; no nvpmodel, governor, voltage or thermal writes."""
from pathlib import Path


# Screening candidates, NOT the three final M2 operating points. Never exceed
# the currently frozen baseline caps, even if the hardware advertises more.
PROFILES = {
    'baseline': (1173000000, 3199000000, 1497600),
    'gpu816': (816000000, 3199000000, 1497600),
    'emc2133': (1173000000, 2133000000, 1497600),
    'cpu1190': (1173000000, 3199000000, 1190400),
}


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
        self.write(self.emc/'mrq_rate_locked', 0)
        self.write(self.emc/'rate', emc)
        self.write(self.emc/'mrq_rate_locked', 1)
        if self.read(self.emc/'rate') != emc or self.read(self.emc/'mrq_rate_locked') != 1:
            raise RuntimeError('EMC readback mismatch')
        for policy in self.cpus:
            self.bounds(policy/'scaling_min_freq', policy/'scaling_max_freq', cpu, cpu)

    def restore(self, saved):
        expected = {str(p.relative_to(self.root)) for p in self.paths()}
        if set(saved) != expected or any(type(v) is not int for v in saved.values()):
            raise ValueError('Snapshot node set or types mismatch')
        value = lambda p: saved[str(p.relative_to(self.root))]
        errors = []
        operations = [(self.bounds, (self.gpu/'min_freq', self.gpu/'max_freq',
                       value(self.gpu/'min_freq'), value(self.gpu/'max_freq')))]
        operations += [(self.write, (self.emc/'mrq_rate_locked', 0)),
                       (self.write, (self.emc/'rate', value(self.emc/'rate'))),
                       (self.write, (self.emc/'mrq_rate_locked', value(self.emc/'mrq_rate_locked')))]
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
