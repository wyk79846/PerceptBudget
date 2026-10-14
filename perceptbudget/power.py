"""Read-only INA3221 VDD_IN power sampling and integration.

The reported rail is the module input (which can include CPU, GPU, memory, and
other loads); it is not an estimate of GPU-only power. INA3221 hwmon voltage
and current values are read in mV and mA and converted to mW.
"""

import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


INA3221_DRIVERS = Path("/sys/bus/i2c/drivers")
_LABEL_PATTERN = re.compile(r"in([0-9]+)_label\Z")


@dataclass(frozen=True)
class PowerReading:
    """One monotonic-clock poll; ``power_mw=None`` represents a missed read."""

    timestamp: float
    power_mw: Optional[float]
    error: Optional[str] = None


@dataclass(frozen=True)
class PowerReport:
    """Window integral and data coverage.

    ``energy_mj`` is only the integral over covered segments. ``total_energy_mj``
    is populated only when those segments cover the full non-empty window.
    ``missing_count`` counts failed reads inside the window; long unobserved gaps
    still reduce coverage even when no failed read was recorded.
    """

    energy_mj: Optional[float]
    coverage_seconds: float
    coverage_fraction: float
    missing_count: int
    total_energy_mj: Optional[float]


def _read_vdd_in_mw(driver_root: Path) -> float:
    """Read VDD_IN from an INA3221 driver's hwmon files, returning mW."""
    found_label = False
    read_errors = []
    for driver in sorted(driver_root.glob("ina3221*")):
        if not driver.is_dir():
            continue
        # Device entries under /sys/bus/i2c/drivers are commonly symlinks.
        for label_path in sorted(driver.glob("*/hwmon/hwmon*/in*_label")):
            match = _LABEL_PATTERN.fullmatch(label_path.name)
            if match is None:
                continue
            try:
                label = label_path.read_text(encoding="ascii").strip()
            except OSError as exc:
                read_errors.append(str(exc))
                continue
            if label != "VDD_IN":
                continue

            found_label = True
            channel = match.group(1)
            try:
                voltage_mv = float(
                    (label_path.parent / "in{}_input".format(channel))
                    .read_text(encoding="ascii").strip()
                )
                current_ma = float(
                    (label_path.parent / "curr{}_input".format(channel))
                    .read_text(encoding="ascii").strip()
                )
            except (OSError, ValueError) as exc:
                read_errors.append(str(exc))
                continue
            power_mw = voltage_mv * current_ma / 1000.0
            if not math.isfinite(power_mw):
                read_errors.append("INA3221 returned a non-finite VDD_IN reading")
                continue
            return power_mw

    if found_label:
        detail = "; ".join(read_errors) or "VDD_IN input files are unavailable"
        raise OSError("无法读取 INA3221 VDD_IN：" + detail)
    raise FileNotFoundError("未发现 INA3221 hwmon 的 VDD_IN 通道")


def _integrate_report(
    readings: Iterable[PowerReading], start: float, stop: float,
    max_gap_seconds: float,
) -> PowerReport:
    if not math.isfinite(start) or not math.isfinite(stop):
        raise ValueError("start 和 stop 必须是有限的 monotonic 时间戳")
    if stop < start:
        raise ValueError("stop 不能早于 start")
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds 必须是正的有限秒数")

    ordered = sorted(readings, key=lambda reading: reading.timestamp)
    energy_mj = 0.0
    coverage_seconds = 0.0
    for left, right in zip(ordered, ordered[1:]):
        if left.power_mw is None or right.power_mw is None:
            continue
        interval = right.timestamp - left.timestamp
        if interval <= 0 or interval > max_gap_seconds:
            continue

        clipped_start = max(start, left.timestamp)
        clipped_stop = min(stop, right.timestamp)
        if clipped_stop <= clipped_start:
            continue

        start_fraction = (clipped_start - left.timestamp) / interval
        stop_fraction = (clipped_stop - left.timestamp) / interval
        clipped_power_start = left.power_mw + (
            right.power_mw - left.power_mw
        ) * start_fraction
        clipped_power_stop = left.power_mw + (
            right.power_mw - left.power_mw
        ) * stop_fraction
        energy_mj += (
            (clipped_power_start + clipped_power_stop) * 0.5
            * (clipped_stop - clipped_start)
        )
        coverage_seconds += clipped_stop - clipped_start

    duration = stop - start
    coverage_seconds = min(coverage_seconds, duration)
    coverage_fraction = coverage_seconds / duration if duration > 0 else 0.0
    if coverage_fraction > 1.0:
        coverage_fraction = 1.0
    has_coverage = coverage_seconds > 0.0
    energy = energy_mj if has_coverage else None
    missing_count = sum(
        1 for reading in ordered
        if reading.power_mw is None and start <= reading.timestamp <= stop
    )
    tolerance = max(1e-9, duration * 1e-9)
    complete = duration > 0 and coverage_seconds >= duration - tolerance
    return PowerReport(
        energy_mj=energy,
        coverage_seconds=coverage_seconds,
        coverage_fraction=coverage_fraction,
        missing_count=missing_count,
        total_energy_mj=energy if complete and energy is not None else None,
    )


def integrate_power_mj(
    readings: Iterable[PowerReading], start: float, stop: float,
    *, max_gap_seconds: float = 1.0,
) -> Optional[float]:
    """Return the covered partial integral in mJ; misses and long gaps break it."""
    return _integrate_report(
        readings, start, stop, max_gap_seconds,
    ).energy_mj


class PowerMonitor:
    """Poll INA3221 VDD_IN in a background thread without controlling hardware."""

    def __init__(
        self, poll_interval: float = 0.1, *, driver_root=INA3221_DRIVERS,
        max_gap_seconds: Optional[float] = None,
    ):
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval 必须是正的有限秒数")
        self.poll_interval = float(poll_interval)
        self.max_gap_seconds = (
            self.poll_interval * 3.0
            if max_gap_seconds is None else float(max_gap_seconds)
        )
        if not math.isfinite(self.max_gap_seconds) or self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds 必须是正的有限秒数")
        self.driver_root = Path(driver_root)
        self._readings = []
        self._lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._final_sampled = False

    @staticmethod
    def mark() -> float:
        """Return a monotonic timestamp for a task's start or stop boundary."""
        return time.monotonic()

    @property
    def readings(self):
        """A thread-safe immutable snapshot, including missed reads."""
        with self._lock:
            return tuple(self._readings)

    def start(self):
        """Sample once, then start polling; missing hardware is recorded safely."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._final_sampled = False
            self.sample()
            self._thread = threading.Thread(
                target=self._poll, name="ina3221-power-monitor", daemon=True,
            )
            self._thread.start()

    def stop(self):
        """Join the poller, then take one final sample to bracket the stop mark."""
        with self._lifecycle_lock:
            if self._thread is None:
                return
            self._stop_event.set()
            self._thread.join()
            if not self._final_sampled:
                self._final_sampled = True
                self.sample()

    def sample(self) -> PowerReading:
        """Read once synchronously; serialized with background polling."""
        with self._sample_lock:
            try:
                power_mw = _read_vdd_in_mw(self.driver_root)
                reading = PowerReading(
                    timestamp=time.monotonic(), power_mw=power_mw,
                )
            except (OSError, OverflowError, ValueError) as exc:
                reading = PowerReading(
                    timestamp=time.monotonic(), power_mw=None, error=str(exc),
                )
            with self._lock:
                self._readings.append(reading)
            return reading

    def integrate_report(self, start: float, stop: float) -> PowerReport:
        """Report partial energy, coverage, misses, and total only for full coverage."""
        return _integrate_report(
            self.readings, start, stop, self.max_gap_seconds,
        )

    def integrate_mj(self, start: float, stop: float) -> Optional[float]:
        """Return the covered partial integral, preserving the earlier API."""
        return self.integrate_report(start, stop).energy_mj

    def _poll(self):
        while not self._stop_event.wait(self.poll_interval):
            self.sample()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.stop()
