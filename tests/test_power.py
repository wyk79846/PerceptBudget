import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from perceptbudget.power import PowerMonitor, PowerReading, integrate_power_mj


class Ina3221PowerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        temp_root = Path(self.temporary.name)
        self.driver_root = temp_root / "sys/bus/i2c/drivers"
        device = temp_root / "sys/devices/platform/ina3221/1-0040"
        driver = self.driver_root / "ina3221x"
        driver.mkdir(parents=True)
        (driver / "1-0040").symlink_to(device, target_is_directory=True)
        self.hwmon = device / "hwmon/hwmon0"
        self.hwmon.mkdir(parents=True)

    def _write(self, name, value):
        (self.hwmon / name).write_text(str(value), encoding="ascii")

    def _record_sample(self, monitor, timestamp, power_mw):
        self._write("in2_label", "VDD_IN")
        self._write("in2_input", "1000")
        current_path = self.hwmon / "curr2_input"
        if power_mw is None:
            current_path.unlink(missing_ok=True)
        else:
            self._write("curr2_input", power_mw)
        with patch("perceptbudget.power.time.monotonic", return_value=timestamp):
            return monitor.sample()

    def _wait_for_reading(self, monitor):
        deadline = time.monotonic() + 2.0
        while not monitor.readings and time.monotonic() < deadline:
            time.sleep(0.002)
        self.assertTrue(monitor.readings, "power monitor did not poll")

    def test_reads_the_channel_named_vdd_in_and_converts_to_mw(self):
        self._write("in1_label", "VDD_CPU")
        self._write("in1_input", "10000")
        self._write("curr1_input", "1000")
        self._write("in2_label", "VDD_IN")
        self._write("in2_input", "5000")
        self._write("curr2_input", "1200")

        monitor = PowerMonitor(0.01, driver_root=self.driver_root)
        monitor.start()
        self._wait_for_reading(monitor)
        monitor.stop()

        reading = monitor.readings[0]
        self.assertAlmostEqual(reading.power_mw, 6000.0)
        self.assertIsNone(reading.error)
        self.assertGreaterEqual(reading.timestamp, 0.0)

    def test_missing_sensor_is_recorded_and_does_not_crash_monitor(self):
        monitor = PowerMonitor(0.01, driver_root=self.driver_root / "absent")
        monitor.start()
        self._wait_for_reading(monitor)
        monitor.stop()

        self.assertIsNone(monitor.readings[0].power_mw)
        self.assertIn("VDD_IN", monitor.readings[0].error)
        report = monitor.integrate_report(
            monitor.readings[0].timestamp - 1.0,
            monitor.readings[-1].timestamp + 1.0,
        )
        self.assertIsNone(report.energy_mj)
        self.assertEqual(report.coverage_seconds, 0.0)
        self.assertEqual(report.coverage_fraction, 0.0)
        self.assertGreater(report.missing_count, 0)
        self.assertIsNone(report.total_energy_mj)
        self.assertIsNone(monitor.integrate_mj(0.0, 1.0))

    def test_report_clips_boundaries_and_marks_complete_total(self):
        monitor = PowerMonitor(
            0.1, driver_root=self.driver_root, max_gap_seconds=10.0,
        )
        self._record_sample(monitor, 0.0, 100.0)
        self._record_sample(monitor, 10.0, 300.0)

        report = monitor.integrate_report(2.0, 8.0)

        # Linear power is 140 mW at t=2 and 260 mW at t=8.
        self.assertAlmostEqual(report.energy_mj, 1200.0)
        self.assertAlmostEqual(report.coverage_seconds, 6.0)
        self.assertAlmostEqual(report.coverage_fraction, 1.0)
        self.assertEqual(report.missing_count, 0)
        self.assertAlmostEqual(report.total_energy_mj, 1200.0)
        self.assertAlmostEqual(monitor.integrate_mj(2.0, 8.0), 1200.0)

    def test_report_exposes_partial_energy_without_calling_it_total(self):
        monitor = PowerMonitor(0.5, driver_root=self.driver_root)
        self._record_sample(monitor, 0.0, 100.0)
        self._record_sample(monitor, 1.0, None)
        self._record_sample(monitor, 2.0, 100.0)
        self._record_sample(monitor, 3.0, 100.0)

        report = monitor.integrate_report(0.0, 3.0)

        self.assertAlmostEqual(report.energy_mj, 100.0)
        self.assertAlmostEqual(report.coverage_seconds, 1.0)
        self.assertAlmostEqual(report.coverage_fraction, 1.0 / 3.0)
        self.assertEqual(report.missing_count, 1)
        self.assertIsNone(report.total_energy_mj)
        self.assertAlmostEqual(monitor.integrate_mj(0.0, 3.0), 100.0)

    def test_report_does_not_bridge_a_long_gap(self):
        monitor = PowerMonitor(0.1, driver_root=self.driver_root)
        self._record_sample(monitor, 0.0, 100.0)
        self._record_sample(monitor, 1.0, 100.0)

        report = monitor.integrate_report(0.0, 1.0)

        self.assertIsNone(report.energy_mj)
        self.assertEqual(report.coverage_seconds, 0.0)
        self.assertEqual(report.coverage_fraction, 0.0)
        self.assertEqual(report.missing_count, 0)
        self.assertIsNone(report.total_energy_mj)

    def test_stop_takes_final_sample_after_window_boundary(self):
        readings = (
            ("in1_label", "VDD_CPU"),
            ("in1_input", "10000"),
            ("curr1_input", "1000"),
            ("in2_label", "VDD_IN"),
            ("in2_input", "5000"),
            ("curr2_input", "1200"),
        )
        for name, value in readings:
            self._write(name, value)
        monitor = PowerMonitor(
            0.02, driver_root=self.driver_root, max_gap_seconds=1.0,
        )
        monitor.start()
        start = monitor.mark()
        time.sleep(0.04)
        stop = monitor.mark()
        monitor.stop()

        self.assertGreaterEqual(monitor.readings[-1].timestamp, stop)
        report = monitor.integrate_report(start, stop)
        self.assertAlmostEqual(report.coverage_fraction, 1.0)
        self.assertIsNotNone(report.total_energy_mj)

    def test_missing_read_breaks_integration_and_no_samples_return_none(self):
        readings = (
            PowerReading(0.0, 100.0),
            PowerReading(1.0, None, "read failed"),
            PowerReading(2.0, 100.0),
            PowerReading(3.0, 100.0),
        )

        self.assertAlmostEqual(
            integrate_power_mj(readings, 0.0, 3.0), 100.0,
        )
        self.assertAlmostEqual(
            integrate_power_mj(
                (PowerReading(0.0, 100.0), PowerReading(10.0, 300.0)),
                2.0, 8.0, max_gap_seconds=10.0,
            ),
            1200.0,
        )
        self.assertIsNone(integrate_power_mj((), 0.0, 3.0))
        self.assertIsNone(
            integrate_power_mj((PowerReading(1.0, None, "missing"),), 0.0, 3.0)
        )

    def test_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            integrate_power_mj((), 2.0, 1.0)


if __name__ == "__main__":
    unittest.main()
