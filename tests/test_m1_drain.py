import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from perceptbudget.power import PowerReading, _integrate_report
from scripts import probe_delay_buffer as probe


class M1DrainTest(unittest.TestCase):
    def test_sampling_covers_shutdown_and_windows_exclude_artifact_work(self):
        clock = SimpleNamespace(now=12.)
        readings = [PowerReading(10., 2000.), PowerReading(12., 2000.)]
        monitor = Mock()
        monitor.readings = readings
        monitor.integrate_report.side_effect = lambda a, b: _integrate_report(
            readings, a, b, max_gap_seconds=5.)

        def shutdown(wait):
            self.assertTrue(wait)
            monitor.stop.assert_not_called()
            clock.now = 15.
            readings.append(PowerReading(15., 2000.))

        def stop():
            # Final sampling/monitor join is outside the inference window.
            clock.now = 16.
            readings.append(PowerReading(16., 2000.))

        monitor.stop.side_effect = stop
        executor = Mock(shutdown=Mock(side_effect=shutdown))
        with patch.object(probe.time, 'monotonic', side_effect=lambda: clock.now):
            result = probe._drain_executor(executor, monitor, 10., 12.)
        self.assertEqual(result['drain_seconds'], 3.)
        self.assertEqual(result['drain_start_monotonic'], 12.)
        self.assertEqual(result['drain_stop_monotonic'], 15.)
        self.assertEqual(result['including_drain_seconds'], 5.)
        self.assertEqual(result['drain_power']['total_energy_mj'], 6000.)
        self.assertEqual(result['including_drain_power']['total_energy_mj'], 10000.)
        clock.now = 100.  # Compression and writes must not expand either window.
        task_power = probe.base._power_result(monitor, 10., 12.)
        self.assertEqual(task_power['total_energy_mj'], 4000.)
        self.assertEqual(task_power['total_energy_mj'] +
                         result['drain_power']['total_energy_mj'],
                         result['including_drain_power']['total_energy_mj'])
        monitor.stop.assert_called_once_with()

    def test_shutdown_exception_still_stops_monitor(self):
        monitor = Mock()
        executor = Mock()
        executor.shutdown.side_effect = RuntimeError('worker shutdown failed')
        with self.assertRaisesRegex(RuntimeError, 'worker shutdown failed'):
            probe._drain_executor(executor, monitor, 10., 12.)
        monitor.stop.assert_called_once_with()

    def test_missing_power_is_not_reported_as_zero_energy(self):
        monitor = Mock()
        monitor.readings = []
        monitor.integrate_report.side_effect = lambda a, b: _integrate_report([], a, b, 1.)
        with patch.object(probe.time, 'monotonic', return_value=13.):
            result = probe._drain_executor(Mock(), monitor, 10., 12.)
        self.assertIsNone(result['drain_power']['total_energy_mj'])
        self.assertIsNone(result['including_drain_power']['total_energy_mj'])


if __name__ == '__main__':
    unittest.main()
