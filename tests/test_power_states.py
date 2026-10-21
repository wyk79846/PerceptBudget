import unittest

from perceptbudget.power import PowerReading
from perceptbudget.power_windows import block_schedule, measurement_summary


class PowerStatesTest(unittest.TestCase):
    def test_long_windows_and_input_power_ratio(self):
        readings = [PowerReading(i*.1, 10000 if i < 80 else 20000) for i in range(162)]
        blocks = [dict(mode='idle', start_monotonic=0., stop_monotonic=8.),
                  dict(mode='busy', start_monotonic=8., stop_monotonic=16.)]
        result = measurement_summary(readings, blocks)
        self.assertAlmostEqual(result['P_idle_w'], 10)
        self.assertAlmostEqual(result['P_on_w'], 20)
        self.assertAlmostEqual(result['r'], .5)
        self.assertEqual(result['blocks'][0]['measured_start'], 1.)
        self.assertEqual(result['blocks'][0]['measured_stop'], 7.5)

    def test_gaps_do_not_invent_power(self):
        result = measurement_summary([PowerReading(0., 10000), PowerReading(16., 20000)],
            [dict(mode='idle', start_monotonic=0., stop_monotonic=8.),
             dict(mode='busy', start_monotonic=8., stop_monotonic=16.)])
        self.assertIsNone(result['P_on_w'])
        self.assertIsNone(result['r'])

    def test_schedule_is_one_minute_not_a_task_scan(self):
        self.assertEqual(block_schedule(), ['idle', 'busy', 'idle', 'busy', 'idle', 'busy', 'idle'])


if __name__ == '__main__':
    unittest.main()
