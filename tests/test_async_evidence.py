import unittest

from scripts.summarize_async_evidence import forecast_positions, group_evidence, render


class AsyncEvidenceTest(unittest.TestCase):
    def test_events_are_future_offsets_not_wall_times(self):
        actions = [[.2, 0, 0, 0, 0, 0, -1] for _ in range(20)]
        actions[5] = [.05, 0, 0, 0, 0, 0, 1]
        events, missing = forecast_positions([
            {'prediction_id': 3, 'observation_step': 12, 'actions': actions}])
        switch = next(e for e in events if e['type'] == 'gripper_switch')
        self.assertEqual(switch['forecast_offset'], 5)
        self.assertEqual(switch['absolute_forecast_step'], 17)
        self.assertEqual(switch['lead_nominal_seconds'], .25)
        self.assertEqual(missing, 0)

    def test_incomplete_forecast_is_not_padded(self):
        self.assertEqual(forecast_positions([{'actions': []}]), ([], 1))

    def test_stage_buffers_do_not_create_safe_interval(self):
        group = group_evidence([{
            'task_id': 0,
            'steady_commits': [{'first_execution_stage_proxy': 'unknown',
                                'new_valid_seconds': .65, 'old_queue_seconds': 0}],
            'stage_proxy_control_steps': {'unknown': 2}, 'forecast_event_positions': []}])
        self.assertIsNone(group['B_safe_seconds'])
        self.assertEqual(group['new_remaining_nominal_seconds']['min'], .65)
        self.assertEqual(group['by_first_execution_stage_proxy']['unknown']['commits'], 1)
        text = render({'q': {'8': group}})
        self.assertIn('未测', text)
        self.assertIn('尚未可靠标注', text)


if __name__ == '__main__':
    unittest.main()
