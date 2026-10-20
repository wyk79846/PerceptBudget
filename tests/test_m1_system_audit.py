"""CPU-only saved-record fixtures, independent of production queue/model code."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_rtc_system import audit


class M1SystemAuditTest(unittest.TestCase):
    def fixture(self, root, sync=False, starvation=False):
        h, q = 50, 40 if starvation else 5
        origin = q
        at = origin if sync else (52 if starvation else 7)
        steps = at + 2
        stats = {'state': {'q01': [0.] * 8, 'q99': [2.] * 8},
                 'actions': {'q01': [-2.] * 7, 'q99': [2.] * 7}}
        (root / 'stats.json').write_text(json.dumps(stats))
        cfg = dict(H=h, candidate_q=[q], norm_stats_path=str(root / 'stats.json'),
                   checkpoint_profile='lerobot_h50_quantiles_v044',
                   rtc_initial_delay_steps=2, rtc_delay_buffer_size=10,
                   rtc_paper_schedule=True, diagnostic_condition='sync_no_rtc' if sync else 'actual_history',
                   rtc_exact_flash_attention=True, rtc_fused_norm=True,
                   native_gemm_policy='autotune')
        requests, commits, controls = [], [], []
        for pid, obs in enumerate([0, origin]):
            raw = np.full((h, 32), 1.25 + pid, dtype=np.float32)
            enabled = bool(pid and not sync)
            g = dict(guidance_enabled=enabled, executed_steps=obs if enabled else 0,
                     delay_estimate_steps=2 if enabled else 0)
            if not sync:
                g['actual_delay_steps'] = at - obs if pid else 0
                if pid:
                    g['delay_history_steps'] = [2]
            request = dict(prediction_id=pid, observation_step=obs, rtc_guidance=g,
                           rtc_solver_inputs=dict(noise=np.zeros((h, 32)).tolist(),
                               previous_raw=requests[0]['actions_raw'][obs:] if enabled else None,
                               delay=2 if enabled else 0, executed=obs if enabled else 0),
                           actions_raw=raw.tolist(), actions=(raw[:, :7]*2).tolist(),
                           observation_raw_state=[1.] * 8, model_state=[0.] * 32,
                           observation_monotonic=obs*10+1., input_ready_monotonic=obs*10+2.,
                           inference_started_monotonic=obs*10+3.,
                           inference_ended_monotonic=(at if pid else 0)*10+4.,
                           ready_monotonic=(at if pid else 0)*10+5.)
            requests.append(request)
            commits.append(dict(event='prediction_committed', prediction_id=pid,
                                commit_step=at if pid else 0,
                                commit_monotonic=(at if pid else 0)*10+6.,
                                expired_new_actions=at-obs if pid else 0,
                                new_prediction_remaining_actions=h-(at-obs) if pid else h))
        for step in range(steps):
            pid = int(step >= at)
            offset = step - requests[pid]['observation_step']
            available = offset < h
            controls.append(dict(event='control_step', step=step,
                before=dict(raw_state=[1.] * 8, sim_state=[0., 1.]),
                tick_monotonic=step*10., action_start_monotonic=step*10+7.,
                action_end_monotonic=step*10+8., applied_arm_gain=1., late_hold=False,
                source=dict(prediction_id=pid, offset=offset,
                            origin_step=requests[pid]['observation_step']) if available else None,
                action=requests[pid]['actions'][offset] if available else [0.] * 7))
        episode = dict(underflow_steps=[50, 51] if starvation else [],
                       initial_raw_state=[1.] * 8,
                       initial_sim_state_sha256=hashlib.sha256(np.asarray([0., 1.], dtype=np.float64).tobytes()).hexdigest())
        return cfg, requests, commits + controls, episode

    def run_fixture(self, sync=False, starvation=False, mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, requests, events, episode = self.fixture(root, sync, starvation)
            if mutate:
                mutate(cfg, requests, events, episode)
            (root/'rtc').mkdir()
            (root/'run.json').write_text(json.dumps(dict(config_snapshot=cfg)))
            (root/f"rtc/q{cfg['candidate_q'][0]}_requests.json").write_text(json.dumps(requests))
            (root/'rtc/trace.jsonl').write_text('\n'.join(map(json.dumps, events)))
            (root/'summary.json').write_text(json.dumps(dict(episode=episode)))
            return audit(root)

    def test_sync_bootstrap_records_pass(self):
        result = self.run_fixture(sync=True)
        self.assertTrue(result['passed'], result['failures'])
        self.assertEqual(result['audit_mode'], 'sync_no_rtc')

    def test_async_pass(self):
        result = self.run_fixture()
        self.assertTrue(result['passed'], result['failures'])

    def test_starvation_is_supply_quality(self):
        result = self.run_fixture(starvation=True)
        self.assertTrue(result['passed'], result['failures'])
        self.assertEqual(result['supply_quality']['underflow_steps'], [50, 51])
        self.assertFalse(result['supply_quality']['passed'])

    def test_corrupt_records_fail(self):
        mutations = [
            (True, False, lambda c, r, e, s: r[1]['rtc_solver_inputs'].update(previous_raw=[[0.]*32])),
            (True, False, lambda c, r, e, s: r[1]['rtc_guidance'].update(guidance_enabled=True)),
            (True, False, lambda c, r, e, s: r[1]['rtc_guidance'].update(delay_history_steps=[2])),
            (True, False, lambda c, r, e, s: r[1].update(model_state=[1.]*32)),
            (True, False, lambda c, r, e, s: r[1].update(actions=[[0.]*7]*50)),
            (True, False, lambda c, r, e, s: r[1].update(observation_step=6)),
            (False, False, lambda c, r, e, s: r[1]['rtc_guidance'].update(delay_history_steps=[99])),
            (False, False, lambda c, r, e, s: e[1].update(expired_new_actions=0)),
            (False, False, lambda c, r, e, s: r[1].update(ready_monotonic=9999)),
            (False, False, lambda c, r, e, s: s.update(initial_raw_state=[2.]*8)),
            (False, True, lambda c, r, e, s: e[52].update(action=[1.]*7)),
            (False, True, lambda c, r, e, s: s.update(underflow_steps=[])),
            (False, False, lambda c, r, e, s: e[3].update(source=None)),
        ]
        for sync, starvation, mutate in mutations:
            with self.subTest(sync=sync, starvation=starvation, mutate=mutate):
                result = self.run_fixture(sync, starvation, mutate)
                self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
