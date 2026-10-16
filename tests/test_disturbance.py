from types import SimpleNamespace
import json
import unittest

import numpy as np

from perceptbudget.disturbance import nudge_object


TARGET = 'akita_black_bowl_1'


class DisturbanceTest(unittest.TestCase):
    def make_env(self):
        # Nonzero offsets ensure the helper cannot accidentally move robot/other qpos.
        model = SimpleNamespace(
            body_name2id=lambda name: {'bowl_main': 2}[name],
            body_parentid=np.array([0, 0, 0]),
            body_jntnum=np.array([0, 1, 1]), body_jntadr=np.array([-1, 0, 1]),
            joint_id2name=lambda index: {0: 'robot_joint', 1: 'bowl_joint'}[index],
            joint_name2id=lambda name: {'robot_joint': 0, 'bowl_joint': 1}[name],
            jnt_bodyid=np.array([1, 2]), jnt_type=np.array([3, 0]),
            get_joint_qpos_addr=lambda name: (2, 9),
            get_joint_qvel_addr=lambda name: (2, 8))
        data = SimpleNamespace(qpos=np.array([8., 9., .1, .2, .3, .5, .5, .5, .5, 7.]),
                               qvel=np.arange(10, dtype=float), time=1.5)
        sim = SimpleNamespace(model=model, data=data, forward_calls=0)

        def forward():
            sim.forward_calls += 1
            data.body_xpos = data.qpos[2:5].copy()

        sim.forward = forward
        domain = SimpleNamespace(sim=sim, objects_dict={TARGET: SimpleNamespace(
            root_body='bowl_main', joints=['bowl_joint'])}, fixtures_dict={})
        return SimpleNamespace(env=domain)

    def test_translation_preserves_quat_vel_time_and_other_state(self):
        for wrapped in (True, False):
            env = self.make_env()
            sim = env.env.sim
            qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
            audit = nudge_object(env if wrapped else env.env, TARGET, [.02, 0, 0])
            expected = qpos.copy()
            expected[2] += .02
            np.testing.assert_array_equal(sim.data.qpos, expected)
            np.testing.assert_array_equal(sim.data.qvel, qvel)
            np.testing.assert_array_equal(sim.data.body_xpos, expected[2:5])
            np.testing.assert_allclose(audit['actual_delta'], [.02, 0, 0])
            self.assertEqual(audit['before_quat'], audit['after_quat'])
            self.assertEqual(audit['before_vel'], audit['after_vel'])
            self.assertEqual(audit['simtime_before'], audit['simtime_after'])
            self.assertEqual(sim.data.time, 1.5)
            self.assertEqual(sim.forward_calls, 1)
            self.assertEqual(audit['body_name'], 'bowl_main')
            self.assertEqual(audit['joint_name'], 'bowl_joint')
            json.dumps(audit, allow_nan=False)

    def assert_rejected(self, env, target=TARGET, delta=(.02, 0, 0)):
        sim = env.env.sim
        before = sim.data.qpos.copy()
        with self.assertRaises(ValueError):
            nudge_object(env, target, delta)
        np.testing.assert_array_equal(sim.data.qpos, before)
        self.assertEqual(sim.forward_calls, 0)

    def test_invalid_or_missing_target_and_fixture(self):
        for target in ('other_bowl', 'robot0', 'cabinet', '', None):
            self.assert_rejected(self.make_env(), target)
        env = self.make_env()
        del env.env.objects_dict[TARGET]
        self.assert_rejected(env)
        env = self.make_env()
        env.env.fixtures_dict[TARGET] = object()
        self.assert_rejected(env)

    def test_invalid_delta(self):
        for delta in (0, [1, 2], [[1, 2, 3]], [np.nan, 0, 0], [0, np.inf, 0]):
            self.assert_rejected(self.make_env(), delta=delta)

    def test_nonfreejoint_and_wrong_mapping(self):
        for field, value in (('jnt_type', [3, 1]), ('body_jntnum', [0, 1, 0]),
                             ('body_parentid', [0, 0, 1]), ('jnt_bodyid', [1, 1]),
                             ('body_jntadr', [-1, 0, 0])):
            env = self.make_env()
            setattr(env.env.sim.model, field, np.array(value))
            self.assert_rejected(env)

    def test_invalid_address_or_nonfinite_state(self):
        for addr in (2, (2, 6), (-1, 6), (8, 15)):
            env = self.make_env()
            env.env.sim.model.get_joint_qpos_addr = lambda name: addr
            self.assert_rejected(env)
        for field in ('qpos', 'qvel'):
            env = self.make_env()
            getattr(env.env.sim.data, field)[3] = np.nan
            self.assert_rejected(env)
        env = self.make_env()
        env.env.sim.data.time = np.inf
        self.assert_rejected(env)

    def test_time_advance_is_detected(self):
        env = self.make_env()
        env.env.sim.forward = lambda: setattr(env.env.sim.data, 'time', 1.6)
        with self.assertRaises(RuntimeError):
            nudge_object(env, TARGET, [.02, 0, 0])
