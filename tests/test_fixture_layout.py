from types import SimpleNamespace
import unittest
import numpy as np
from perceptbudget.fixtures import enforce_fixture_layout


class FixtureLayoutTest(unittest.TestCase):
    def env(self):
        model = SimpleNamespace(body_pos=np.arange(9, dtype=float).reshape(3, 3),
                                body_quat=np.tile([1., 0., 0., 0.], (3, 1)),
                                body_name2id=lambda name: {'cabinet': 0, 'stove': 2}[name])
        domain = SimpleNamespace(sim=SimpleNamespace(model=model), fixtures_dict={
            'a': SimpleNamespace(root_body='cabinet'), 'b': SimpleNamespace(root_body='stove')})
        return SimpleNamespace(env=domain, _pb_lock_fixtures=True)

    def test_capture_then_restore_only_fixed_bodies(self):
        env = self.env()
        model = env.env.sim.model
        original = model.body_pos.copy()
        enforce_fixture_layout(env)
        model.body_pos[:] += .02
        model.body_quat[0] = [.99, 0, 0, .01]
        enforce_fixture_layout(env)
        np.testing.assert_array_equal(model.body_pos[[0, 2]], original[[0, 2]])
        np.testing.assert_array_equal(model.body_pos[1], original[1]+.02)
        np.testing.assert_array_equal(model.body_quat[0], [1, 0, 0, 0])

    def test_disabled_is_noop_for_legacy_environment(self):
        enforce_fixture_layout(SimpleNamespace())

    def test_names_and_nonfinite_pose_are_rejected(self):
        env = self.env()
        enforce_fixture_layout(env)
        env._pb_fixture_layout['cabinet']['pos'][0] = float('nan')
        with self.assertRaises(RuntimeError):
            enforce_fixture_layout(env)
        env = self.env()
        enforce_fixture_layout(env)
        del env._pb_fixture_layout['cabinet']
        with self.assertRaises(RuntimeError):
            enforce_fixture_layout(env)
