"""Keep a task's fixed-body layout identical across hard resets.

LIBERO's state vector does not contain model.body_pos/body_quat for fixtures.
The first prepared layout is retained; no joint state or control is modified.
"""


def enforce_fixture_layout(env):
    if not getattr(env, '_pb_lock_fixtures', False):
        return
    import numpy as np
    domain = env.env
    roots = sorted(obj.root_body for obj in domain.fixtures_dict.values())
    model = domain.sim.model
    layout = getattr(env, '_pb_fixture_layout', None)
    if layout is None:
        env._pb_fixture_layout = {
            name: {'pos': model.body_pos[model.body_name2id(name)].tolist(),
                   'quat': model.body_quat[model.body_name2id(name)].tolist()} for name in roots}
        return
    if set(layout) != set(roots):
        raise RuntimeError('Fixture body names changed across reset')
    for name, values in layout.items():
        pos, quat = np.asarray(values['pos']), np.asarray(values['quat'])
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all():
            raise RuntimeError('Invalid frozen fixture pose: ' + name)
        index = model.body_name2id(name)
        model.body_pos[index] = pos
        model.body_quat[index] = quat
