"""Explicit, one-shot translation; scheduling and fresh observation are caller-owned."""

import numpy as np


def nudge_object(env, target, delta_xyz):
    """Translate akita_black_bowl_1 in world coordinates and return a JSON audit.

    Call once at formal control step=30, before observation and action execution.
    Accept a LIBERO ControlEnv or its domain. No reset, step, observation-cache
    refresh, collision resolution, or automatic trigger is performed. Quaternions
    use MuJoCo wxyz; velocities are the freejoint's six generalized velocities.
    """
    delta = np.asarray(delta_xyz, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError('delta_xyz must be a finite vector of shape (3,)')
    if not isinstance(target, str) or target != 'akita_black_bowl_1':
        raise ValueError('Only explicit target akita_black_bowl_1 is supported')
    domain = env if hasattr(env, 'objects_dict') else env.env
    if target not in domain.objects_dict or target in domain.fixtures_dict:
        raise ValueError('Target must be a registered movable object')
    obj = domain.objects_dict[target]
    sim = domain.sim
    model, data = sim.model, sim.data
    body = obj.root_body
    body_id = model.body_name2id(body)
    if body_id <= 0 or model.body_parentid[body_id] != 0:
        raise ValueError('Target body must be a direct child of world')
    if model.body_jntnum[body_id] != 1:
        raise ValueError('Target body must have exactly one freejoint')
    joint_id = int(model.body_jntadr[body_id])
    joint = model.joint_id2name(joint_id)
    if (joint not in obj.joints or model.joint_name2id(joint) != joint_id
            or model.jnt_bodyid[joint_id] != body_id
            or model.jnt_type[joint_id] != 0):  # mjJNT_FREE = 0 (mjmodel.h)
        raise ValueError('Object/body/freejoint mapping mismatch')
    qaddr = model.get_joint_qpos_addr(joint)
    vaddr = model.get_joint_qvel_addr(joint)
    for addr, width, size in ((qaddr, 7, data.qpos.size),
                              (vaddr, 6, data.qvel.size)):
        if (not isinstance(addr, tuple) or len(addr) != 2
                or addr[0] < 0 or addr[1] - addr[0] != width or addr[1] > size):
            raise ValueError('Invalid freejoint address range')
    qs, vs = slice(*qaddr), slice(*vaddr)
    before_q, before_v = data.qpos[qs].copy(), data.qvel[vs].copy()
    before_time = float(data.time)
    after_pos = before_q[:3] + delta
    if not all(np.isfinite(x).all() for x in
               (before_q, before_v, before_time, after_pos)):
        raise ValueError('Nonfinite joint state, time, or translated position')
    data.qpos[qaddr[0]:qaddr[0] + 3] = after_pos
    sim.forward()
    after_q, after_v = data.qpos[qs].copy(), data.qvel[vs].copy()
    after_time = float(data.time)
    if (after_time != before_time or not np.array_equal(after_q[3:], before_q[3:])
            or not np.array_equal(after_v, before_v)
            or not np.array_equal(after_q[:3], after_pos)):
        raise RuntimeError('forward violated translation invariants; stop episode')
    return {
        'target': target, 'body_name': body, 'body_id': int(body_id),
        'joint_name': joint, 'joint_id': joint_id,
        'before_pos': before_q[:3].tolist(), 'after_pos': after_q[:3].tolist(),
        'before_quat': before_q[3:].tolist(), 'after_quat': after_q[3:].tolist(),
        'before_vel': before_v.tolist(), 'after_vel': after_v.tolist(),
        'requested_delta': delta.tolist(),
        'actual_delta': (after_q[:3] - before_q[:3]).tolist(),
        'simtime_before': before_time, 'simtime_after': after_time,
    }



TARGET = 'akita_black_bowl_1'

def make_hook(enabled, events):
    def hook(env, observation, step):
        if not enabled or step != 30:
            return observation, None
        if events:
            raise RuntimeError('Disturbance must occur exactly once')
        event = nudge_object(env, TARGET, (.02, 0., 0.))
        observation = env.env._get_observations(force_update=True)
        observed = observation[TARGET + '_pos']
        if any(abs(float(x)-y) > 1e-8 for x, y in zip(observed, event['after_pos'])):
            raise RuntimeError('Object observation cache did not refresh after disturbance')
        event.update(step=step, timing='before observation and action; sim time unchanged',
                     refreshed_observed_target_pos=observed.tolist())
        events.append(event)
        return observation, event
    return hook
