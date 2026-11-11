from copy import deepcopy

from perceptbudget.stable_segments import PhysicalTeacher,online_features,phase_flags


def row(step,obj_z=.0,eef_z=.1,g=1,contact=True):
    state=[0.,0.,eef_z,0.,0.,0.,.02,-.02]
    contacts=([dict(geom1='item_1_g0',geom2='gripper_finger1',distance=-.001),
               dict(geom1='item_1_g0',geom2='gripper_finger2',distance=-.001)] if contact else [])
    snap=dict(raw_state=state,observations=dict(item_1_pos=[0.,0.,obj_z],robot0_joint_vel=[0.]*7,
        robot0_gripper_qvel=[0.,0.]),sim_time=(step+1)*.05,contacts=contacts)
    before=deepcopy(snap);before['sim_time']=step*.05
    return dict(step=step,action=[0.,0.,.1,0.,0.,0.,g],before=before,after=snap,
        action_start_monotonic=10.+step*.05,inference_active_at_action=False,
        late_hold=False,critical_wait_seconds=0.)


def test_privileged_object_and_contact_data_never_enter_online_features():
    a=row(4);b=deepcopy(a)
    b['after']['observations']['item_1_pos']=[100.,200.,300.]
    b['after']['contacts']=[];b['episode_outcome']='failure'
    assert online_features(a)==online_features(b)
    assert len(online_features(a))==25


def test_lifted_object_separation_is_local_and_normal_release_is_excluded():
    sequence=[row(0,g=-1),row(1,.025,.125),row(2,.045,.145),
              row(3,.035,.155,contact=False),row(4,.005,.165,contact=False)]
    teacher=PhysicalTeacher()
    seen=[teacher.observe(r) for r in sequence]
    assert seen[-1]['events'][0]['kind']=='held_object_separation'
    released=deepcopy(sequence)
    for r in released[3:]:r['action'][6]=-1.
    teacher=PhysicalTeacher()
    assert not any(teacher.observe(r)['events'] for r in released)


def test_continue_mask_uses_past_controller_decision_and_excludes_inflight_and_key_tick():
    r=row(10);d=dict(decision_monotonic=10.,wait_seconds=.8,start_now=False,critical_step=40,
        m3_fast_formula=dict(ordinary=False),reason='action_event')
    flags=phase_flags(r,d)
    assert flags['eligible'] and not flags['ordinary_declared']
    assert not phase_flags(dict(r,inference_active_at_action=True),d)['eligible']
    assert not phase_flags(r,dict(d,critical_step=11))['eligible']
    assert not phase_flags(r,dict(d,decision_monotonic=100.))['eligible']
