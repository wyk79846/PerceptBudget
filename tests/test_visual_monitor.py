import numpy as np

from perceptbudget.visual_monitor import WristSlipMonitor, continuation_eligible, cap_visual_refresh


def image_sequence(lost=False):
    rng = np.random.default_rng(83)
    background = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    foreground = rng.integers(0, 256, (48, 80), dtype=np.uint8)
    for i in range(7):
        image = np.roll(background, (i*3, i*2), axis=(0, 1))
        image[64:112, 88:168] = 0 if lost and i >= 4 else foreground
        state = [0., 0., i*.012, 0., 0., 0., .012, -.012]
        yield image, state, [0., 0., .1, 0., 0., 0., 1.], i*5


def test_attached_texture_survives_background_motion():
    monitor = WristSlipMonitor()
    rows = [monitor.observe(image, state, action, step=step)
            for image, state, action, step in image_sequence()]
    assert any(r['locked'] for r in rows)
    assert not any(r['alarm'] for r in rows)


def test_persistent_loss_requires_two_fresh_images_and_emits_once():
    monitor = WristSlipMonitor()
    rows = [monitor.observe(image, state, action, step=step)
            for image, state, action, step in image_sequence(lost=True)]
    assert rows[4]['loss_fraction'] == 1.
    assert not rows[4]['alarm']
    assert rows[5]['alarm']
    assert sum(r['alarm'] for r in rows) == 1


def test_expected_release_resets_tracking_without_alarm():
    monitor = WristSlipMonitor()
    rows = []
    for image, state, action, step in image_sequence(lost=True):
        if step >= 20:
            action[6] = -1.
        rows.append(monitor.observe(image, state, action, step=step))
    assert not any(r['alarm'] for r in rows)
    assert monitor.anchor is None


def test_inflight_and_due_refresh_are_not_continue_sampling():
    waiting = dict(start_now=False, wait_seconds=.3)
    assert continuation_eligible(waiting, inflight=False)
    assert not continuation_eligible(waiting, inflight=True)
    assert not continuation_eligible(dict(start_now=True, wait_seconds=0), inflight=False)


def test_release_between_images_invalidates_old_carried_reference():
    monitor = WristSlipMonitor()
    for image, state, action, step in image_sequence():
        monitor.observe(image, state, action, step=step)
    assert monitor.anchor is not None
    monitor.on_control([0.,0.,0.,0.,0.,0.,-1.])
    assert monitor.anchor is None and monitor.previous is None


def test_joint_visual_and_jaw_loss_can_alarm_without_waiting_another_image():
    monitor=WristSlipMonitor()
    samples=list(image_sequence(lost=True))
    for image,state,action,step in samples[:4]:
        monitor.observe(image,state,action,step=step)
    assert monitor.anchor is not None
    image,state,action,step=samples[4]
    state[6],state[7]=.001,-.001
    row=monitor.observe(image,state,action,step=step)
    assert row['alarm'] and row['reason']=='closed_jaw_collapse_and_texture_loss'


def test_gripper_closing_alone_without_image_loss_does_not_alarm():
    monitor=WristSlipMonitor()
    samples=list(image_sequence())
    for image,state,action,step in samples[:4]:
        monitor.observe(image,state,action,step=step)
    image,state,action,step=samples[4]
    state[6],state[7]=.001,-.001
    row=monitor.observe(image,state,action,step=step)
    assert not row['alarm']


def test_visual_response_preserves_earlier_mandatory_deadline():
    d=dict(estimated_control_period_seconds=.05,latest_start=9.8,
        first_refresh_cap=dict(latest_start=9.9),start_now=True)
    pending=dict(request_step=22,sample_id=0)
    result=cap_visual_refresh(d,pending,now=10.,step=20,current_profile='baseline')
    assert result['latest_start']==9.8
    assert result['first_refresh_cap']==d['first_refresh_cap']
    assert result['visual_refresh']==pending


def test_camera_sampling_does_not_insert_stale_other_view_into_policy():
    from types import SimpleNamespace
    from perceptbudget.visual_monitor import fresh_wrist_image
    calls=[]
    image=np.zeros((256,256,3),dtype=np.uint8)
    sensor=SimpleNamespace(is_enabled=lambda:True,update=lambda **kw:calls.append(kw))
    inner=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=1.)),
        _observables={'robot0_eye_in_hand_image':sensor},model_timestep=.002,_obs_cache={},
        _get_observations=lambda **kw:{'robot0_eye_in_hand_image':image,
                                     'agentview_image':np.full_like(image,255)})
    result,stamp=fresh_wrist_image(SimpleNamespace(env=inner))
    assert result.shape==(256,256,3) and calls[0]['force']
    assert not stamp['physics_advanced'] and not stamp['policy_observation_replaced']
    assert not np.shares_memory(result,image)


def test_resident_pair_toggles_only_monitor_and_keeps_both_real_policy_cameras():
    import copy
    from perceptbudget.device_quality import (frequency_pair_views,
        camera_view_for_condition, configure_frequency_variant)
    settings=dict(version='wrist_slip_v1',mode='refresh',interval=5)
    cfg=dict(m4_visual_monitor_pair=True,m4_visual_monitor_settings=settings,
        visual_monitor=settings.copy(),m3_three_frequency=True,m3_fast_critical=True)
    assert frequency_pair_views(cfg,0)==('visual_off','visual_on')
    assert frequency_pair_views(cfg,1)==('visual_on','visual_off')
    off=copy.deepcopy(cfg);on=copy.deepcopy(cfg)
    configure_frequency_variant(off,'visual_off')
    configure_frequency_variant(on,'visual_on')
    assert 'visual_monitor' not in off and on['visual_monitor']==settings
    assert off['m3_fast_critical']==on['m3_fast_critical']
    assert camera_view_for_condition('visual_off')==camera_view_for_condition('visual_on')=='full'


def test_pair_summary_counts_real_task_drain_without_inventing_a_terminal_hold(tmp_path):
    import json
    from perceptbudget.full_frequency_pairs import summarize_pairs
    episodes=[]
    for view in ('visual_off','visual_on'):
        folder=tmp_path/view;folder.mkdir();(folder/'rtc').mkdir()
        (folder/'run.json').write_text(json.dumps(dict(config_snapshot=dict(candidate_q=[5]))))
        (folder/'rtc/q5_requests.json').write_text(json.dumps([dict(prediction_id=0)]))
        episode=dict(real_seconds=1.,steps=20,power=dict(total_energy_mj=10000),
            including_drain_power=dict(total_energy_mj=11000),
            commits=[dict(prediction_id=0,commit_step=0)])
        (folder/'summary.json').write_text(json.dumps(dict(episode=episode)))
        episodes.append(dict(run=str(folder),suite='example',task_id=0,view=view,
                             outcome='success',steps=20))
    result=summarize_pairs(episodes,tmp_path,groups=('visual_off','visual_on'))
    assert all(r['task_plus_inference_j']==11. for r in result['rows'])
    assert all(r['terminal_low_j'] is None for r in result['rows'])
