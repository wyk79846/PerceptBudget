import json
import sys

import pytest

from perceptbudget.ordinary_fast import choose,FEEDBACK_POLICY
from tests.test_three_frequency import ordinary_costs


def decision():
    return dict(decision_monotonic=1.,deadline=2.,margin_seconds=.025,latest_start=1.4,
        compute_profile='gpu_emc',service_seconds=.5,rtc_delay_steps=10,
        estimated_control_period_seconds=.05,reason='buffer',deadline_already_unreachable=False,
        start_constraint='service_deadline')


def test_energy_fast_does_not_resurrect_predictable_motion_phases():
    d=decision();args=dict(step=30,previous_mode='Full',phases=[dict(start_step=28,stop_step=35)],
        events=[dict(kind='turn',target_step=32)],costs=ordinary_costs(),fast_delay=5)
    assert choose(d,**args)['mode']=='Full'
    r=choose(d,**args,policy=FEEDBACK_POLICY)
    assert r['mode']=='Fast' and r['ordinary']
    assert all(d[k]==v for k,v in r['full_schedule'].items())


def test_energy_fast_keeps_gripper_margin_required_feedback_and_full_after_fast():
    d=decision();args=dict(step=30,previous_mode='Full',phases=[],events=[],
        costs=ordinary_costs(),fast_delay=5,policy=FEEDBACK_POLICY)
    assert choose(d,**args)['mode']=='Fast'
    assert choose(dict(d,reason='action_event'),**args)['mode']=='Full'
    assert choose(d,**dict(args,events=[dict(kind='gripper_open',target_step=34)]))['mode']=='Full'
    assert choose(d,**dict(args,previous_mode='Fast'))['mode']=='Full'
    assert choose(d,**dict(args,fast_delay=11))['mode']=='Full'


def test_idle_threshold_does_not_override_fast_pair_labels():
    from perceptbudget.device_quality import frequency_pair_views
    c=dict(m3_ordinary_fast_pair=True,m3_idle_min_ms=50,m3_three_frequency=True)
    assert frequency_pair_views(c,0)==('ordinary_full','ordinary_fast')
    assert frequency_pair_views(c,1)==('ordinary_fast','ordinary_full')


def test_energy_fast_cli_forces_request_only_images_and_keeps_frozen_controller(monkeypatch,capsys):
    from scripts import run_m4
    args=['run_m4.py','--acceptance-group','m3','--task-indices','0','--energy-refresh',
        '--ordinary-fast','--ordinary-fast-pair','--low-full','--three-frequency',
        '--idle-min-ms','50','--start-floor-ms','25','--dry-run']
    monkeypatch.setattr(sys,'argv',args)
    assert run_m4.main()==0
    r=json.loads(capsys.readouterr().out)
    assert r['request_only_images'] and r['lean_continue'] and r['ordinary_fast']
    assert r['feedback_policy']=='overlap_pacing_v2' and r['feedback_minimum_steps']==20
    assert r['H']==50 and r['N']==5 and r['max_refresh_interval']==50
    assert r['episodes']==2 and r['Fast']=='selective320_no_state'
    assert r['idle_min_ms']==50 and r['three_frequency']


def test_fast_capture_cannot_silently_revert_to_unused_inference_images(monkeypatch):
    from scripts import run_m4
    monkeypatch.setattr(sys,'argv',['run_m4.py','--acceptance-group','m3','--ordinary-fast',
        '--low-full','--three-frequency','--full-capture','--dry-run'])
    with pytest.raises(SystemExit):run_m4.main()
