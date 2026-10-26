import json
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.action_events import EventConfig, analyze_episode, detect_events


def action(x=1., y=0., grip=-1.):
    return [x, y, 0., 0., 0., 0., grip]


def test_gripper_hysteresis_ignores_jitter_and_labels_both_directions():
    values = [-1, -1, .05, .3, -.1, 1, 1, -.1, -1, -1]
    events = detect_events([action(grip=g) for g in values])
    grip = [e for e in events if e['kind'].startswith('gripper')]
    assert [(e['kind'], e['offset']) for e in grip] == [('gripper_close', 5), ('gripper_open', 8)]
    assert detect_events([action(grip=g) for g in values], start_offset=7) == [grip[-1]]


def test_motion_hints_use_translation_only_and_require_sustained_change():
    actions = [action(), action(), action(x=.4), action(x=.3), action(x=0,y=1), action(x=0,y=1)]
    events = detect_events(actions)
    assert [(e['kind'], e['offset']) for e in events] == [('slowdown', 2), ('turn', 4)]
    actions = [[0.,0.,0.,1.,0.,0.,-1.] for _ in range(5)]
    assert detect_events(actions) == []  # Rotation commands are not xyz motion.


def test_tiny_or_single_tick_turns_do_not_trigger():
    assert detect_events([action(), action(x=0,y=1), action()]) == []
    assert detect_events([action(x=.001), action(x=0,y=.001), action(x=0,y=.001)]) == []


@pytest.mark.parametrize('data', [[[0]*32], [[float('nan')]*7]])
def test_normalized_padded_or_nonfinite_actions_rejected(data):
    with pytest.raises(ValueError):
        detect_events(data)


def make_episode(path):
    (path/'rtc').mkdir(parents=True)
    acts = [action(), action(), action(grip=1), action(grip=1), action(grip=1)]
    req = [dict(prediction_id=0, actions=acts), dict(prediction_id=1, actions=acts)]
    (path/'rtc/q5_requests.json').write_text(json.dumps(req))
    metadata = dict(config_snapshot=dict(rtc_suite='libero_spatial', rtc_initial_control_period_seconds=.05))
    (path/'run.json').write_text(json.dumps(metadata))
    trace = [dict(event='prediction_committed', prediction_id=0, observation_step=0, commit_step=1, commit_monotonic=10.)]
    for i, a in enumerate(acts):
        trace.append(dict(event='control_step', step=i, q=5, task_id=0, action=a, applied_arm_gain=1.,
                          action_start_monotonic=10.+i*.07, action_end_monotonic=10.05+i*.07,
                          source=dict(prediction_id=0 if i<2 else 2), after={'contacts': []}))
    (path/'rtc/trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))


def test_offline_review_excludes_consumed_prefix_and_marks_overwritten_tail(tmp_path):
    make_episode(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report = analyze_episode(tmp_path)
    close = next(e for e in report['forecasts'] if e['kind']=='gripper_close')
    assert close['target_step']==2 and close['known_step']==1
    assert close['disposition']=='target_overwritten_or_underflow'
    assert close['command_step_error']==0
    assert close['predicted_lead_seconds']==pytest.approx(.05)  # No future tick leakage.
    assert report['summary']['uncommitted_requests']==1
    assert report['chunks'][0]['remaining_output_actions']==4
    assert all(p.read_bytes()==contents for p,contents in before.items())


def test_recording_session_uses_one_resident_runner_without_policy_changes(tmp_path):
    from perceptbudget.closed_loop import recording_session
    event_path = tmp_path/'events.json'
    report = dict(run='episode',suite='libero_spatial',task_id=0,q=5,control_steps=5,mode='record_only',
                  summary=dict(committed_chunks=1, forecast_counts={}))
    event_path.write_text(json.dumps(report))
    record = dict(episodes=[dict(action_events=str(event_path))],model_loads=1,suffix_loads=1)
    settings = dict(calibration_tasks=[['libero_spatial',0]], events={})
    with patch('perceptbudget.device_quality.resident_quality', return_value=record) as runner, \
         patch('perceptbudget.closed_loop.compiled_environments', return_value=nullcontext({})):
        result=recording_session({},settings,tmp_path,Path('gate'))
    assert runner.call_args.kwargs['record_only']
    assert runner.call_args.kwargs['task_plan']==[('libero_spatial',0)]
    assert not result['policy_changed'] and not result['clocks_changed']
    assert result['model_loads']==1


def test_recording_dry_run_never_spawns_worker(capsys):
    from scripts.run_m3 import main
    with patch('sys.argv',['run_m3','--dry-run']), patch('subprocess.run') as launch:
        assert main()==0
    d=json.loads(capsys.readouterr().out)
    assert len(d['tasks'])==3 and d['model_loads']==1
    assert not d['policy_changed'] and not d['clocks_changed']
    launch.assert_not_called()
