import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from perceptbudget.watchdog import RequestWatchdog, InferenceTimeout, expired_request, inject_request_fault


def test_request_timeout_includes_drain_and_rejects_overlapping_requests(tmp_path):
    path=tmp_path/'request.json'
    guard=RequestWatchdog(path,3.)
    guard.arm(dict(prediction_id=1,observation_step=38,observation_monotonic=10.),tmp_path)
    assert expired_request(path,12.99) is None
    with pytest.raises(RuntimeError,match='overlap'):
        guard.arm(dict(prediction_id=2,observation_step=48,observation_monotonic=12.),tmp_path)
    with patch('perceptbudget.watchdog.time.monotonic',return_value=13.):
        with pytest.raises(InferenceTimeout):guard.check()
    assert expired_request(path,13.)['state']=='expired'


def test_timely_result_is_not_late_only_because_next_tick_is_late(tmp_path):
    path=tmp_path/'request.json';guard=RequestWatchdog(path,3.)
    guard.arm(dict(prediction_id=1,observation_step=38,observation_monotonic=10.),tmp_path)
    with patch('perceptbudget.watchdog.time.monotonic',return_value=14.):guard.complete(12.)
    assert guard.record is None and expired_request(path,14.) is None
    guard.arm(dict(prediction_id=2,observation_step=48,observation_monotonic=14.),tmp_path)
    with pytest.raises(InferenceTimeout):guard.complete(17.01)


@pytest.mark.parametrize('seconds',[0,-1,float('nan'),float('inf')])
def test_bad_request_timeout_rejected(tmp_path,seconds):
    with pytest.raises(ValueError):RequestWatchdog(tmp_path/'w.json',seconds)


@pytest.mark.parametrize('case,seconds',[('late',.25),('underflow',1.)])
def test_return_delay_only_affects_explicit_second_request(tmp_path,case,seconds):
    cfg=dict(m3_watchdog_case=case,m3_watchdog_path=str(tmp_path/'w.json'))
    request=dict(prediction_id=0,observation_step=0)
    with patch('perceptbudget.watchdog.time.sleep') as sleep:
        inject_request_fault(request,cfg,tmp_path)
        sleep.assert_not_called()
        request=dict(prediction_id=1,observation_step=38)
        inject_request_fault(request,cfg,tmp_path)
        sleep.assert_called_once_with(seconds)
    assert json.loads((tmp_path/'fault_injection.json').read_text())['case']==case


def test_outer_monitor_stops_real_unreturning_child(tmp_path):
    from scripts.profile_clock_screen import monitored_worker
    path=tmp_path/'request.json'
    code='''import json,pathlib,time
p=pathlib.Path(%r)
now=time.monotonic()
p.write_text(json.dumps(dict(state='armed',prediction_id=1,timeout_seconds=.2,deadline_monotonic=now+.2)))
while True:time.sleep(.05)
''' % str(path)
    started=time.monotonic()
    with patch('scripts.guarded_run.available_ram_bytes',return_value=2**40):
        with pytest.raises(InferenceTimeout):
            monitored_worker([sys.executable,'-c',code],None,request_watchdog=path)
    assert time.monotonic()-started<4.
    saved=json.loads((tmp_path/'outer_watchdog_timeout.json').read_text())
    assert saved['prediction_id']==1 and saved['detected_monotonic']>=saved['deadline_monotonic']


def test_watchdog_cli_uses_one_model_three_short_windows_and_dvfs(capsys):
    from scripts.run_m3 import main
    with patch('sys.argv',['m3','--mode','watchdog','--suite','libero_spatial','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes']==3 and plan['model_loads']==1 and plan['N']==5 and plan['state']
    assert plan['fault_checks']==['late','underflow','hang'] and plan['request_timeout_seconds']==3.
    assert plan['denoising_triggers']==[True]*3
    assert plan['clocks_changed'] and plan['action_offset_limits']==[50]
    launch.assert_not_called()


def test_watchdog_window_order_and_source_settings_are_explicit():
    from perceptbudget.closed_loop import comparison_variants
    assert comparison_variants({'m3_watchdog_checks':True})==('watchdog_late','watchdog_underflow','watchdog_hang')


@pytest.mark.parametrize('broken',[None,'late','underflow','wrong_hang','no_local_abort'])
def test_fault_suite_only_passes_with_all_independent_evidence(tmp_path,broken):
    from perceptbudget.watchdog import validate_fault_suite
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    episodes=[]
    for case in ('late','underflow'):
        run=tmp_path/case
        write(run/'summary.json',dict(episode=dict(status='fault_window_completed',underflow_steps=[50] if case=='underflow' and broken!='underflow' else [])))
        write(run/'rtc/q5_requests.json',[dict(watchdog_fault=dict(case=case),m3_timing=dict(arrival_lateness_seconds=0 if broken=='late' else .1))])
        (run/'rtc/runtime_clocks.jsonl').write_text('RESTORED\n')
        episodes.append(dict(view='watchdog_'+case,run=str(run),audit_passed=True))
    hang=tmp_path/'hang/rtc';hang.mkdir(parents=True)
    (hang/'runtime_clocks.jsonl').write_text('RESTORED\n')
    write(hang.parent/'run.json',{'status':'running'})
    write(tmp_path/'fault_injection.json',dict(case='late' if broken=='wrong_hang' else 'hang',prediction_id=1,episode_directory=str(hang)))
    write(tmp_path/'outer_watchdog_timeout.json',dict(prediction_id=1,episode_directory=str(hang),deadline_monotonic=10.,detected_monotonic=11.))
    write(tmp_path/'recording.json',dict(episodes=episodes,model_loads=1))
    if broken!='no_local_abort':write(hang/'watchdog_abort.json',dict(error='InferenceTimeout',request=dict(prediction_id=1)))
    if broken is None:
        result=validate_fault_suite(tmp_path)
        assert result['status']=='completed' and result['model_loads']==1
        assert json.loads((hang.parent/'run.json').read_text())['status']=='aborted_by_watchdog'
    else:
        with pytest.raises(RuntimeError,match='incomplete'):validate_fault_suite(tmp_path)
