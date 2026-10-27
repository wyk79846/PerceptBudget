import os
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from perceptbudget.device_clocks import runtime_trace


def test_runtime_protocol_accepts_serial_high_low_and_stop():
    read,write=os.pipe()
    applied=threading.Event()
    events=[]
    errors=[]
    control=SimpleNamespace(actual=lambda:{'clock':1})
    def emit(event):
        events.append(event)
        if event['status']=='APPLIED':applied.set()
    def run():
        try:runtime_trace(control,emit,read,seconds=2)
        except BaseException as e:errors.append(e)
    with patch('perceptbudget.device_clocks.observed_switch',return_value={'observed_match_monotonic':1.}):
        thread=threading.Thread(target=run);thread.start()
        try:
            os.write(write,b'LOW\n');assert applied.wait(1);applied.clear()
            os.write(write,b'HIGH\n');assert applied.wait(1)
            os.write(write,b'STOP\n');thread.join(2)
            assert not thread.is_alive() and not errors
        finally:
            os.close(write);thread.join(3);os.close(read)
    changes=[e for e in events if e['status']=='APPLIED']
    assert [e['profile'] for e in changes]==['gpu_emc','baseline']
    assert [e['request_index'] for e in changes]==[0,1]
    assert events[-1]['status']=='STOPPED'


@pytest.mark.parametrize('payload',[b'REBOOT\n',b'LOW\nHIGH\n',b'x'*33])
def test_runtime_rejects_unknown_batched_or_oversize_commands(payload):
    read,write=os.pipe()
    try:
        os.write(write,payload)
        with pytest.raises(ValueError):runtime_trace(SimpleNamespace(actual=lambda:{}),lambda e:None,read,seconds=1)
    finally:os.close(read);os.close(write)


def test_runtime_writer_exit_returns_so_owner_can_restore():
    read,write=os.pipe();os.close(write)
    try:runtime_trace(SimpleNamespace(actual=lambda:{}),lambda e:None,read,seconds=1)
    finally:os.close(read)


def test_runtime_expiry_is_bounded():
    read,write=os.pipe()
    try:
        with pytest.raises(TimeoutError):
            runtime_trace(SimpleNamespace(actual=lambda:{}),lambda e:None,read,seconds=1)
    finally:os.close(read);os.close(write)


def test_dvfs_plan_keeps_root_model_off_and_uses_one_task(capsys):
    import json
    from scripts.run_m3 import main
    with patch('sys.argv',['m3','--mode','deadline-dvfs','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    d=json.loads(capsys.readouterr().out)
    assert d['clocks_changed'] and d['policy_changed']
    assert len(d['tasks'])==1 and d['model_loads']==1
    launch.assert_not_called()


@pytest.mark.parametrize('failed',[False,True])
def test_clock_preflight_restores_before_model_load_even_on_failure(tmp_path,failed):
    import json
    from perceptbudget.closed_loop import clock_preflight
    closed=[]
    class Clock:
        pending=None
        last_change={}
        def __init__(self,*a):pass
        def __enter__(self):return self
        def switch(self,profile):
            if failed:raise RuntimeError('frequency unavailable')
            self.last_change={'profile':profile}
        def close(self):closed.append(True)
    with patch('perceptbudget.closed_loop.RuntimeClocks',Clock):
        if failed:
            with pytest.raises(RuntimeError,match='frequency unavailable'):clock_preflight(tmp_path)
        else:assert clock_preflight(tmp_path)['status']=='completed'
    d=json.loads((tmp_path/'clock_preflight.json').read_text())
    assert d['restored'] and not d['model_loaded'] and closed==[True]


def test_variance_dvfs_plan_is_three_h50_tasks_without_sudo_or_model_load(capsys):
    import json
    from scripts.run_m3 import main
    with patch('sys.argv',['m3','--mode','trigger-comparison','--strategies','variance_only',
                           '--device-dvfs','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['episodes']==3 and plan['model_loads']==1
    assert {t[0] for t in plan['tasks']}=={'libero_spatial','libero_object','libero_goal'}
    assert plan['clocks_changed'] and plan['action_offset_limits']==[50]
    launch.assert_not_called()


@pytest.mark.parametrize('device_args,episode_count,strategy',[
    (['--device-dvfs'],3,'variance_only'),
    (['--compare-device','--trigger-suite','libero_spatial'],2,'variance_only'),
    (['--device-dvfs','--compare-event-dedup','--trigger-suite','libero_10','--trigger-task-id','1'],2,'events_and_variance'),
])
def test_variance_dvfs_preflights_before_worker_and_forwards_clock_flag(tmp_path,device_args,episode_count,strategy):
    import json
    from pathlib import Path
    from scripts.run_m3 import main
    from perceptbudget.experiment import load_config
    cfg=load_config(repo_root=Path(__file__).resolve().parents[1]);cfg['output_root']=str(tmp_path)
    run=tmp_path/'M3'/'fake';run.mkdir(parents=True)
    (run/'run.json').write_text(json.dumps({'status':'created'}))
    order=[];commands=[]
    def preflight(directory):
        assert directory==run
        order.append('preflight')
    def worker(command,log):
        order.append('worker');commands.append(command)
        (run/'summary.json').write_text(json.dumps({'status':'completed','episodes':[{}]*episode_count}))
        return 0
    with patch('sys.argv',['m3','--mode','trigger-comparison','--strategies',strategy,'--no-age-limit',*device_args]), \
         patch('perceptbudget.experiment.load_config',return_value=cfg), \
         patch('perceptbudget.experiment.create_run',return_value=run), \
         patch('subprocess.run') as sudo, \
         patch('perceptbudget.closed_loop.clock_preflight',side_effect=preflight), \
         patch('scripts.profile_clock_screen.monitored_worker',side_effect=worker):
        assert main()==0
    sudo.assert_called_once_with(['sudo','-v'],check=True)
    assert order==['preflight','worker']
    command=commands[0]
    assert command[0]==cfg['python_executable'] and device_args[0] in command and '--no-age-limit' in command
    assert command[command.index('--strategies')+1]==strategy
    if '--compare-event-dedup' in device_args:
        assert '--compare-event-dedup' in command
        assert command[command.index('--trigger-task-id')+1]=='1'
    assert not any('guarded_run.py' in part for part in command)
