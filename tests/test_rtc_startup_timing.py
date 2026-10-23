from types import SimpleNamespace
import pytest
from perceptbudget.gpu_start_trace import timing_from_trace


def test_request_dispatch_and_gpu_computation_partition_total_without_double_count():
    trace={'traceEvents':[
        {'name':'r','cat':'user_annotation','ph':'X','ts':100.,'dur':500000.},
        {'name':'r','cat':'gpu_user_annotation','ph':'X','ts':130.,'dur':10.},
        {'name':'real','cat':'kernel','ph':'X','ts':130.,'dur':10.}]}
    r=timing_from_trace(trace,'r',cpu_bracket_ns=[1000000000,1000010000],issued=.999,ready=1.5)
    assert r['request_to_first_kernel_ms']==pytest.approx(1.035)
    assert r['request_to_first_kernel_bounds_ms']==pytest.approx([1.03,1.04])
    assert r['request_to_first_kernel_ms']+r['first_kernel_to_actions_ready_ms']==pytest.approx(501.)
    with pytest.raises(ValueError,match='request interval'):
        timing_from_trace(trace,'r',cpu_bracket_ns=[1000000000,1000010000],issued=1.1,ready=1.5)


def test_privilege_drop_is_irreversible_before_model_gate(monkeypatch):
    from scripts import profile_rtc_startup as module
    calls=[]
    monkeypatch.setenv('SUDO_UID','1001')
    for name in ('HOME','USER','LOGNAME'):monkeypatch.setenv(name,'before')
    monkeypatch.setattr(module.os,'geteuid',lambda:0)
    monkeypatch.setattr(module.pwd,'getpwuid',lambda uid:SimpleNamespace(pw_name='wyk',pw_gid=1001,pw_dir='/home/wyk'))
    monkeypatch.setattr(module.os,'initgroups',lambda *a:calls.append(('groups',a)))
    monkeypatch.setattr(module.os,'setresgid',lambda *a:calls.append(('gid',a)))
    monkeypatch.setattr(module.os,'setresuid',lambda *a:calls.append(('uid',a)))
    monkeypatch.setattr(module.os,'getresuid',lambda:(1001,1001,1001))
    monkeypatch.setattr(module,'identity',lambda:dict(CapEff='0',CapPrm='0',CapAmb='0'))
    module.drop_to_invoker()
    assert calls==[('groups',('wyk',1001)),('gid',(1001,1001,1001)),('uid',(1001,1001,1001))]
    monkeypatch.setattr(module,'identity',lambda:dict(CapEff='200000',CapPrm='0',CapAmb='0'))
    with pytest.raises(RuntimeError,match='retained capabilities'):module.drop_to_invoker()


def test_startup_dry_plan_does_not_require_root_or_load_model(monkeypatch,capsys):
    import json
    from scripts import profile_rtc_startup as module
    monkeypatch.setattr('sys.argv',['test','--dry-run'])
    monkeypatch.setattr(module,'initialize_and_drop',lambda:pytest.fail('Must not initialize GPU'))
    assert module.main()==0
    r=json.loads(capsys.readouterr().out)
    assert r['kernel_permission_gate_before_model'] and r['profiled_recovery_requests']==21
    assert r['profiler_armed_before_idle'] and not r['driver_settings_changed']


def test_clock_credentials_refreshed_after_drop_before_noninteractive_helpers(monkeypatch):
    from scripts import profile_rtc_startup as module
    calls=[]
    monkeypatch.setattr(module.os,'geteuid',lambda:1001)
    monkeypatch.setattr(module.subprocess,'run',lambda command,**kw:calls.append((command,kw)))
    module.authorize_clock_control()
    assert calls==[(['sudo','-v'],{'check':True}),(['sudo','-n','true'],{'check':True})]
    calls.clear()
    monkeypatch.setattr(module.os,'geteuid',lambda:0)
    with pytest.raises(RuntimeError,match='after privilege drop'):
        module.authorize_clock_control()
    assert not calls


def test_permission_only_does_not_request_clock_credentials(monkeypatch,capsys):
    from scripts import profile_rtc_startup as module
    monkeypatch.setattr('sys.argv',['test','--permission-only'])
    monkeypatch.setattr(module,'initialize_and_drop',lambda:(None,{'model_uid':1001}))
    monkeypatch.setattr(module,'authorize_clock_control',lambda:pytest.fail('Permission-only must not access clocks'))
    assert module.main()==0
