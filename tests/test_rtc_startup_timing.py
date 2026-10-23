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


def test_deterministic_workspace_is_set_before_first_cuda_initialization(monkeypatch):
    import sys
    from scripts import profile_rtc_startup as module
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    for name in ('MUJOCO_GL','PYOPENGL_PLATFORM','MUJOCO_EGL_DEVICE_ID'):
        monkeypatch.setenv(name,'before')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)))
    module.configure_runtime_environment()
    assert module.os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert module.os.environ['MUJOCO_GL']=='egl'
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True)))
    module.configure_runtime_environment()  # Already configured at context creation.
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    with pytest.raises(RuntimeError,match='restart process'):
        module.configure_runtime_environment()
    assert 'CUBLAS_WORKSPACE_CONFIG' not in module.os.environ


def test_gpu_timing_sidecar_is_saved_before_profiler_export_and_completed_after_parse(tmp_path):
    import json,time
    from contextlib import nullcontext
    from perceptbudget.gpu_start_trace import PairCollector
    events=[]
    marker='PB_MODEL_RECOVERY_0'
    class Prof:
        def start(self):events.append('start')
        def stop(self):events.append('stop')
        def export_chrome_trace(self,path):
            evidence=json.loads((tmp_path/(marker+'_timing.json')).read_text())
            assert evidence['status']=='pending_trace_parse'
            events.append('export')
            from pathlib import Path
            Path(path).write_text(json.dumps({'traceEvents':[
                {'name':marker,'cat':'user_annotation','ph':'X','ts':100.,'dur':10000.},
                {'name':'actual','cat':'kernel','ph':'X','ts':110.,'dur':1.}]}))
    def predict(*args,**kwargs):
        events.append('predict');time.sleep(.003)
        kwargs['request']['inference_ended_monotonic']=time.monotonic()
        return 'actions'
    torch=SimpleNamespace(profiler=SimpleNamespace(profile=lambda **kw:Prof(),
        ProfilerActivity=SimpleNamespace(CPU=0,CUDA=1),record_function=lambda name:nullcontext()))
    collector=PairCollector(SimpleNamespace(predict=predict),tmp_path,torch)
    collector.predict(request={})
    assert events==['start','predict']  # Armed before the pause, never after issue.
    request={'caller_request_monotonic':time.monotonic()}
    assert collector.predict(request=request)=='actions'
    assert events==['start','predict','predict','stop','export']
    saved=json.loads((tmp_path/(marker+'_timing.json')).read_text())
    assert saved['status']=='completed' and saved['timing']==request['gpu_start_timing']
    assert collector.prof is None
