import io
from pathlib import Path
from types import SimpleNamespace
import pytest
from perceptbudget.gpu_railgate import confirmed_off,set_enabled
from scripts import control_gpu_railgate as helper


def test_confirmation_requires_both_hardware_flags():
    assert confirmed_off({'is_railgated':'yes','gpu_powered_on':'off'})
    assert not confirmed_off({'is_railgated':'yes','gpu_powered_on':'on'})
    assert not confirmed_off({'power/runtime_status':'suspended'})


def test_switch_is_only_zero_or_one(tmp_path):
    p=tmp_path/'railgate_enable';p.write_text('0')
    set_enabled('1',tmp_path);assert p.read_text()=='1'
    with pytest.raises(ValueError):set_enabled('force',tmp_path)
    assert p.read_text()=='1'


def test_root_protocol_restores_on_invalid_command(tmp_path,monkeypatch):
    class Directory:
        def mkdir(self,**kwargs):pass
        def is_symlink(self):return False
        def stat(self):return SimpleNamespace(st_uid=0,st_mode=0o40700)
        def __truediv__(self,name):return tmp_path/name
    value={'enabled':'0'};events=[]
    def write(v):
        if v not in ('0','1'):raise ValueError('Invalid command')
        value['enabled']=v
    monkeypatch.setattr(helper,'Path',lambda _:Directory())
    monkeypatch.setattr(helper.os,'geteuid',lambda:0)
    monkeypatch.setattr(helper.os,'umask',lambda _:0)
    monkeypatch.setattr(helper.signal,'signal',lambda *a:None)
    monkeypatch.setattr(helper,'state',lambda:{'railgate_enable':value['enabled']})
    monkeypatch.setattr(helper,'set_enabled',write)
    monkeypatch.setattr(helper,'emit',lambda status,**kw:events.append(status))
    monkeypatch.setattr(helper.sys,'argv',['control_gpu_railgate.py'])
    monkeypatch.setattr(helper.sys,'stdin',io.StringIO('1\nbad\n'))
    monkeypatch.setattr(helper.select,'select',lambda *a:([helper.sys.stdin],[],[]))
    with pytest.raises(ValueError):helper.main()
    assert value['enabled']=='0'
    assert not (tmp_path/'original.json').exists()
    assert events==['READY','APPLIED','RESTORED']
