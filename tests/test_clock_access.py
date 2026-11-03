from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from perceptbudget import clock_access


def test_no_install_keeps_original_interactive_command(tmp_path,monkeypatch):
    monkeypatch.setattr(clock_access,'INSTALLED',tmp_path/'absent')
    command=clock_access.helper_command('runtime3','--seconds','120')
    assert command[:3]==['sudo','-n','/usr/bin/python3']
    assert command[4:]==['runtime3','--seconds','120']


def test_installed_helper_has_no_user_python_argument(monkeypatch):
    class Protected:
        def is_file(self):return True
        def is_symlink(self):return False
        def stat(self):return SimpleNamespace(st_uid=0,st_mode=0o100755)
        def __str__(self):return '/usr/local/sbin/perceptbudget-clocks'
    monkeypatch.setattr(clock_access,'INSTALLED',Protected())
    assert clock_access.helper_command('runtime3','--seconds','120')==[
        'sudo','-n','/usr/local/sbin/perceptbudget-clocks','runtime3','--seconds','120']


def test_refuse_user_owned_installed_helper(tmp_path,monkeypatch):
    p=tmp_path/'helper';p.write_text('untrusted')
    monkeypatch.setattr(clock_access,'INSTALLED',p)
    with pytest.raises(RuntimeError,match='protected'):clock_access.helper_command('inspect')


def test_nopasswd_health_skips_password_prompt():
    with patch.object(clock_access.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
        clock_access.authorize()
    assert run.call_count==1
    assert run.call_args.args[0]==['sudo','-n','/usr/bin/true']


def test_missing_health_grant_uses_existing_interactive_authorization():
    with patch.object(clock_access.subprocess,'run',side_effect=[SimpleNamespace(returncode=1),None]) as run:
        clock_access.authorize()
    assert run.call_args_list[-1].args[0]==['sudo','-v']


def test_install_rules_allow_only_fixed_protocols():
    script=clock_access.ROOT/'scripts/install_clock_sudo.sh'
    rules=subprocess.check_output(['/bin/bash',str(script),'--dry-run'],text=True)
    assert 'runtime --seconds 120' in rules and 'runtime3 --seconds 120' in rules
    assert '/usr/bin/true ""' in rules
    policy='\n'.join(line for line in rules.splitlines() if not line.startswith('#'))
    assert 'python' not in policy.lower() and '*' not in policy


def test_copied_clock_module_runs_isolated_without_user_repo_package(tmp_path):
    (tmp_path/'scripts').mkdir();(tmp_path/'perceptbudget').mkdir()
    (tmp_path/'perceptbudget/__init__.py').write_text('')
    for source,dest in [('scripts/control_device_clocks.py','scripts/control_device_clocks.py'),
                        ('perceptbudget/device_clocks.py','perceptbudget/device_clocks.py')]:
        shutil.copyfile(clock_access.ROOT/source,tmp_path/dest)
    output=subprocess.check_output(['/usr/bin/python3','-I',str(tmp_path/'scripts/control_device_clocks.py'),'--help'],text=True)
    assert 'runtime3' in output and 'restore' in output
