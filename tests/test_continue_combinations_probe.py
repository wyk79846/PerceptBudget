import json
from unittest.mock import patch
from scripts import probe_continue_combinations as probe
from perceptbudget import device_clocks,device_continue


def test_dry_plan_has_gpu_fixed_and_one_variable_candidates(capsys):
    with patch('sys.argv',['probe','--dry-run']):
        assert probe.main()==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['model_loads']==1 and plan['maximum_windows']==10
    ref=probe.CANDIDATES['idle306']
    for candidate in probe.CANDIDATES.values():
        assert candidate[0]==306000000
        assert sum(a!=b for a,b in zip(candidate,ref))<=1


def test_wrapper_calls_original_not_patched_self_and_restores_globals(tmp_path):
    (tmp_path/'preflight.json').write_text(json.dumps({'accepted':['idle306','cpu1190low']}))
    prior_profiles=dict(device_clocks.PROFILES)
    prior_matrix=device_continue.MATRIX_PROFILES
    seen=[]
    def original(*args,**kwargs):
        assert device_continue.MATRIX_PROFILES==('idle306','cpu1190low')
        assert device_continue.REPEATS==2
        assert kwargs==dict(history_windows=True,history_points=(1.,),report_cadence_misses=True)
        seen.append(True)
        return {'windows':[],'summary':[],'status':'completed'}
    result=probe.screen(None,None,None,None,None,tmp_path,{},original_screen=original)
    assert seen==[True] and not result['production_defaults_changed']
    assert device_clocks.PROFILES==prior_profiles
    assert device_continue.MATRIX_PROFILES==prior_matrix
