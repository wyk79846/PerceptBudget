"""CLI plans must retain both cameras and exactly the requested task count."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/eval_pooled_views.py'


def test_two_pool_factors_mean_twenty_distinct_paired_episodes():
    result=subprocess.run([sys.executable,str(SCRIPT),'--pool-factor','2','4','--dry-run'],
                          capture_output=True,text=True,check=True)
    r=json.loads(result.stdout)
    assert r['episodes']==20 and r['tasks']==list(range(10))
    assert r['num_views']==2 and r['model_lifetimes']==2
    assert r['encoded_visual_tokens']=={'2':128,'4':32}
    assert (r['q'],r['H'],r['N'],r['seed'],r['init'])==(5,50,5,7,0)


def test_duplicate_tasks_or_factors_are_rejected_before_cuda():
    for tail in (['--task-ids','0','0'],['--pool-factor','2','2']):
        r=subprocess.run([sys.executable,str(SCRIPT),'--pool-factor','2','--dry-run',*tail],
                         capture_output=True,text=True)
        assert r.returncode==2


def test_mixed_pool_plan_has_four_groups_and_forty_episodes():
    r=subprocess.run([sys.executable,str(SCRIPT),'--pool-factor','2','4',
        '--pool-view','agent','wrist','--dry-run'],capture_output=True,text=True,check=True)
    plan=json.loads(r.stdout)
    assert plan['episodes']==40 and plan['model_lifetimes']==4
    assert plan['mixed_visual_tokens']=={'2':320,'4':272}


def test_custom_quality_plan_rejects_unrelated_suites_and_duplicate_tasks(tmp_path):
    import pytest
    from perceptbudget.device_quality import resident_quality
    for plan in ([('libero_90',0)],[('libero_spatial',0)]*2):
        with pytest.raises(ValueError,match='Custom quality plan'):
            resident_quality({},tmp_path,'full',task_plan=plan)
