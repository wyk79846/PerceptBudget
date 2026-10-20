import json
from pathlib import Path


def test_five_fresh_pairs_alternate_and_audit_only_candidate(tmp_path,monkeypatch):
    from scripts import compare_rtc_attention_five as batch
    parent=tmp_path/'batch'
    parent.mkdir()
    monkeypatch.setattr(batch,'load_config',lambda **kw:{'output_root':str(tmp_path)})
    monkeypatch.setattr(batch,'create_run',lambda *a,**k:parent)
    monkeypatch.setattr(batch,'audit',lambda run:{'passed':True,'checks':1})
    monkeypatch.setattr(batch.base,'_save_run_status',lambda *a,**k:None)
    commands=[]

    def child(command,**kwargs):
        commands.append(command)
        task=int(command[command.index('--task-id')+1])
        dest=Path(command[command.index('--parent')+1])
        run=dest/'run'
        run.mkdir()
        e=dict(status='success',steps=78,real_seconds=6.,initial_sim_state_sha256=f'sim{task}',initial_frames_sha256=f'frame{task}')
        (dest/'actual_history.json').write_text(json.dumps(dict(status='completed',run=str(run),episode=e,steady_inference_median_ms=600.)))
        return 0

    monkeypatch.setattr(batch.subprocess,'call',child)
    monkeypatch.setattr('sys.argv',['compare'])
    assert batch.main()==0
    assert len(commands)==10
    assert sum('--audit-attention-inputs' in c for c in commands)==5
    assert ['flash' if '--exact-flash-attention' in c else 'original' for c in commands]==[
        'original','flash','flash','original','original','flash','flash','original','original','flash']
    summary=json.loads((parent/'summary.json').read_text())
    assert summary['completed']==10 and summary['successes']=={'original':5,'flash':5}
