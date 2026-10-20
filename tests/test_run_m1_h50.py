"""CPU-only scheduler tests: no diagnose/model/GPU process is launched."""
import copy
import json
from pathlib import Path

import pytest

from scripts import run_m1_h50 as batch


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_result(parent, cell, outcome='timeout', state='a'):
    s, t, q, m = cell
    run = parent / 'worker'
    cfg = dict(H=50, N=5, state_enabled=True, candidate_q=[q], rtc_suite=s,
               rtc_beta=5, rtc_exact_flash_attention=True, rtc_fused_norm=True,
               native_gemm_policy='autotune', rtc_calibrate_control_period=True,
               checkpoint_profile=batch.FROZEN['checkpoint_profile'],
               model_id=batch.FROZEN['checkpoint_profile'], state_prompt_capacity=200,
               checkpoint_provenance_revision=batch.FROZEN['revision'],
               diagnostic_condition=m, execution_mode='diagnostic_' + m, rtc_suffix_only_vjp=False,
               screen=dict(seed=7, init_state_index=0, suites=[s], task_ids=[t]))
    episode = dict(status=outcome, q=q, task_id=t, steps=20, real_seconds=2.,
                   initial_sim_state_sha256=state*64, initial_frames_sha256='b'*64,
                   initial_raw_state=[0.]*8)
    dump(run / 'run.json', dict(config_snapshot=cfg))
    dump(run / 'control_period_calibration.json', dict(suite=s, task_id=t,
         samples_seconds=[.05]*12, median_seconds=.05))
    dump(run / 'summary.json', dict(status='completed', suite=s, task_id=t,
                                   max_steps=batch.SUITES[s], diagnostic_condition=m, episode=episode))
    dump(parent / 'summary.json', dict(status='completed', expected=1,
         conditions=[dict(status='completed', exit_code=0, run=str(run))]))
    dump(run / 'rtc/trace.jsonl', dict(event='control_step'))
    dump(run / f'rtc/q{q}_requests.json', [{}])
    return run


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    index = dict(cells={}, provenance=dict(source_hashes={}), completed=0)
    calls = []
    monkeypatch.setattr(batch, 'provenance', lambda ref: dict(source_hashes={}))
    monkeypatch.setattr(batch, 'independent_audit', lambda run: dict(passed=True))

    def execute(cell, reference, log):
        calls.append(cell)
        parent = tmp_path / f'parent{len(calls)}'
        fixture_result(parent, cell)
        log.write_text('comparison_dir=' + str(parent) + '\n')
        return 0

    monkeypatch.setattr(batch, 'execute', execute)
    return tmp_path, index, calls


def run(scheduler, **kwargs):
    path, index, _ = scheduler
    return batch.run_batch(path, index, Path('/reference'), pilot_only=False,
                           approved=kwargs.pop('approved', False), min_free_gib=.00001, **kwargs)


def test_grid_and_parent_command():
    cells = batch.cells()
    assert len(cells) == len(set(cells)) == 300
    assert cells[:4] == batch.PILOT
    assert [(c[0], c[1]) for c in cells[:4]] == [
        ('libero_spatial', 0)]*2 + [('libero_object', 0)]*2
    for cell in cells:
        cmd = batch.command(cell, Path('/reference'))
        assert '--worker' not in cmd and 'guarded_run.py' not in ' '.join(cmd)
        for flag in ('--calibrate-control-period', '--exact-flash-attention', '--fused-norm'):
            assert flag in cmd
        assert cmd[cmd.index('--gemm-policy')+1] == 'autotune'
        assert cmd[cmd.index('--suite')+1] == cell[0]
        assert cmd[cmd.index('--q')+1] == str(cell[2])


def test_pilot_resume_then_full(scheduler):
    path, index, calls = scheduler
    assert run(scheduler) == 0
    assert len(calls) == 4 and index['status'] == 'awaiting_pilot_review'
    assert run(scheduler) == 0 and len(calls) == 4
    assert run(scheduler, approved=True) == 0
    assert len(calls) == 300 and index['completed'] == 300
    assert index['status'] == 'completed'
    assert run(scheduler, approved=True) == 0 and len(calls) == 300
    assert all(r['result']['outcome'] == 'timeout' for r in index['cells'].values())
    assert len(list(path.glob('*.audit.json'))) == 300


def test_cannot_approve_unrun_pilot(scheduler):
    with pytest.raises(ValueError, match='previously completed'):
        run(scheduler, approved=True)
    assert not scheduler[2]


@pytest.mark.parametrize('outcome', ['success', 'timeout', 'failure'])
def test_policy_outcomes_valid(tmp_path, outcome):
    fixture_result(tmp_path, batch.PILOT[0], outcome)
    assert batch.validate(tmp_path, batch.PILOT[0])['outcome'] == outcome


@pytest.mark.parametrize('field,value', [('H', 20), ('N', 10), ('state_enabled', False),
    ('native_gemm_policy', 'heuristic-first'), ('checkpoint_provenance_revision', 'old'),
    ('rtc_calibrate_control_period', False), ('rtc_suite', 'libero_goal'),
    ('rtc_exact_flash_attention', False), ('rtc_fused_norm', False)])
def test_reject_config_drift(tmp_path, field, value):
    worker = fixture_result(tmp_path, batch.PILOT[0])
    data = batch.read(worker / 'run.json')
    data['config_snapshot'][field] = value
    dump(worker / 'run.json', data)
    with pytest.raises(ValueError, match='config mismatch'):
        batch.validate(tmp_path, batch.PILOT[0])


def test_guard_exit_stops_and_explicit_resume_retries(scheduler, monkeypatch):
    original = batch.execute
    monkeypatch.setattr(batch, 'execute', lambda *a: 125)
    assert run(scheduler) == 2
    assert len(scheduler[1]['cells']) == 1
    monkeypatch.setattr(batch, 'execute', original)
    assert run(scheduler) == 0
    first = scheduler[1]['cells'][batch.key(batch.PILOT[0])]
    assert len(first['attempts']) == 2
    assert first['attempts'][0]['exit_code'] == 125


def test_failed_audit_stops(scheduler, monkeypatch):
    monkeypatch.setattr(batch, 'independent_audit', lambda run: dict(passed=False))
    assert run(scheduler) == 2 and len(scheduler[2]) == 1


def test_initial_hash_mismatch_stops(scheduler, monkeypatch):
    original = batch.validate

    def validate(parent, cell):
        info = original(parent, cell)
        if cell[3] == 'actual_history':
            info['initial_hashes']['initial_frames_sha256'] = 'c'*64
        return info

    monkeypatch.setattr(batch, 'validate', validate)
    assert run(scheduler) == 2 and len(scheduler[2]) == 2


def test_resume_rechecks_metadata_before_launch(scheduler):
    assert run(scheduler) == 0
    record = scheduler[1]['cells'][batch.key(batch.PILOT[0])]
    worker = Path(record['result']['run'])
    cfg = batch.read(worker / 'run.json')
    cfg['config_snapshot']['screen']['seed'] = 8
    dump(worker / 'run.json', cfg)
    assert run(scheduler, approved=True) == 2
    assert len(scheduler[2]) == 4


def test_main_rejects_frozen_change(tmp_path, monkeypatch):
    saved = dict(source_commit='old', source_hashes={}, reference='ref', reference_hash='x', frozen={})
    dump(tmp_path / 'index.json', dict(provenance=saved))
    current = copy.deepcopy(saved)
    current['source_hashes'] = {'runtime.py': 'changed'}
    monkeypatch.setattr(batch, 'provenance', lambda ref: current)
    with pytest.raises(ValueError, match='refusing resume'):
        batch.main(['--resume', str(tmp_path)])


def test_subprocess_uses_per_cell_log(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch.subprocess, 'call', lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    assert batch.execute(batch.PILOT[0], Path('/ref'), tmp_path / 'cell.log') == 0
    assert calls[0][1]['stderr'] == batch.subprocess.STDOUT
    assert calls[0][1]['cwd'] == batch.ROOT


def test_resume_allows_head_change_with_same_runtime_hashes(tmp_path, monkeypatch):
    saved = dict(source_commit='original', source_hashes={'runtime.py': 'same'},
                 reference='ref', reference_hash='x', frozen={}, git_status='')
    dump(tmp_path / 'index.json', dict(provenance=saved))
    current = dict(saved, source_commit='analysis-only-commit')
    monkeypatch.setattr(batch, 'provenance', lambda ref: current)
    seen = []
    monkeypatch.setattr(batch, 'run_batch', lambda path, index, ref, **kw: seen.append(index) or 0)
    assert batch.main(['--resume', str(tmp_path)]) == 0
    assert seen[0]['provenance']['source_commit'] == 'original'
    assert seen[0]['resume_history'][0]['source_commit'] == 'analysis-only-commit'


def test_offline_scripts_excluded_from_runtime_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'ROOT', tmp_path)
    monkeypatch.setattr(batch.subprocess, 'check_output', lambda *a, **k: 'head')
    names = ('diagnose_rtc_handoff.py', 'eval_rtc_openpi.py', 'probe_delay_buffer.py',
             'eval_q.py', 'probe_rtc_step.py', 'guarded_run.py', 'run_m1_h50.py',
             'audit_rtc_system.py', 'eval_seam_sync.py')
    for name in names:
        dump(tmp_path / 'scripts' / name, {})
    dump(tmp_path / 'config.json', {})
    dump(tmp_path / 'run.json', {})
    before = batch.provenance(tmp_path)['source_hashes']
    dump(tmp_path / 'scripts/analyze_m1_h50.py', {'offline': True})
    assert batch.provenance(tmp_path)['source_hashes'] == before
