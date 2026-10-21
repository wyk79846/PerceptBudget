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
             'audit_rtc_system.py', 'libero_tasks.py')
    for name in names:
        dump(tmp_path / 'scripts' / name, {})
    dump(tmp_path / 'config.json', {})
    dump(tmp_path / 'run.json', {})
    before = batch.provenance(tmp_path)['source_hashes']
    dump(tmp_path / 'scripts/analyze_m1_h50.py', {'offline': True})
    assert batch.provenance(tmp_path)['source_hashes'] == before


def repair_fixture(path, full=False):
    old = dict(source_commit='old', git_status='', reference='ref', reference_hash='same',
               frozen={'model': 'unchanged'}, source_hashes={
                   **{p: 'old' for p in batch.REPAIR_FILES}, 'config.json': 'unchanged',
                   'perceptbudget/rtc_guidance.py': 'unchanged'})
    current = copy.deepcopy(old)
    current.update(source_commit='repair-head')
    current['source_hashes'].update({p: 'new' for p in batch.REPAIR_FILES})
    chosen = batch.cells()[:141] if full else batch.PILOT + [
        ('libero_spatial', 7, q, 'actual_history') for q in (5, 10)]
    index = dict(provenance=old, cells={}, completed=len(chosen), status='stopped_invalid')
    for cell in chosen:
        name = batch.key(cell)
        parent = path / name
        worker = fixture_result(parent, cell)
        if name in batch.REPAIR_CELLS:
            dump(worker / f'rtc/q{cell[2]}_requests.json', [dict(prediction_id=7,
                 rtc_guidance=dict(guidance_enabled=True, variable_length_fallback=True))])
        audit_path = path / (name + '.audit.json')
        dump(audit_path, dict(passed=True))
        index['cells'][name] = dict(cell=list(cell), status='completed', parent=str(parent),
            attempts=[dict(status='completed', exit_code=0, log='original.log')],
            audit=str(audit_path), audit_sha256=batch.sha(audit_path),
            result=batch.validate(parent, cell, allow_guided_fallback=True))
    invalid = ('libero_object', 4, 5, 'actual_history')
    index['cells'][batch.key(invalid)] = dict(cell=list(invalid), status='invalid',
        attempts=[dict(status='invalid', exit_code=1, log='original-object4.log')])
    return index, current


def test_repair_retains_139_and_supersedes_only_two_without_mutating_old(tmp_path):
    index, current = repair_fixture(tmp_path, full=True)
    original = copy.deepcopy(index)
    migrated = batch.migrate_graph_coverage(index, current)
    assert index == original
    assert migrated['completed'] == 139
    assert {k for k, r in migrated['cells'].items() if r['status'] == 'pending'} == batch.REPAIR_CELLS
    assert len(migrated['superseded_records']) == 2
    for archived in migrated['superseded_records']:
        name = archived['key']
        assert archived['record'] == original['cells'][name]
        assert migrated['cells'][name]['attempts'] == original['cells'][name]['attempts']
        migrated['cells'][name]['attempts'].append({'new': True})
        assert archived['record'] == original['cells'][name]
    for name, record in original['cells'].items():
        if name not in batch.REPAIR_CELLS:
            assert migrated['cells'][name] == record
    epoch = migrated['runtime_repair_epochs'][0]
    assert epoch['provenance_before'] == original['provenance']
    assert epoch['provenance_after'] == current
    assert set(epoch['hashdiff']) == batch.REPAIR_FILES
    assert 'tokenizer exact full range at most 17' in epoch['reason']
    assert 'rule h50_state8_tokenizer_bounds_v1' in epoch['reason']
    assert 'ketchup bounds 645..661' in epoch['reason']
    assert 'real task text' in epoch['reason'] and 'spread 2, total span 16' in epoch['reason']
    assert 'max_lengths cap is 17' in epoch['reason']
    assert '33' not in epoch['reason']
    assert 'initial valid prefix +/-16' not in epoch['reason']
    with pytest.raises(ValueError, match='already accepted'):
        batch.migrate_graph_coverage(migrated, current)


def test_repair_rejects_other_runtime_config_reference_changes(tmp_path):
    index, current = repair_fixture(tmp_path)
    for field in ('frozen', 'reference', 'reference_hash'):
        bad = copy.deepcopy(current)
        bad[field] = 'changed'
        with pytest.raises(ValueError, match='cannot change'):
            batch.migrate_graph_coverage(index, bad)
    for name in ('config.json', 'perceptbudget/rtc_guidance.py', 'unlisted.py'):
        bad = copy.deepcopy(current)
        bad['source_hashes'][name] = 'changed'
        with pytest.raises(ValueError, match='non-permitted|add/remove'):
            batch.migrate_graph_coverage(index, bad)
    bad = copy.deepcopy(current)
    del bad['source_hashes']['scripts/diagnose_rtc_handoff.py']
    with pytest.raises(ValueError, match='add/remove'):
        batch.migrate_graph_coverage(index, bad)


def test_repair_rejects_unexpected_fallback_without_partial_migration(tmp_path):
    index, current = repair_fixture(tmp_path)
    original = copy.deepcopy(index)
    cell = batch.PILOT[1]
    record = index['cells'][batch.key(cell)]
    dump(Path(record['result']['run']) / 'rtc/q5_requests.json', [dict(prediction_id=3,
         rtc_guidance=dict(guidance_enabled=True, variable_length_fallback=True))])
    with pytest.raises(ValueError, match='exactly Spatial7'):
        batch.migrate_graph_coverage(index, current)
    assert index == original


@pytest.mark.parametrize('guided,fallback,mode,rejected', [
    (True, True, 'actual_history', True), (False, True, 'actual_history', False),
    (True, False, 'actual_history', False), (False, True, 'sync_no_rtc', False)])
def test_only_formal_guided_fallback_invalid(tmp_path, guided, fallback, mode, rejected):
    cell = ('libero_spatial', 0, 5, mode)
    worker = fixture_result(tmp_path, cell)
    dump(worker / 'rtc/q5_requests.json', [dict(prediction_id=1, rtc_guidance=dict(
        guidance_enabled=guided, variable_length_fallback=fallback))])
    if rejected:
        with pytest.raises(ValueError, match='variable_length_fallback'):
            batch.validate(tmp_path, cell)
    else:
        batch.validate(tmp_path, cell)


def test_new_guided_fallback_stops_batch(scheduler, monkeypatch):
    original = batch.execute

    def execute(cell, reference, log):
        result = original(cell, reference, log)
        if cell[3] == 'actual_history':
            parent = batch.find_parent(log)
            dump(parent / 'worker/rtc/q5_requests.json', [dict(prediction_id=1,
                 rtc_guidance=dict(guidance_enabled=True, variable_length_fallback=True))])
        return result

    monkeypatch.setattr(batch, 'execute', execute)
    assert run(scheduler) == 2 and len(scheduler[2]) == 2
    assert scheduler[1]['cells'][batch.key(batch.PILOT[1])]['status'] == 'invalid'


def test_repair_only_backs_up_and_never_launches(tmp_path, monkeypatch):
    path = tmp_path / batch.REPAIR_BATCH
    index, current = repair_fixture(path)
    dump(path / 'index.json', index)
    before_bytes = (path / 'index.json').read_bytes()
    monkeypatch.setattr(batch, 'provenance', lambda ref: current)
    monkeypatch.setattr(batch, 'run_batch', lambda *a, **k: pytest.fail('no GPU execution allowed'))
    assert batch.main(['--resume', str(path), '--accept-graph-coverage-repair', '--repair-only']) == 0
    migrated = batch.read(path / 'index.json')
    backup = Path(migrated['runtime_repair_epochs'][0]['original_index_backup'])
    assert batch.read(backup) == index
    assert migrated['provenance'] == current
    assert migrated['completed'] == 4
    # Subsequent resumes use the new runtime hashes strictly.
    bad = copy.deepcopy(current)
    bad['source_hashes']['scripts/diagnose_rtc_handoff.py'] = 'another-change'
    monkeypatch.setattr(batch, 'provenance', lambda ref: bad)
    with pytest.raises(ValueError, match='refusing resume'):
        batch.main(['--resume', str(path)])
    assert backup.exists() and before_bytes


def test_repair_resume_retries_pending_and_invalid_preserving_attempts(tmp_path, monkeypatch):
    index, current = repair_fixture(tmp_path)
    migrated = batch.migrate_graph_coverage(index, current)
    needed = [tuple(migrated['cells'][k]['cell']) for k in sorted(batch.REPAIR_CELLS)]
    invalid = ('libero_object', 4, 5, 'actual_history')
    monkeypatch.setattr(batch, 'cells', lambda: batch.PILOT + needed + [invalid])
    monkeypatch.setattr(batch, 'provenance', lambda ref: current)
    monkeypatch.setattr(batch, 'independent_audit', lambda run: dict(passed=True))
    calls = []

    def execute(cell, reference, log):
        calls.append(cell)
        parent = tmp_path / ('new-' + batch.key(cell))
        fixture_result(parent, cell)
        log.write_text('comparison_dir=' + str(parent) + '\n')
        return 0

    monkeypatch.setattr(batch, 'execute', execute)
    batch.run_batch(tmp_path, migrated, Path('/ref'), pilot_only=False, approved=True, min_free_gib=.00001)
    assert calls == needed + [invalid]
    for cell in calls:
        record = migrated['cells'][batch.key(cell)]
        assert len(record['attempts']) == 2
        assert record['attempts'][1]['source_epoch'] == 1
        assert record['attempts'][0] == index['cells'][batch.key(cell)]['attempts'][0]
