"""Serial M1 H50 batch. Default stops after four pilot cells for human review.

Resume full scan with --resume BATCH --pilot-approved after reviewing pilot.
Each subprocess is a diagnose parent (which owns guard/GPU locking). This
module imports only the standard library and never acquires the GPU lock.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/home/wyk/workspace/任务产出/perceptbudget/M1')
REFERENCE = OUTPUT / '20261020T035931.024165Z-openpi-rtc-actual-cfaef1e3'
SUITES = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300}
QS = (5, 10, 20, 30, 40)
MODES = ('sync_no_rtc', 'actual_history')
PILOT = [('libero_spatial', 0, 5, m) for m in MODES] + [
    ('libero_object', 0, 5, m) for m in MODES]
FROZEN = dict(H=50, N=5, seed=7, init_state_index=0, beta=5,
              checkpoint_profile='lerobot_h50_quantiles_v044',
              revision='b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd',
              exact_flash_attention=True, fused_norm=True, gemm_policy='autotune',
              calibrate_control_period=True, calibration_ticks=12,
              suites=SUITES, qs=QS, modes=MODES, ram_floor_mib=768)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(reference):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    # Hash live code as well as recording HEAD: shared worktrees may be dirty.
    entrypoints = ('diagnose_rtc_handoff.py', 'eval_rtc_openpi.py', 'probe_delay_buffer.py',
                   'eval_q.py', 'probe_rtc_step.py', 'guarded_run.py', 'run_m1_h50.py',
                   'audit_rtc_system.py', 'eval_seam_sync.py')
    files = sorted({ROOT / 'scripts' / name for name in entrypoints}
                   | set(ROOT.glob('perceptbudget/**/*.py'))
                   | set(ROOT.glob('perceptbudget/**/*.cu'))
                   | set(ROOT.glob('perceptbudget/**/*.cpp'))
                   | set(ROOT.glob('perceptbudget/**/*.h')))
    files = [p for p in files if p.is_file()]
    files += [ROOT / 'config.json']
    return dict(source_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                source_hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
                reference=str(reference), reference_hash=sha(reference / 'run.json'),
                frozen=FROZEN)


def cells():
    return PILOT + [c for s in SUITES for t in range(10) for q in QS
                    for m in MODES if (c := (s, t, q, m)) not in PILOT]


def key(cell):
    s, t, q, m = cell
    return f'{s}_task{t:02d}_q{q}_{m}'


def command(cell, reference):
    s, t, q, m = cell
    return [sys.executable, '-u', str(ROOT / 'scripts/diagnose_rtc_handoff.py'),
            '--reference-run', str(reference), '--suite', s, '--task-ids', str(t),
            '--conditions', m, '--q', str(q), '--beta', '5', '--horizon', '50',
            '--checkpoint-profile', 'h50', '--exact-flash-attention', '--fused-norm',
            '--gemm-policy', 'autotune', '--calibrate-control-period']


def validate(parent, cell):
    """Read small metadata only; return references plus initial-condition hashes."""
    s, t, q, m = cell
    summary = read(parent / 'summary.json')
    if summary.get('status') != 'completed' or summary.get('expected') != 1:
        raise ValueError('diagnose parent incomplete/invalid')
    records = summary['conditions']
    if len(records) != 1:
        raise ValueError('expected exactly one worker')
    item = records[0]
    run = Path(item['run'])
    result = read(run / 'summary.json')
    config = read(run / 'run.json')['config_snapshot']
    expected = dict(H=50, N=5, state_enabled=True, candidate_q=[q], rtc_suite=s,
                    rtc_beta=5, rtc_exact_flash_attention=True, rtc_fused_norm=True,
                    native_gemm_policy='autotune', rtc_calibrate_control_period=True,
                    checkpoint_profile=FROZEN['checkpoint_profile'],
                    model_id=FROZEN['checkpoint_profile'], state_prompt_capacity=200,
                    checkpoint_provenance_revision=FROZEN['revision'],
                    diagnostic_condition=m, execution_mode='diagnostic_' + m,
                    rtc_suffix_only_vjp=False)
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f'config mismatch: {name}')
    if config.get('controlled_delay_steps') is not None:
        raise ValueError('unexpected controlled delay')
    calibration = read(run / 'control_period_calibration.json')
    samples = calibration['samples_seconds']
    if (len(samples) != 12 or calibration.get('suite') != s or calibration.get('task_id') != t
            or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples)
            or not math.isfinite(calibration['median_seconds']) or calibration['median_seconds'] <= 0):
        raise ValueError('invalid same-environment control-period calibration')
    screen = config['screen']
    if any(screen.get(k) != v for k, v in dict(seed=7, init_state_index=0,
                                               suites=[s], task_ids=[t]).items()):
        raise ValueError('screen/initial conditions mismatch')
    episode = result['episode']
    if (item.get('exit_code') != 0 or item.get('status') != 'completed'
            or result.get('status') != 'completed'
            or result.get('diagnostic_condition') != m
            or result.get('suite') != s or result.get('task_id') != t
            or result.get('max_steps') != SUITES[s]
            or episode.get('q') != q or episode.get('task_id') != t
            or episode.get('status') not in ('success', 'timeout', 'failure')):
        raise ValueError('result identity/status mismatch')
    if (not isinstance(episode.get('steps'), int) or not 0 < episode['steps'] <= SUITES[s]
            or not isinstance(episode.get('real_seconds'), (int, float))
            or not math.isfinite(episode['real_seconds']) or episode['real_seconds'] <= 0):
        raise ValueError('invalid episode timing/steps')
    hashes = {k: episode[k] for k in ('initial_sim_state_sha256', 'initial_frames_sha256')}
    raw_state = episode['initial_raw_state']
    if not isinstance(raw_state, list) or len(raw_state) != 8 or not all(math.isfinite(x) for x in raw_state):
        raise ValueError('missing initial raw state')
    hashes['initial_raw_state_sha256'] = hashlib.sha256(struct.pack('<8d', *raw_state)).hexdigest()
    if not all(isinstance(v, str) and len(v) == 64 and
               all(c in '0123456789abcdef' for c in v) for v in hashes.values()):
        raise ValueError('missing initial-condition hashes')
    for p in (run / 'rtc/trace.jsonl', run / f'rtc/q{q}_requests.json'):
        if not p.is_file() or not p.stat().st_size:
            raise ValueError(f'missing result artifact: {p}')
    return dict(run=str(run), outcome=episode['status'], initial_hashes=hashes,
                metadata_hashes={str(p): sha(p) for p in
                                 (parent / 'summary.json', run / 'summary.json', run / 'run.json',
                                  run / 'control_period_calibration.json')})


def independent_audit(run):
    sys.path.insert(0, str(ROOT))
    from scripts.audit_rtc_system import audit
    return audit(run)


def execute(cell, reference, log):
    with log.open('w') as stream:
        return subprocess.call(command(cell, reference), cwd=ROOT,
                               stdout=stream, stderr=subprocess.STDOUT)


def find_parent(log):
    with log.open() as stream:
        for line in stream:
            if line.startswith('comparison_dir='):
                return Path(line.strip().split('=', 1)[1])
    raise ValueError('diagnose parent path absent from log')


def run_batch(batch, index, reference, *, pilot_only, approved, min_free_gib):
    # Validate existing completed cells before any new subprocess is launched.
    initial = {}

    def check_initial(cell, info):
        identity = f'{cell[0]}/{cell[1]}'
        value = info['initial_hashes']
        if identity in initial and initial[identity] != value:
            raise ValueError(f'initial sim/frame/raw-state mismatch: {identity}')
        initial[identity] = value

    valid = set()
    for cell in cells():
        record = index['cells'].get(key(cell), {})
        if record.get('status') != 'completed':
            continue
        try:
            info = validate(Path(record['parent']), cell)
            if info != record['result']:
                raise ValueError('saved result metadata changed')
            audit_path = Path(record['audit'])
            if not read(audit_path).get('passed') or sha(audit_path) != record['audit_sha256']:
                raise ValueError('independent audit missing/changed/failed')
            check_initial(cell, info)
            valid.add(key(cell))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record.update(status='invalid', error=str(exc))
            index['status'] = 'stopped_invalid'
            write(batch / 'index.json', index)
            return 2
    pilot_valid_at_entry = all(key(c) in valid for c in PILOT)
    if approved and not pilot_valid_at_entry:
        raise ValueError('--pilot-approved requires four previously completed, valid pilot cells')
    if approved:
        index['pilot_review'] = dict(approved=True, at=datetime.now(timezone.utc).isoformat())
    full = approved and not pilot_only
    for cell in cells() if full else PILOT:
        name = key(cell)
        if name in valid:
            continue
        record = index['cells'].setdefault(name, dict(cell=list(cell), attempts=[]))
        log = batch / f'{name}.attempt{len(record["attempts"]) + 1}.log'
        attempt = dict(log=str(log), command=command(cell, reference), status='running')
        record['attempts'].append(attempt)
        record['status'] = 'running'
        index['status'] = 'running'
        write(batch / 'index.json', index)
        try:
            if provenance(reference)['source_hashes'] != index['provenance']['source_hashes']:
                raise ValueError('source changed during batch')
            if shutil.disk_usage(batch).free < min_free_gib * 1024**3:
                raise ValueError('insufficient free disk space')
            code = execute(cell, reference, log)
            attempt['exit_code'] = code
            if code:
                raise ValueError(f'diagnose exited {code}; batch stopped without retry')
            if provenance(reference)['source_hashes'] != index['provenance']['source_hashes']:
                raise ValueError('source changed while cell was running')
            parent = find_parent(log)
            attempt['parent'] = record['parent'] = str(parent)
            info = validate(parent, cell)
            audited = independent_audit(Path(info['run']))
            audit_path = batch / f'{name}.attempt{len(record["attempts"])}.audit.json'
            write(audit_path, audited)
            record.update(audit=str(audit_path), audit_sha256=sha(audit_path))
            if audited.get('passed') is not True:
                raise ValueError('independent system audit failed')
            check_initial(cell, info)
            record.update(status='completed', result=info)
            attempt['status'] = 'completed'
            valid.add(name)
        except Exception as exc:
            record.update(status='invalid', error=str(exc))
            attempt.update(status='invalid', error=str(exc))
            index['status'] = 'stopped_invalid'
            write(batch / 'index.json', index)
            return 2
        index['completed'] = len(valid)
        write(batch / 'index.json', index)
    index['status'] = 'completed' if len(valid) == 300 else 'awaiting_pilot_review'
    write(batch / 'index.json', index)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--pilot-approved', action='store_true')
    parser.add_argument('--min-free-gib', type=float, default=10)
    args = parser.parse_args(argv)
    if args.pilot_approved and (not args.resume or args.pilot_only):
        parser.error('--pilot-approved requires --resume and excludes --pilot-only')
    if not math.isfinite(args.min_free_gib) or args.min_free_gib <= 0:
        parser.error('--min-free-gib must be positive and finite')
    reference = REFERENCE.resolve()
    batch = args.resume.resolve() if args.resume else OUTPUT / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-m1-h50-' + uuid.uuid4().hex[:8])
    if not args.resume:
        batch.mkdir(parents=True)
    # This is a batch-only lock, never the worker's .async_gpu.lock.
    with (batch / '.scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = provenance(reference)
        current = json.loads(json.dumps(current))
        if args.resume:
            index = read(batch / 'index.json')
            saved = index['provenance']
            if any(saved[k] != current[k] for k in
                   ('source_hashes', 'reference', 'reference_hash', 'frozen')):
                raise ValueError('frozen configuration/source/reference changed; refusing resume')
            index.setdefault('resume_history', []).append(dict(
                at=datetime.now(timezone.utc).isoformat(), source_commit=current['source_commit'],
                git_status=current['git_status']))
        else:
            index = dict(schema=1, expected=300, completed=0, provenance=current, cells={},
                         deferred=['M1.2 power', 'six perturbation episodes'])
            write(batch / 'index.json', index)
        print('batch_dir=' + str(batch), flush=True)
        return run_batch(batch, index, reference, pilot_only=args.pilot_only,
                         approved=args.pilot_approved, min_free_gib=args.min_free_gib)


if __name__ == '__main__':
    raise SystemExit(main())
