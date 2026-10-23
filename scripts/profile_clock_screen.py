#!/usr/bin/env python3
"""Four candidate clock points; one Full model load; explicit sudo helper only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def clock_preflight(run, profiles, summary):
    """Reject unavailable matrix clocks BEFORE expensive model loading."""
    summary['clock_preflight'] = []
    for profile in profiles:
        path = run/(profile+'_preflight.log')
        command = ['sudo', '-n', '/usr/bin/python3', str(ROOT/'scripts/control_device_clocks.py'),
                   'lease', '--profile', profile, '--seconds', '1', '--trace']
        with path.open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        restored = 'RESTORED' in path.read_text().splitlines()
        summary['clock_preflight'].append(dict(profile=profile, exit_code=code,
                                               restored=restored, log=str(path)))
        if code or not restored:
            raise RuntimeError(f'Clock preflight failed for {profile}; restored={restored}; '
                               f'NO MODEL LOADED; inspect {path}')


def monitored_worker(command, out):
    # Keep the user's controlling terminal/session for sudo's tty-bound cache.
    # Do not use guarded_run's setsid: that would lose this authorization.
    # Never send group signals: a finite privileged lease may be a descendant.
    from scripts.guarded_run import available_ram_bytes, RAM_FLOOR_BYTES
    if available_ram_bytes() < RAM_FLOOR_BYTES:
        raise RuntimeError('Available RAM below 768MiB; not starting worker')
    child = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
    try:
        while child.poll() is None:
            if available_ram_bytes() < RAM_FLOOR_BYTES:
                raise RuntimeError('RAM protection stopped worker; wait for finite root helper and verify restoration')
            time.sleep(.1)
        return child.returncode
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    from perceptbudget.baseline import validate
    from perceptbudget.experiment import load_config, create_run, _write_json_atomically
    from perceptbudget.device_clocks import PROFILES, SCREEN_PROFILES, MATRIX_PROFILES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--transitions', action='store_true', help='Resident static combination holds and transition cost')
    modes.add_argument('--recovery', action='store_true', help='Short/long idle recovery followed by prepared-input Full RTC')
    modes.add_argument('--matrix', action='store_true', help='Nine Full cost cells, three candidate clock combinations by three idle states')
    modes.add_argument('--continue-control', action='store_true', help='Fifteen bounded live LIBERO Continue windows, five fresh actions each')
    modes.add_argument('--continue-camera-compare', action='store_true', help='Resident paired camera-on/off Continue at three clock points')
    modes.add_argument('--full-service', action='store_true', help='Five high-frequency Full services including live sensor acquisition and queue preparation')
    modes.add_argument('--fast-clocks', action='store_true', help='Old-KV expert-only fixed-input RTC, three clock points, five samples each')
    modes.add_argument('--visual-clocks', action='store_true', help='Single wrist camera context plus RTC at three clock points')
    modes.add_argument('--pooled-clocks', action='store_true', help='Dual-view factor2 token pooling context plus RTC at three clock points')
    modes.add_argument('--selected-fast-clocks', action='store_true', help='Selected Fast: external pool2, full wrist, no state; three clocks, five requests each')
    parser.add_argument('--idle-recovery',action='store_true',help='With selected Fast only: three clocks by hot/0.2s/1s histories, five requests each')
    parser.add_argument('--service-profiles', nargs='+', choices=MATRIX_PROFILES,
                        help='Full service clock points; five paired requests per point')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.idle_recovery and not args.selected_fast_clocks:
        parser.error('--idle-recovery requires --selected-fast-clocks')
    if args.service_profiles and not args.full_service:
        parser.error('--service-profiles requires --full-service')
    from perceptbudget.device_service import validate_profiles
    service_profiles = validate_profiles(args.service_profiles or ('baseline',))
    if os.geteuid() == 0:
        parser.error('Run model as ordinary user; only clock helper uses sudo')
    config = load_config(repo_root=ROOT)
    validate(config)
    source = Path(config['output_root'])/'M1/20261021T081454.319874Z-rtc-diag-actual_history-7444fb8d'
    if args.dry_run:
        if args.selected_fast_clocks:
            print(json.dumps(dict(model_loads=1,profiles=MATRIX_PROFILES,samples=45 if args.idle_recovery else 15,
                num_views=2,pool_factor=2,pooled_view='agent',visual_tokens=320,H=50,N=5,
                state=False,full_preparation_state=True,service_measurement=False,
                context_recomputed=True,static_resident_hold=not args.idle_recovery,
                idle_histories_seconds=[0,.2,1.] if args.idle_recovery else None,
                recovery_matrix=args.idle_recovery)))
            return 0
        if args.continue_camera_compare:
            print(json.dumps(dict(model_loads=1, profiles=MATRIX_PROFILES, windows=30,
                max_control_steps=150, repeats_per_camera_mode=5, step_period_seconds=.07,
                fresh_images_after_both=True, inference_during_continue=False,
                production_baseline_changed=False)))
            return 0
        if args.pooled_clocks:
            print(json.dumps(dict(model_loads=1, profiles=MATRIX_PROFILES, samples=15,
                num_views=2, pool_factor=2, pooled_view='both', visual_tokens=128, H=50, N=5,
                service_measurement=False, context_recomputed=True)))
            return 0
        if args.visual_clocks:
            print(json.dumps(dict(model_loads=1, profiles=MATRIX_PROFILES, samples=15,
                num_views=1, view='wrist', visual_tokens=256, H=50, N=5,
                service_measurement=False, context_recomputed=True)))
            return 0
        if args.fast_clocks:
            print(json.dumps(dict(model_loads=1, profiles=MATRIX_PROFILES, samples=15,
                fresh_full_before_each=True, mode='expert_only', H=50, N=5,
                service_measurement=False, context_and_state_stale=True)))
            return 0
        if args.full_service:
            print(json.dumps(dict(model_loads=1, services=5*len(service_profiles), profiles=service_profiles, H=50, N=5,
                                  state=True, live_observation=True, observation_step=5,
                                  consumer_steps_during_service=0, runtime_commit_reused=True,
                                  extra_idle_scan=False)))
            return 0
        if args.continue_control:
            from perceptbudget.device_clocks import MATRIX_PROFILES
            print(json.dumps(dict(profiles=MATRIX_PROFILES, model_loads=1, H=50, N=5,
                                  state=True, suite='libero_spatial', task_id=0, seed=7, init=0,
                                  repeats=5, committed_steps=5, windows=15, max_control_steps=75,
                                  inference_during_continue=False, steady_hold_claim=False)))
            return 0
        if args.matrix:
            from perceptbudget.device_clock_recovery import MATRIX_PLAN
            print(json.dumps(dict(plan=MATRIX_PLAN, model_loads=1, H=50, N=5, state=True,
                                  requests_per_condition=5, total_requests=45, full_cells=9,
                                  changes_power_mode=False, live_observation=False)))
            return 0
        if args.recovery:
            from perceptbudget.device_clock_recovery import PLAN
            print(json.dumps(dict(plan=PLAN, model_loads=1, H=50, N=5, state=True,
                                  requests_per_condition=5, total_requests=30,
                                  changes_power_mode=False, live_observation=False)))
            return 0
        print(json.dumps(dict(profiles={p: PROFILES[p] for p in (
                              ('baseline', 'gpu_emc', 'gpu_emc_cpu') if args.transitions else SCREEN_PROFILES)},
                              transitions=args.transitions, cycles_per_condition=5 if args.transitions else None,
                              model_loads=1, requests_per_profile=0 if args.transitions else 5,
                              static_hold_seconds=3, H=50, N=5, views=2, state=True,
                              changes_clocks=True, changes_power_mode=False,
                              source_run=str(source)), ensure_ascii=False))
        return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent != (Path(config['output_root'])/'M2').resolve():
            parser.error('Worker requires M2 output directory')
        # Check the inherited tty credential before loading any weights.
        subprocess.run(['sudo', '-n', 'true'], check=True)
        from perceptbudget.device_profile import fixed_view_probe
        fixed_view_probe(config, args.run_dir, source, 'single' if args.visual_clocks else 'full', clock_screen=True,
                         clock_transitions=args.transitions, clock_recovery=args.recovery,
                         clock_matrix=args.matrix or args.idle_recovery, continue_control=args.continue_control or args.continue_camera_compare,
                         full_service=args.full_service, service_profiles=service_profiles,
                         fast_clocks=args.fast_clocks, visual_clocks=args.visual_clocks,
                         pooled_clocks=args.pooled_clocks, camera_compare=args.continue_camera_compare,
                         selected_fast_clocks=args.selected_fast_clocks)
        return 0
    # Fail before loading weights if no interactive terminal/authorization.
    subprocess.run(['sudo', '-v'], check=True)
    if args.selected_fast_clocks:
        config=dict(config,m2_selected_fast_clocks=True,m2_fast_no_state=True,
                    m2_selected_fast_geometry='external64_wrist256',m2_selected_fast_recovery=args.idle_recovery)
    purpose = 'selected-fast-recovery' if args.idle_recovery else 'selected-fast-clock-screen' if args.selected_fast_clocks else 'continue-camera-compare' if args.continue_camera_compare else 'pooled-clock-screen' if args.pooled_clocks else 'visual-clock-screen' if args.visual_clocks else 'fast-clock-screen' if args.fast_clocks else 'full-service' if args.full_service else 'continue-control' if args.continue_control else 'clock-matrix' if args.matrix else 'clock-recovery' if args.recovery else 'clock-transitions' if args.transitions else 'clock-screen'
    run = create_run(config, 'M2', purpose, repo_root=ROOT)
    print('run_dir='+str(run), flush=True)
    summary = dict(status='running', source_run=str(source), scope='four candidate fixed-input screen')
    try:
        with (Path(config['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.matrix or args.continue_control or args.continue_camera_compare or args.fast_clocks or args.visual_clocks or args.pooled_clocks or args.selected_fast_clocks:
                from perceptbudget.device_clocks import MATRIX_PROFILES
                clock_preflight(run, MATRIX_PROFILES, summary)
            elif args.full_service:
                clock_preflight(run, service_profiles, summary)
            command = [config['python_executable'], '-u', str(Path(__file__).resolve()),
                       '--worker', '--run-dir', str(run)]
            if args.transitions:
                command.append('--transitions')
            if args.recovery:
                command.append('--recovery')
            if args.matrix:
                command.append('--matrix')
            if args.continue_control:
                command.append('--continue-control')
            if args.continue_camera_compare:
                command.append('--continue-camera-compare')
            if args.full_service:
                command.append('--full-service')
                command.extend(['--service-profiles', *service_profiles])
            if args.fast_clocks:
                command.append('--fast-clocks')
            if args.visual_clocks:
                command.append('--visual-clocks')
            if args.pooled_clocks:
                command.append('--pooled-clocks')
            if args.selected_fast_clocks:
                command.append('--selected-fast-clocks')
            if args.idle_recovery:
                command.append('--idle-recovery')
            with (run/'worker.log').open('w') as out:
                code = monitored_worker(command, out)
            summary['worker_exit_code'] = code
            if code:
                raise RuntimeError('Worker failed; inspect worker.log and any clock logs for restoration')
            if args.full_service:
                record = json.loads((run/'full_service.json').read_text())
                if record['status'] != 'completed' or len(record['requests']) != 5*len(service_profiles):
                    raise RuntimeError('Incomplete live Full service measurement')
                fields = ('service_median_ms', 'service_max_ms', 'mean_service_energy_j',
                          'acquisition_median_ms', 'preprocess_median_ms', 'predict_median_ms',
                          'submit_median_ms', 'restored')
                summary.update(status='completed', scope=record['scope'], model_loads=record['model_loads'],
                               services=len(record['requests']), conditions=record['conditions'],
                               **{key: record[key] for key in fields if key in record})
                return 0
            if args.continue_control or args.continue_camera_compare:
                record = json.loads((run/'continue_control.json').read_text())
                if record['status'] != 'completed' or len(record['windows']) != (30 if args.continue_camera_compare else 15):
                    raise RuntimeError('Incomplete Continue control screen')
                summary.update(status='completed', scope=record['scope'], model_loads=record['model_loads'],
                               conditions=record['summary'], all_final_states_equal=record['all_final_states_equal'])
                return 0
            filename = 'clock_matrix.json' if args.matrix or args.idle_recovery else 'clock_recovery.json' if args.recovery else 'clock_transitions.json' if args.transitions else 'clock_screen.json'
            record = json.loads((run/filename).read_text())
            if record['status'] != 'completed' or len(record['conditions']) != (9 if args.idle_recovery else 3 if args.fast_clocks or args.visual_clocks or args.pooled_clocks or args.selected_fast_clocks else 9 if args.matrix else 6 if args.recovery else 4):
                raise RuntimeError('Incomplete clock screen')
            if args.transitions or args.recovery or args.matrix or args.idle_recovery:
                summary.update(status='completed', scope=record['scope'], model_loads=record['model_loads'],
                               conditions=record['summary'])
                return 0
            summary.update(status='completed', scope=record['scope'], model_loads=record['model_loads'], conditions=[
                dict(profile=r['profile'], median_ms=r['median_ms'],
                     mean_compute_energy_per_request_j=r['mean_compute_energy_per_request_j'],
                     compute_frequency_check=r['compute_frequency_check'],
                     max_abs_action_difference_vs_baseline=r['max_abs_action_difference_vs_baseline'],
                     static_hold_power_w=r['static_hold']['mean_power_w'], restored=r['restored'])
                for r in record['conditions']])
    except BaseException as exc:
        summary.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        _write_json_atomically(run/'summary.json', summary)
        meta = json.loads((run/'run.json').read_text())
        meta['status'] = summary['status']
        _write_json_atomically(run/'run.json', meta)
        print('CLOCK_SCREEN_STATUS='+json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
