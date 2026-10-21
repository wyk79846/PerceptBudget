"""The sole deployment baseline, frozen from the completed H50 M1 run."""
from pathlib import Path

BASELINE_ID = 'pi05-h50-n5-state-rtc-flash-norm-autotune'
FROZEN = {
    'baseline_id': BASELINE_ID,
    'checkpoint_profile': 'lerobot_h50_quantiles_v044',
    'checkpoint_provenance_revision': 'b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd',
    'checkpoint_key_prefix': 'model.',
    'H': 50, 'N': 5, 'precision': 'bfloat16', 'state_enabled': True,
    'state_prompt_capacity': 200, 'action_handoff': 'rtc_vjp_actual_history',
    'pending_arm_gain': 1.0, 'rtc_beta': 5.0,
    'rtc_author_order': True, 'rtc_paper_schedule': True,
    'rtc_fixed_delay_estimate': None, 'rtc_delay_buffer_size': 10,
    'rtc_calibrate_control_period': True,
    'rtc_exact_flash_attention': True, 'rtc_fused_norm': True,
    'rtc_suffix_only_vjp': False, 'native_gemm_policy': 'autotune',
    'source_packed_conditioning': True, 'release_unused_native_expert': True,
}
QS = (5, 10, 20, 30, 40)
SUITES = ('libero_spatial', 'libero_object', 'libero_goal')


def apply_to_reference(reference, canonical):
    """Keep proof/history fields, but never inherit an old deployment baseline."""
    import copy
    validate(canonical)
    result = copy.deepcopy(reference)
    keys = set(FROZEN) | {
        'model_id', 'checkpoint_path', 'model_path', 'norm_stats_path',
        'checkpoint_validation', 'verified_reference_run', 'execution_mode',
        'baseline_status', 'baseline_reference_run', 'source_prefix_coverage_rule',
    }
    result.update({key: canonical[key] for key in keys})
    return result


def validate(config):
    for key, value in FROZEN.items():
        if key not in config or config[key] != value:
            raise ValueError(f'Baseline mismatch: {key}; expected {value!r}')
    if config.get('candidate_q') != list(QS):
        raise ValueError('Baseline candidate_q must be 5, 10, 20, 30, 40')
    checkpoint = Path(config['checkpoint_path'])
    if checkpoint.name != 'lerobot_pi05_libero_h50_quantiles_v044':
        raise ValueError('Baseline requires the pinned H50 checkpoint')
    if Path(config['model_path']) != checkpoint / 'model.safetensors':
        raise ValueError('Baseline model_path does not match checkpoint_path')
    if Path(config['norm_stats_path']) != checkpoint / 'perceptbudget_norm_stats.json':
        raise ValueError('Baseline requires matching H50 normalization statistics')


def command(config, root, *, suite='libero_spatial', task_id=0, q=5):
    """One task, one model lifetime; diagnostic subprocess owns guard and GPU lock."""
    validate(config)
    if suite not in SUITES or type(task_id) is not int or not 0 <= task_id < 10:
        raise ValueError('Invalid LIBERO suite/task')
    if q not in QS:
        raise ValueError('Unsupported q')
    return [config['python_executable'], '-u',
            str(Path(root) / 'scripts/diagnose_rtc_handoff.py'),
            '--reference-run', config['baseline_reference_run'],
            '--suite', suite, '--task-ids', str(task_id),
            '--conditions', 'actual_history', '--q', str(q), '--beta', '5',
            '--horizon', '50', '--checkpoint-profile', 'h50',
            '--exact-flash-attention', '--fused-norm',
            '--gemm-policy', 'autotune', '--calibrate-control-period']
