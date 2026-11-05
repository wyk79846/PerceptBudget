#!/usr/bin/env python3
"""Opt-in small real-async M4 joint-selection test; one resident model."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from perceptbudget.experiment import load_config,create_run,_write_json_atomically
from perceptbudget.closed_loop import load_settings,recording_session,clock_preflight
from perceptbudget.baseline import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suites',nargs='+',choices=('libero_spatial','libero_object','libero_goal','libero_10'),default=['libero_spatial'])
    parser.add_argument('--task-id',type=int,choices=range(10),default=0)
    parser.add_argument('--acceptance-group',choices=('q30','joint','m3','m3_fast'),help='Fixed four-suite/three-task acceptance group')
    parser.add_argument('--all-tasks',action='store_true',help='With M3 or q30: all10 tasks in each of the four LIBERO suites; preserve the default12-task manifest')
    parser.add_argument('--reuse-prefix-graphs',action='store_true',help='Reuse the bounded exact-length Full CUDA graph bank across tasks, without skipping resets or calibration')
    parser.add_argument('--first-tasks',type=int,choices=(3,12),default=12,help='Use the first3 fixed tasks for an M3 screen')
    parser.add_argument('--task-indices',type=int,nargs='+',choices=range(40),help='Explicit canonical task indices;0..11 by default,0..39 with --all-tasks')
    parser.add_argument('--result-index',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--resume-from',type=Path,help='Keep audited completed outcomes and run only missing tasks of the same group')
    parser.add_argument('--cover-close-slowdown',action='store_true',help='Experimental timely Full close also covers following slowdown within existing5-step range')
    parser.add_argument('--m3-eta-ms',type=int,choices=(25,50),default=25,help='M3 safety margin;50 reproduces original delivery setting')
    parser.add_argument('--reuse-preparation',action='store_true',help='Experimental setup reuse; pending cross-task GPU parity validation')
    parser.add_argument('--fast-advance','--fast-reserve-full',dest='fast_reserve_full',action='store_true',help='Advance Fast submission by its complete recovery-to-commit budget plus margin, rounded to control steps')
    parser.add_argument('--fast-selective416',action='store_true',help='Experimental416 visual tokens with state retained')
    parser.add_argument('--critical-wait-phases',action='store_true',help='With critical wait: one completed wait per finite predicted motion phase')
    parser.add_argument('--three-frequency',action='store_true',help='Opt-in306MHz Continue; medium Full first, high fallback; requires --low-full')
    parser.add_argument('--full-energy',action='store_true',help='At the original Full preparation trigger, compare complete service and remaining Continue energy for HIGH/MID')
    parser.add_argument('--idle-min-ms',type=int,choices=(40,50,100,150,200),
        help='Minimum predicted Continue gap before entering306MHz; with --full-frequency-pair compare the reference threshold against this value')
    parser.add_argument('--idle-reference-ms',type=int,choices=(100,200),default=200,
        help='Reference for an idle threshold pair;200 reproduces the original comparison')
    parser.add_argument('--ordinary-fast',action='store_true',help='Replace one ordinary Full with reduced-input Fast; keep Full schedule/frequency, then require Full')
    parser.add_argument('--ordinary-fast-tokens',type=int,choices=(320,464),default=320,help='Selective visual budget for ordinary Fast;464 merges only8 blocks per camera')
    parser.add_argument('--ordinary-fast-view',choices=('dual','wrist'),default='dual',help='Ordinary Fast: selective dual views without state, or native wrist view with latest state')
    parser.add_argument('--ordinary-fast-costs',type=Path,help='Completed measured costs for the explicitly selected ordinary Fast geometry')
    parser.add_argument('--ordinary-fast-positions',action='store_true',help='Energy Fast: preserve original camera/text coordinates and matching action positions')
    parser.add_argument('--ordinary-fast-pair',action='store_true',help='Resident current Full/Continue versus ordinary Fast pair; requires --ordinary-fast')
    parser.add_argument('--fast-alternate',action='store_true',help='Simple Full/Fast alternation with state in both; retain Full refresh and clock rules')
    capture_options=parser.add_mutually_exclusive_group()
    capture_options.add_argument('--lean-continue',dest='lean_continue',action='store_true',default=None,
        help='Skip idle camera rendering and save request frames only; default for new Full/Continue M3 and q30 runs')
    capture_options.add_argument('--full-capture',dest='lean_continue',action='store_false',
        help='Retain original per-step camera/recording behavior for reproduction')
    parser.add_argument('--continue-overhead-pair',action='store_true',help='Resident same-policy pair: original capture versus lean Continue, for M3 or q30')
    parser.add_argument('--request-only-images',action='store_true',help='With lean capture, also skip unused rendering during inference; acquire current dual images before every request')
    parser.add_argument('--request-capture-pair',action='store_true',help='Resident current lean Continue versus request-only images; requires --request-only-images')
    parser.add_argument('--common-low-tail',action='store_true',help='After pending inference finishes, record the same3s LOW endpoint separately from task energy')
    parser.add_argument('--start-floor-ms',type=int,choices=(0,25),default=0,
        help='Three-state startup-point floor: permit at most25ms intra-tick clock acknowledgement before inference')
    parser.add_argument('--low-full',action='store_true',help='M3 only: deadline-first Full frequency selection; energy recorded only; no Fast')
    parser.add_argument('--rtc-delay-sensitivity',action='store_true',help='Five resident one-task trials: initial8/9/10/11 plus unguided, one each')
    parser.add_argument('--rtc-delay-reference-run',type=Path,help='Replay a recorded per-task initial RTC delay; keep online updates enabled')
    parser.add_argument('--frequency-reference-run',type=Path,help='Merge completed Full frequency pairs from an earlier batch; do not rerun them')
    parser.add_argument('--full-frequency-pair',action='store_true',help='Resident Full high/low pair; with --three-frequency compare original two-state and new three-state policies')
    parser.add_argument('--phase-refresh',action='store_true',help='M3: suppress repeated event refresh within an already refreshed motion phase')
    parser.add_argument('--simplified-refresh',action='store_true',help='Combine active phase memory, turn-direction identity and smooth-bend filtering; allow the H50 window instead of a20/30-step cap')
    parser.add_argument('--energy-refresh',action='store_true',help='Simplified Full plus prediction-consistency filtering and a20-step ordinary feedback window; no30-step cap')
    parser.add_argument('--first-refresh-step',type=int,choices=(30,),help='M3 experiment: first event-free refresh request at step30')
    parser.add_argument('--critical-wait',action='store_true',help='M3 only: stop simulation at a predicted key action until the pending refresh arrives')
    parser.add_argument('--fast-key-phases',action='store_true',help='Experimental320 key-phase protection and one Full response')
    parser.add_argument('--full-only',action='store_true',help='M3 Fast ablation: retain window/controller but forbid Fast candidates')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--prepare-environments',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--environment-cache-from',type=Path,help='Reuse a passed identical M4 task environment preflight')
    parser.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.simplified_refresh=args.simplified_refresh or args.energy_refresh
    # Fast needs the request snapshot, never extra images while it computes.
    args.request_only_images=args.request_only_images or args.ordinary_fast
    if args.all_tasks and args.acceptance_group not in ('m3','q30'):
        parser.error('--all-tasks requires --acceptance-group m3 or q30')
    if args.reuse_prefix_graphs and args.acceptance_group not in ('m3','q30'):
        parser.error('--reuse-prefix-graphs requires Full-only M3 or q30')
    if args.task_indices and not args.all_tasks and max(args.task_indices)>=12:
        parser.error('Task indices above11 require --all-tasks')
    if args.idle_reference_ms!=200 and (args.idle_min_ms is None or not args.full_frequency_pair):
        parser.error('--idle-reference-ms requires --idle-min-ms and --full-frequency-pair')
    if args.idle_min_ms is not None and (not args.three_frequency or args.acceptance_group!='m3' or args.full_energy or args.continue_overhead_pair):
        parser.error('--idle-min-ms requires three-frequency M3, without energy-selection or capture comparisons')
    if args.full_energy and (not args.three_frequency or args.acceptance_group!='m3' or args.ordinary_fast or args.continue_overhead_pair):
        parser.error('--full-energy requires three-frequency M3, without Fast or capture changes')
    if args.ordinary_fast and (not args.three_frequency or args.acceptance_group!='m3' or args.full_frequency_pair or args.continue_overhead_pair):
        parser.error('--ordinary-fast requires the three-frequency M3 path, without other paired policies')
    if args.ordinary_fast_pair and not args.ordinary_fast:parser.error('--ordinary-fast-pair requires --ordinary-fast')
    if args.fast_alternate and (not args.ordinary_fast or args.ordinary_fast_positions or args.ordinary_fast_costs is None):
        parser.error('--fast-alternate requires --ordinary-fast and matching stateful costs without position overrides')
    if args.ordinary_fast_tokens!=320 and not args.ordinary_fast:parser.error('--ordinary-fast-tokens requires --ordinary-fast')
    if args.ordinary_fast_positions and not (args.ordinary_fast and args.energy_refresh):parser.error('--ordinary-fast-positions requires --ordinary-fast and --energy-refresh')
    if (args.ordinary_fast_view!='dual' or args.ordinary_fast_costs is not None) and not args.ordinary_fast:
        parser.error('Ordinary Fast geometry/costs require --ordinary-fast')
    if args.ordinary_fast_view=='wrist' and (args.ordinary_fast_positions or args.ordinary_fast_tokens!=320 or args.ordinary_fast_costs is None):
        parser.error('Stateful wrist needs its measured --ordinary-fast-costs, without selective-token overrides')
    if (args.lean_continue is not None or args.continue_overhead_pair) and args.acceptance_group not in ('m3','q30'):
        parser.error('Continue capture requires --acceptance-group m3 or q30')
    if args.continue_overhead_pair and (args.full_frequency_pair or args.rtc_delay_sensitivity):
        parser.error('Compare capture variants separately from frequency pairs or RTC delay sensitivity')
    if args.request_only_images and (args.acceptance_group not in ('m3','q30') or args.lean_continue is False):
        parser.error('--request-only-images requires lean M3 or q30 capture')
    if args.request_capture_pair and (not args.request_only_images or args.continue_overhead_pair or args.full_frequency_pair or args.rtc_delay_sensitivity or args.ordinary_fast):
        parser.error('--request-capture-pair requires --request-only-images without another paired policy or Fast')
    if args.request_only_images and args.continue_overhead_pair:
        parser.error('Use --request-capture-pair to compare inference image rendering')
    if args.start_floor_ms and not args.three_frequency:
        parser.error('--start-floor-ms requires --three-frequency')
    if args.three_frequency and (not args.low_full or args.acceptance_group!='m3' or args.rtc_delay_sensitivity or args.frequency_reference_run):
        parser.error('--three-frequency requires --low-full --acceptance-group m3, without sensitivity or an old frequency-reference batch')
    if args.full_frequency_pair and (not args.low_full or args.acceptance_group!='m3'):
        parser.error('--full-frequency-pair requires --low-full --acceptance-group m3')
    if args.rtc_delay_sensitivity and (args.acceptance_group!='m3' or args.low_full or args.full_frequency_pair or args.rtc_delay_reference_run or not args.task_indices or len(args.task_indices)!=1):
        parser.error('Sensitivity requires one --task-indices with --acceptance-group m3, no other delay/frequency overrides')
    if args.rtc_delay_reference_run and args.acceptance_group!='m3':
        parser.error('--rtc-delay-reference-run requires the Full/Continue policy')
    if args.frequency_reference_run and not args.full_frequency_pair:
        parser.error('--frequency-reference-run requires --full-frequency-pair')
    if args.frequency_reference_run:
        previous=json.loads((args.frequency_reference_run/'summary.json').read_text())
        if previous.get('status')!='completed' or 'full_frequency_pairs' not in previous:
            parser.error('Reference must contain completed Full frequency pairs')
    if args.low_full and args.acceptance_group!='m3':parser.error('--low-full requires M3 without Fast')
    if args.phase_refresh and args.acceptance_group!='m3':parser.error('--phase-refresh requires M3')
    if args.simplified_refresh and (args.acceptance_group!='m3' or args.first_refresh_step is not None):
        parser.error('--simplified-refresh requires Full-only M3 without --first-refresh-step')
    if args.first_refresh_step is not None and args.acceptance_group!='m3':parser.error('--first-refresh-step requires M3')
    if args.critical_wait_phases and not args.critical_wait:parser.error('--critical-wait-phases requires --critical-wait')
    if args.critical_wait and args.acceptance_group!='m3':parser.error('--critical-wait requires M3 without Fast')
    if args.fast_reserve_full and (args.acceptance_group!='m3_fast' or args.full_only):parser.error('--fast-reserve-full requires m3_fast without --full-only')
    if args.fast_selective416 and (not args.fast_reserve_full or args.full_only):parser.error('--fast-selective416 requires --fast-advance without --full-only')
    if args.fast_key_phases and (not args.fast_reserve_full or args.fast_selective416 or args.full_only):parser.error('--fast-key-phases requires --fast-advance with320 Fast')
    if args.full_only and args.acceptance_group!='m3_fast':parser.error('--full-only requires m3_fast')
    if args.first_tasks!=12 and args.acceptance_group not in ('m3','m3_fast','q30'):parser.error('Three-task screen requires an M3 or q30 candidate')
    if args.task_indices and (args.acceptance_group not in ('m3','m3_fast','q30') or args.first_tasks!=12 or len(set(args.task_indices))!=len(args.task_indices)):
        parser.error('Explicit indices require M3 or q30, default first-tasks and unique indices')
    if args.m3_eta_ms!=25 and args.acceptance_group not in ('m3','m3_fast'):parser.error('Margin override requires M3')
    cfg=load_config(repo_root=ROOT);validate(cfg)
    cfg['m3_low_full']=args.low_full
    cfg['m3_three_frequency']=args.three_frequency
    cfg['m3_full_energy']=args.full_energy
    if args.idle_min_ms is not None:
        cfg.update(m3_idle_min_ms=args.idle_min_ms,m3_idle_reference_ms=args.idle_reference_ms)
    cfg['m3_start_floor_ms']=args.start_floor_ms
    cfg['m4_lean_continue']=args.lean_continue if args.lean_continue is not None else args.acceptance_group in ('m3','q30')
    cfg['m4_continue_overhead_pair']=args.continue_overhead_pair
    cfg['m4_request_only_images']=args.request_only_images
    cfg['m4_request_capture_pair']=args.request_capture_pair
    cfg['m4_all_tasks']=args.all_tasks
    cfg['m4_resident_prefix_graphs']=args.reuse_prefix_graphs
    if args.three_frequency:
        from perceptbudget.three_frequency import enable_profiles,transition_costs
        enable_profiles()
        cfg['m3_three_frequency_transitions']=transition_costs(cfg['output_root'])
    cfg['m4_rtc_delay_sensitivity']=args.rtc_delay_sensitivity
    if args.rtc_delay_reference_run:cfg['rtc_initial_delay_reference_run']=str(args.rtc_delay_reference_run.resolve())
    cfg['m4_full_frequency_pair']=args.full_frequency_pair
    if args.common_low_tail or args.full_frequency_pair or args.continue_overhead_pair or args.request_capture_pair:cfg['m4_common_low_tail_seconds']=3.
    cfg['m3_phase_refresh']=args.phase_refresh or args.simplified_refresh
    cfg['m3_first_refresh_step']=args.first_refresh_step
    if args.simplified_refresh:
        cfg.update(m3_critical_forecast='remaining_v1',m3_phase_event_source='remaining_v2',
            m3_phase_coverage_memory='all_active_v1',m3_event_identity='turn_direction_v1',
            m3_ordinary_turn_policy='context_v1',m3_max_refresh_interval=50)
    if args.energy_refresh:
        cfg.update(m3_feedback_policy='overlap_pacing_v2',m3_feedback_minimum_steps=20)
    cfg['m3_critical_wait_phases']=args.critical_wait_phases
    cfg['m3_critical_wait']=args.critical_wait
    if args.critical_wait:
        cfg['rtc_critical_wait_prefix_policy']='committed_stop_v1'
    cfg['experimental_setup_reuse']=args.reuse_preparation
    settings=load_settings(ROOT/'m3.json');m4=json.loads((ROOT/'m4.json').read_text())
    from perceptbudget.m4_planner import POLICY_REVISION
    cfg['m4_policy_revision']=POLICY_REVISION
    settings.update(mode='deadline_dvfs',calibration_tasks=[[s,args.task_id] for s in args.suites])
    settings['timing'].update(m4['timing'],maximum_action_offset=50,maximum_observation_age_seconds=None,margin_updates=False)
    cfg.update(m4_enabled=True,m4_settings=m4,m4_cover_close_slowdown=args.cover_close_slowdown,m3_action_events_enabled=True,m3_dvac_enabled=False,
        m3_event_coverage_mode='event_only',m3_signal_policy='earliest',m3_merge_slowdown_steps=None)
    if args.acceptance_group:
        from perceptbudget.m4_acceptance import manifest
        fixed=manifest(cfg,all_tasks=args.all_tasks)
        cfg.update(m4_acceptance_group=args.acceptance_group,m4_acceptance_manifest=fixed)
        if args.acceptance_group=='joint':
            cfg.update(m4_fast_prompt_capacity=20,
                m4_fast_cost_scope='M2 costs are initial estimates from exact no-state text shapes; '
                'acceptance uses fixed20 (maximum across the12 fixed tasks) capacity with valid-token masking; task calibration measures actual latency, '
                'whole-task energy is measured independently')
        settings['calibration_tasks']=fixed['tasks']
        if args.acceptance_group=='q30':
            settings['mode']='record_only'
            cfg.update(m4_enabled=False,m3_hold_high_clocks=True)
            settings['calibration_tasks']=[fixed['tasks'][i] for i in args.task_indices] if args.task_indices else fixed['tasks'] if args.first_tasks==12 else fixed['tasks'][:args.first_tasks]
        if args.acceptance_group in ('m3','m3_fast'):
            from perceptbudget.m4_acceptance import configure_m3,configure_m3_fast
            (configure_m3_fast if args.acceptance_group=='m3_fast' else configure_m3)(cfg,settings)
            cfg['m3_fast_reserve_full']=False
            cfg['m3_fast_advance']=args.fast_reserve_full
            cfg['m3_fast_key_phases']=args.fast_key_phases
            if args.fast_key_phases:cfg['m3_key_phase_response_policy']='full_once_v1'
            else:cfg.pop('m3_key_phase_response_policy',None)
            if args.fast_selective416:
                cfg.update(m3_fast_selective416=True,m2_fast_no_state=False)
                cfg.pop('m4_fast_prompt_capacity',None)
            if args.fast_reserve_full:cfg['m3_fast_policy']='m3_service_ledger_fast_advance_v4'
            cfg['m3_fast_full_only']=args.full_only
            settings['timing']['safety_margin_seconds']=args.m3_eta_ms/1000
            cfg['m3_replay_eta_ms']=args.m3_eta_ms
            settings['calibration_tasks']=[fixed['tasks'][i] for i in args.task_indices] if args.task_indices else fixed['tasks'] if args.first_tasks==12 else fixed['tasks'][:args.first_tasks]
            cfg['m3_comparison_tasks']=settings['calibration_tasks']
    if args.low_full:
        from perceptbudget.m3_low_full import load_costs
        cfg['m3_low_full_policy']='shared_deadline_v4'
        cfg['m3_low_full_dispatch_policy']='before_next_control_tick'
        cfg['m3_low_full_rtc_policy']='per_profile_v2'
        cfg['m3_low_full_costs']=load_costs(m4)
    if args.ordinary_fast:
        from perceptbudget.ordinary_fast import load_costs as fast_costs,POLICY,FEEDBACK_POLICY,ALTERNATING_POLICY
        policy=ALTERNATING_POLICY if args.fast_alternate else FEEDBACK_POLICY if args.energy_refresh else POLICY
        cfg.update(m3_ordinary_fast=True,m3_ordinary_fast_context=True,m3_ordinary_fast_pair=args.ordinary_fast_pair,
            m2_fast_no_state=True,m4_fast_prompt_capacity=32 if args.energy_refresh else 20,m4_policy_revision=policy,
            m3_ordinary_fast_policy=policy,m3_ordinary_fast_visual_mode='selective'+str(args.ordinary_fast_tokens),
            m3_ordinary_fast_positions=args.ordinary_fast_positions)
        if args.ordinary_fast_view=='wrist':
            cfg.update(m3_ordinary_fast_visual_mode='wrist',m2_fast_no_state=False)
        if args.fast_alternate:
            cfg.update(m3_fast_alternation=True,m2_fast_no_state=False)
        cost_key='ordinary_fast_position_cost_source' if args.ordinary_fast_positions else 'ordinary_fast_cost_source' if args.ordinary_fast_tokens==320 else 'ordinary_fast_464_cost_source'
        if args.ordinary_fast_costs is None and cost_key not in m4:parser.error('Measure the selected Fast costs before closed-loop use: '+cost_key)
        cfg['ordinary_fast_costs']=fast_costs(args.ordinary_fast_costs or m4[cost_key],cfg)
    if args.full_energy:
        from perceptbudget.full_energy import load_costs as energy_costs,POLICY as ENERGY_POLICY
        cfg.update(m3_full_energy=True,m3_full_energy_costs=energy_costs(m4['full_energy_cost_source'],cfg),
                   m4_policy_revision=ENERGY_POLICY)
    if args.idle_min_ms is not None:
        settings['timing']['minimum_low_idle_seconds']=args.idle_min_ms/1000
        cfg['m3_short_idle_policy']='reuse_planned_ack_v1'
    pair=bool(args.full_frequency_pair or args.continue_overhead_pair or args.ordinary_fast_pair or args.request_capture_pair)
    variant_count=len(cfg['m4_diagnostic_variants']) if pair and cfg.get('m4_diagnostic_variants') else 2
    if args.ordinary_fast_pair:cfg['m4_common_low_tail_seconds']=3.
    continuation=None
    if args.resume_from:
        from perceptbudget.m4_acceptance import resume_joint,resume_full
        if args.acceptance_group not in ('joint','m3','q30'):parser.error('Resume requires a joint or Full-only acceptance group')
        continuation=(resume_joint if args.acceptance_group=='joint' else resume_full)(args.resume_from,cfg)
        cfg['m3_resume_plan']=continuation
    pending_tasks=continuation['remaining_tasks'] if continuation else settings['calibration_tasks']
    if args.frequency_reference_run:
        old_tasks={(e['suite'],e['task_id']) for e in previous['episodes']}
        if old_tasks & {tuple(t) for t in pending_tasks}:
            parser.error('Reference tasks overlap the requested new tasks')
        source_cfg=json.loads((args.frequency_reference_run/'run.json').read_text())['config_snapshot']
        if source_cfg.get('m4_acceptance_manifest')!=cfg.get('m4_acceptance_manifest'):
            parser.error('Reference model/task manifest differs')
    if args.dry_run:
        print(json.dumps(dict(tasks=pending_tasks,reused_episodes=len(continuation['episodes']) if continuation else 0,group=args.acceptance_group or 'joint',model_loads=1,Full=True,Fast=cfg['m3_ordinary_fast_visual_mode']+('_no_state' if cfg.get('m2_fast_no_state') else '_with_state') if args.ordinary_fast else ('selective416_with_state' if args.fast_selective416 else 'pool2_agent_no_state') if cfg['m4_enabled'] or cfg.get('m3_fast_enabled') else 'disabled',
            eta_ms=None if args.acceptance_group=='q30' else args.m3_eta_ms if args.acceptance_group in ('m3','m3_fast') else 25,H=50,N=5,cover_close_slowdown=args.cover_close_slowdown,
            request_only_images=args.request_only_images,request_capture_pair=args.request_capture_pair,
            ordinary_fast_positions=args.ordinary_fast_positions,
            fast_alternate=args.fast_alternate,
            all_tasks=args.all_tasks,reuse_prefix_graphs=args.reuse_prefix_graphs,
            simplified_refresh=args.simplified_refresh,phase_coverage_memory=cfg.get('m3_phase_coverage_memory'),
            event_identity=cfg.get('m3_event_identity'),ordinary_turn_policy=cfg.get('m3_ordinary_turn_policy'),
            max_refresh_interval=cfg.get('m3_max_refresh_interval'),
            feedback_policy=cfg.get('m3_feedback_policy'),feedback_minimum_steps=cfg.get('m3_feedback_minimum_steps'),
            compute_profiles=['baseline','gpu_emc'] if cfg['m4_enabled'] or args.low_full else ['baseline'],
            policies=10 if cfg['m4_enabled'] else 1,
            three_frequency=args.three_frequency,full_energy=args.full_energy,idle_min_ms=settings['timing']['minimum_low_idle_seconds']*1000,idle_pair_ms=[args.idle_reference_ms,args.idle_min_ms] if args.full_frequency_pair and args.idle_min_ms is not None else None,start_floor_ms=args.start_floor_ms,full_frequency_pair=args.full_frequency_pair,lean_continue=cfg['m4_lean_continue'],continue_overhead_pair=args.continue_overhead_pair,ordinary_fast=args.ordinary_fast,ordinary_fast_pair=args.ordinary_fast_pair,episodes=len(pending_tasks)*(5 if args.rtc_delay_sensitivity else variant_count if pair else 1),common_low_tail_seconds=cfg.get('m4_common_low_tail_seconds'),low_full=args.low_full,low_full_policy=cfg.get('m3_low_full_policy'),phase_refresh=cfg['m3_phase_refresh'],first_refresh_step=cfg['m3_first_refresh_step'],critical_wait_phases=args.critical_wait_phases,critical_wait=args.critical_wait,key_phase_response_policy=cfg.get('m3_key_phase_response_policy'),fast_key_phases=args.fast_key_phases,fast_selective416=args.fast_selective416,fast_reserve_full=args.fast_reserve_full,full_only=args.full_only,m3_fast_selection=bool(cfg.get('m3_fast_enabled')),perception_policy=cfg.get('m3_fast_policy'),source='original M3 D_joint DeadlineRuntime, Full-high/low-Continue, explicit replay margin' if args.acceptance_group in ('m3','m3_fast') else 'fixed q30 Full high-frequency actual async' if args.acceptance_group=='q30' else 'complete common-window plans, no inference in Continue')));return 0
    if args.prepare_environments:
        if not args.acceptance_group or args.run_dir is None or args.run_dir.resolve().parent!=(Path(cfg['output_root'])/'M4').resolve():
            parser.error('Environment preparation requires acceptance group and M4 directory')
        if args.environment_cache_from:
            from perceptbudget.m4_acceptance import copy_precompiled_environment_cache,precompiled_environments
            reused=copy_precompiled_environment_cache(args.environment_cache_from,args.run_dir,pending_tasks,cfg)
            with precompiled_environments(args.run_dir,pending_tasks,cfg):
                pass
            print('ENVIRONMENT_CACHE_REUSED='+json.dumps(reused),flush=True)
        else:
            from perceptbudget.closed_loop import compiled_environments
            with compiled_environments(args.run_dir,pending_tasks,cfg):
                pass
        print('ENVIRONMENT_PREPARATION=completed',flush=True)
        return 0
    if args.worker:
        if args.run_dir is None or args.run_dir.resolve().parent!=(Path(cfg['output_root'])/'M4').resolve():
            parser.error('Worker requires M4 directory')
        subprocess.run(['sudo','-n','/usr/bin/true'],check=True)
        cfg.update(m3_watchdog_path=str(args.run_dir/'request_watchdog.json'),m3_request_timeout_seconds=3.)
        if args.acceptance_group:cfg['m4_precompiled_environments']=True
        from scripts.eval_pooled_views import CLEANUP_GATE
        result=recording_session(cfg,settings,args.run_dir,CLEANUP_GATE)
        # Audit selected compute frequency, not M3's fixed high-frequency assumption.
        from perceptbudget.m4_acceptance import clock_audit
        audits=[];service_ledgers=[]
        from perceptbudget.service_ledger import write_run_ledger
        for row in result['episodes']:
            run=Path(row['run']);audit=clock_audit(run)
            _write_json_atomically(run/'m4_clock_audit.json',audit);audits.append(audit)
            if not audit['passed']:raise RuntimeError('M4 frequency/power/restore audit failed')
            service_ledgers.append(write_run_ledger(run,forbid_hot_fast=bool(cfg.get('m3_fast_advance'))))
        if args.rtc_delay_sensitivity:
            from perceptbudget.rtc_sensitivity import summarize as summarize_sensitivity
            result['rtc_delay_sensitivity']=summarize_sensitivity(result['episodes'],args.run_dir)
        if args.full_frequency_pair or args.ordinary_fast_pair:
            from perceptbudget.full_frequency_pairs import summarize_pairs
            result['full_frequency_pairs']=summarize_pairs(result['episodes'],args.run_dir,
                groups=('ordinary_full','ordinary_fast') if args.ordinary_fast_pair else (f'full_idle{args.idle_reference_ms}','full_idle_candidate') if args.idle_min_ms is not None else ('full_three_medium','full_three_energy') if args.full_energy else ('full_three_strict','full_three_start25') if args.start_floor_ms else
                       ('full_two','full_three') if args.three_frequency else ('full_high','full_low'))
            if args.frequency_reference_run:
                combined=args.run_dir/'combined_frequency_pairs'
                combined.mkdir()
                result['combined_frequency_pairs']=summarize_pairs(previous['episodes']+result['episodes'],combined)
                result['combined_frequency_pairs']['source_runs']=[str(args.frequency_reference_run),str(args.run_dir)]
                _write_json_atomically(combined/'sources.json',result['combined_frequency_pairs']['source_runs'])
        if args.continue_overhead_pair:
            from perceptbudget.full_frequency_pairs import summarize_pairs
            result['continue_overhead_pairs']=summarize_pairs(result['episodes'],args.run_dir,
                groups=('full_capture','lean_continue'),stem='continue_overhead_pairs')
        if args.request_capture_pair:
            from perceptbudget.full_frequency_pairs import summarize_pairs
            result['request_capture_pairs']=summarize_pairs(result['episodes'],args.run_dir,
                groups=('inflight_images','request_images'),stem='request_capture_pairs')
        result.update(mode='short_idle' if args.idle_min_ms is not None else 'full_energy' if args.full_energy else ('ordinary_fast_pair' if args.ordinary_fast_pair else 'ordinary_fast') if args.ordinary_fast else 'm3_three_frequency' if args.three_frequency else ('m3_fast_selection' if args.acceptance_group=='m3_fast' else 'm3_frozen_joint') if args.acceptance_group in ('m3','m3_fast') else 'm4_'+(args.acceptance_group or 'joint'),m4_clock_audits=audits,service_continue_ledgers=service_ledgers)
        _write_json_atomically(args.run_dir/'summary.json',result)
        return 0
    from perceptbudget.clock_access import authorize
    authorize()
    run=create_run(dict(cfg,m4_tasks=settings['calibration_tasks']),'M4',
        'acceptance-'+args.acceptance_group if args.acceptance_group else 'joint-smoke',repo_root=ROOT)
    if args.result_index:
        _write_json_atomically(args.result_index,dict(run=str(run)))
    print('run_dir='+str(run),flush=True);status=dict(status='running')
    stage_times={}
    def finish_stage(name,started):
        stage_times[name]=time.monotonic()-started
        _write_json_atomically(run/'stage_timings.json',stage_times)
        print('ACCEPTANCE_STAGE='+json.dumps(dict(stage=name,seconds=round(stage_times[name],3))),flush=True)
    try:
        with (Path(cfg['output_root'])/'.async_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.three_frequency:
                from perceptbudget.three_frequency import preflight
                preflight(run)
            else:clock_preflight(run)
            command=[cfg['python_executable'],'-u',str(Path(__file__).resolve()),'--worker','--run-dir',str(run),
                '--task-id',str(args.task_id),'--suites',*args.suites]
            if args.acceptance_group in ('m3','m3_fast'):command.extend(['--m3-eta-ms',str(args.m3_eta_ms)])
            if args.fast_reserve_full:command.append('--fast-reserve-full')
            if args.fast_selective416:command.append('--fast-selective416')
            if args.fast_key_phases:command.append('--fast-key-phases')
            if args.critical_wait:command.append('--critical-wait')
            if args.critical_wait_phases:command.append('--critical-wait-phases')
            if args.first_refresh_step is not None:command.extend(['--first-refresh-step',str(args.first_refresh_step)])
            if args.phase_refresh:command.append('--phase-refresh')
            if args.simplified_refresh:command.append('--simplified-refresh')
            if args.energy_refresh:command.append('--energy-refresh')
            if args.low_full:command.append('--low-full')
            if args.three_frequency:command.append('--three-frequency')
            if args.full_energy:command.append('--full-energy')
            if args.idle_min_ms is not None:command.extend(['--idle-min-ms',str(args.idle_min_ms)])
            if args.idle_reference_ms!=200:command.extend(['--idle-reference-ms',str(args.idle_reference_ms)])
            if args.ordinary_fast:command.extend(['--ordinary-fast','--ordinary-fast-tokens',str(args.ordinary_fast_tokens)])
            if args.ordinary_fast_view!='dual':command.extend(['--ordinary-fast-view',args.ordinary_fast_view])
            if args.ordinary_fast_costs is not None:command.extend(['--ordinary-fast-costs',str(args.ordinary_fast_costs)])
            if args.ordinary_fast_positions:command.append('--ordinary-fast-positions')
            if args.ordinary_fast_pair:command.append('--ordinary-fast-pair')
            if args.fast_alternate:command.append('--fast-alternate')
            if cfg['m4_lean_continue']:command.append('--lean-continue')
            elif args.lean_continue is False:command.append('--full-capture')
            if args.continue_overhead_pair:command.append('--continue-overhead-pair')
            if args.request_only_images:command.append('--request-only-images')
            if args.request_capture_pair:command.append('--request-capture-pair')
            if args.all_tasks:command.append('--all-tasks')
            if args.reuse_prefix_graphs:command.append('--reuse-prefix-graphs')
            if args.common_low_tail:command.append('--common-low-tail')
            if args.start_floor_ms:command.extend(['--start-floor-ms',str(args.start_floor_ms)])
            if args.full_frequency_pair:command.append('--full-frequency-pair')
            if args.rtc_delay_sensitivity:command.append('--rtc-delay-sensitivity')
            if args.rtc_delay_reference_run:command.extend(['--rtc-delay-reference-run',str(args.rtc_delay_reference_run)])
            if args.frequency_reference_run:command.extend(['--frequency-reference-run',str(args.frequency_reference_run)])
            if args.full_only:command.append('--full-only')
            if args.reuse_preparation:command.append('--reuse-preparation')
            if args.cover_close_slowdown:command.append('--cover-close-slowdown')
            if args.acceptance_group:command.extend(['--acceptance-group',args.acceptance_group])
            command.extend(['--first-tasks',str(args.first_tasks)])
            if args.task_indices:command.extend(['--task-indices',*map(str,args.task_indices)])
            if args.resume_from:command.extend(['--resume-from',str(args.resume_from)])
            from scripts.profile_clock_screen import monitored_worker
            if args.acceptance_group:
                # Resume validates the capture/timing policy even in the
                # preparation process. Share the exact worker arguments.
                preparation=list(command)
                preparation.remove('--worker')
                preparation.append('--prepare-environments')
                if args.environment_cache_from:preparation.extend(['--environment-cache-from',str(args.environment_cache_from)])
                started=time.monotonic()
                print('ACCEPTANCE_STAGE=environment_preparation_started',flush=True)
                with (run/'environment_preparation.log').open('w') as log:
                    prepared=monitored_worker(preparation,log)
                finish_stage('environment_preparation',started)
                if prepared:raise RuntimeError('Environment preparation failed; no model loaded')
            started=time.monotonic()
            print('ACCEPTANCE_STAGE=model_worker_started',flush=True)
            with (run/'worker.log').open('w') as log:
                code=monitored_worker(command,log,request_watchdog=run/'request_watchdog.json',watchdog_grace_seconds=3.)
            finish_stage('model_worker',started)
            if code:raise RuntimeError('Worker failed; inspect '+str(run/'worker.log'))
            result=json.loads((run/'summary.json').read_text())
            if result['status']!='completed' or len(result['episodes'])!=len(settings['calibration_tasks'])*(5 if args.rtc_delay_sensitivity else variant_count if pair else 1):raise RuntimeError('Incomplete M4 test')
            status=result
    except BaseException as exc:
        status=dict(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        _write_json_atomically(run/'summary.json',status)
        meta=json.loads((run/'run.json').read_text());meta['status']=status['status'];_write_json_atomically(run/'run.json',meta)
        print('M4_TERMINAL_STATUS='+status['status'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
