from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import json
import time

import numpy as np
import pytest


@pytest.mark.parametrize('fast',[False,True])
@pytest.mark.parametrize('batch',[1,2,3])
def test_both_validation_tags_create_real_run_directories(tmp_path,fast,batch):
    from scripts.profile_clock_screen import validation_purpose
    from perceptbudget.experiment import create_run
    repo=tmp_path/'repo';repo.mkdir()
    config={'output_root':str(tmp_path/'outputs')}
    purpose=validation_purpose(fast,batch)
    # Exercise the same production validator which rejected the old Fast tag.
    directory=create_run(config,'M2',purpose,repo_root=repo)
    assert directory.is_dir()
    assert len(purpose)<=32
    assert json.loads((directory/'run.json').read_text())['purpose']==purpose


@pytest.mark.parametrize('flags,state', [(['--full-service'], True),
    (['--selected-fast-clocks','--live-service'], False)])
def test_validation_plan_unifies_sixty_live_requests_and_keeps_state_choice(flags,state,capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv',['probe',*flags,'--service-validation','--validation-batch','2','--dry-run']), \
         patch('subprocess.run') as launch:
        assert main()==0
    d=json.loads(capsys.readouterr().out)
    assert d['requests']==60 and d['requests_per_condition']==20 and d['model_loads']==1
    assert d['profiles']==['baseline'] and d['histories_seconds']==[0,.2,1.]
    assert d['batch']==2 and d['state_enabled'] is state
    launch.assert_not_called()


def test_three_batches_have_six_model_sessions_without_per_request_reload(capsys):
    from scripts.run_m2_validation import main
    with patch('sys.argv',['probe','--all','--dry-run']),patch('subprocess.run') as launch:
        assert main()==0
    d=json.loads(capsys.readouterr().out)
    assert d['model_loads']==6 and len(d['jobs'])==6
    assert d['service_requests']==360 and d['continue_windows']==180
    launch.assert_not_called()


def test_missing_full_cells_plan_skips_completed_continue(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv',['probe','--full-service','--service-validation',
            '--service-profiles','balanced','gpu_emc','--skip-continue-validation','--dry-run']), \
         patch('subprocess.run') as launch:
        assert main()==0
    result=json.loads(capsys.readouterr().out)
    assert result['requests']==120
    assert result['profiles']==['balanced','gpu_emc']
    assert not result['runs_continue_after_service']
    launch.assert_not_called()


@pytest.mark.parametrize('flags',[[],['--full-service'],
    ['--selected-fast-clocks','--live-service','--service-validation']])
def test_skip_continue_rejects_unrelated_modes(flags):
    from scripts.profile_clock_screen import main
    with patch('sys.argv',['probe',*flags,'--skip-continue-validation','--dry-run']), \
         pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2


@pytest.mark.parametrize('missing_cells_only', [False, True])
def test_live_service_validation_keeps_all_sixty_requests_and_reuses_model_for_continue(tmp_path, missing_cells_only):
    from perceptbudget.device_service import screen
    gate=tmp_path/'gate';gate.mkdir()
    (gate/'reset_reuse_audit.json').write_text(json.dumps(dict(passed=True,cases=[{}]*6,render_cleanup={'errors':0})))
    image=np.zeros((2,2,3),dtype=np.uint8)
    obs={'agentview_image':image,'robot0_eye_in_hand_image':image}
    raw=np.zeros((50,32),dtype=np.float32);actions=np.zeros((50,7),dtype=np.float32)
    closed=[];lease_calls=[];seeds=[];executed=[]
    env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(get_state=lambda:SimpleNamespace(flatten=lambda:np.zeros(3)))),
        step=lambda a:(executed.append(a.copy()) or obs,0.,False,{}),check_success=lambda:False,close=lambda:closed.append(True))
    def bootstrap(snapshot,prompt,request):
        request.update(actions_raw=(raw+5).tolist(),rtc_guidance={'valid_prefix_length':665})
        return actions+5
    adapter=SimpleNamespace(config={'norm_stats_path':'unused','m2_service_validation':True,'m2_validation_batch':1},
        expert=SimpleNamespace(),step_bank=None,bootstrap=bootstrap,
        prepare_step_bank=lambda lengths:dict(passed=True),predict=lambda *a,**kw:actions)
    if missing_cells_only:
        adapter.config['m2_skip_continue_validation'] = True
    torch=SimpleNamespace(manual_seed=seeds.append,cuda=SimpleNamespace(empty_cache=lambda:None))
    def measured(*args,**kwargs):
        assert np.array_equal(args[4], raw[5:])
        now=time.monotonic()
        return dict(acquisition=dict(start_monotonic=now,stop_monotonic=now,physics_sha256='physics'),
            preprocess_start_monotonic=now,preprocess_stop_monotonic=now,
            inference_start_monotonic=now,inference_stop_monotonic=now,
            submit_start_monotonic=now,ready_monotonic=now+.001,
            frame_sha256='frame',raw_state=[0]*8,request={'actions_raw':raw.tolist()},
            packet={'action_payload':actions.copy()})
    class Lease:
        events=[]
        def __init__(self,profile,idle,directory,label,*,serve_profile):
            lease_calls.append((profile,idle,serve_profile));self.profile=serve_profile
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def send(self,token):pass
        def receive(self,status,index):
            now=time.monotonic()
            return dict(serve_profile=self.profile,down={'command_monotonic':now-.003,'observed_match_monotonic':now-.002},
                        up={'command_monotonic':now-.001,'observed_match_monotonic':now})
    class Monitor:
        readings=[]
        def __init__(self,*a,**kw):pass
        def start(self):pass
        def stop(self):pass
        def sample(self):pass
    def continuation(*args,**kwargs):
        assert not missing_cells_only
        assert args[0] is adapter and closed and adapter.config['m2_continue_validation']
        assert kwargs['history_windows']
        return dict(status='completed',windows=[{}]*60)
    with ExitStack() as stack:
        patches={
            'scripts.eval_pooled_views.CLEANUP_GATE':gate,
            'scripts.eval_q._patch_egl_cleanup':lambda:None,
            'scripts.eval_q._seed_process':lambda *a:None,
            'scripts.eval_q._libero_api':lambda:(None,object),
            'scripts.eval_q._load_tasks':lambda *a:[{'instruction':'test'}],
            'scripts.eval_q._make_env':lambda *a:env,
            'scripts.eval_q._prepare_env_episode':lambda *a:(obs,20),
            'scripts.eval_q._make_snapshot':lambda *a:{},
            'perceptbudget.device_quality.install_render_buffer_cleanup':lambda **kw:{},
            'perceptbudget.libero_input.load_libero_norm_stats':lambda *a:{},
            'perceptbudget.rtc_memory.memory_record':lambda *a:{'available_mib':2000},
            'perceptbudget.device_service.fresh_observation':lambda *a:(obs,{'physics_sha256':'physics'}),
            'perceptbudget.device_service.measure_one':measured,
            'perceptbudget.device_service.validation_old_chunk':lambda *a:(raw.copy(),actions.copy()),
            'perceptbudget.device_service.RecoveryLease':Lease,
            'perceptbudget.device_service.PowerMonitor':Monitor,
            'perceptbudget.device_service.energy_window':lambda *a:{'energy_j':10.},
            'perceptbudget.device_service.frequency_check':lambda *a:{'matches':True},
            'perceptbudget.device_service.inventory':lambda:{'power_mode':40,'thermal':[]},
            'perceptbudget.device_service.matched_hold_baseline':lambda *a,**kw:{'energy':{'energy_j':5.}},
            'perceptbudget.device_service.time.sleep':lambda *a:None,
            'perceptbudget.device_continue.screen':continuation,
        }
        for name,value in patches.items():stack.enter_context(patch(name,value))
        profiles=('balanced','gpu_emc') if missing_cells_only else ('baseline',)
        result=screen(adapter,{},'test',np.zeros((45,32)),torch,tmp_path,{'valid_prefix':665,'model_loads':1},profiles=profiles)
    count=len(profiles)
    assert result['status']=='completed' and len(result['requests'])==60*count
    assert len(result['conditions'])==3*count and all(c['services']==20 for c in result['conditions'])
    assert len(lease_calls)==12*count
    assert all(c['max_raw_action_difference_vs_reference']==0 for c in result['conditions'])
    assert [r['index'] for r in result['requests']]==list(range(20))*3*count
    assert not adapter.config.get('m2_continue_validation')
    assert len(executed)==5 and np.array_equal(np.asarray(executed), actions[:5])


def test_sensor_response_reports_combined_latency_not_update_rate():
    from scripts.probe_measurement_quality import observed_response
    blocks=[dict(mode='idle',start_monotonic=0.,stop_monotonic=3.),
            dict(mode='busy',start_monotonic=3.,stop_monotonic=6.)]
    samples=[dict(timestamp=t,power_mw=p) for t,p in [(2.6,1000),(2.8,1000),
        (3.1,2000),(3.8,10000),(5.6,10000),(5.8,10000)]]
    result=observed_response({'blocks':blocks,'samples':samples})
    assert result[0]['t10_ms']==pytest.approx(100.)
    assert result[0]['t90_ms']==pytest.approx(800.)


@pytest.mark.parametrize('complete',[True,False])
def test_matched_hold_uses_common_duration_and_refuses_missing_energy(tmp_path,complete):
    from perceptbudget.device_service import matched_hold_baseline
    now=[10.];prepared=[]
    rows=[dict(timing={'down':{'command_monotonic':float(i)}},
               measured={'ready_monotonic':i+.5}) for i in range(5)]
    class Lease:
        events=[]
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def send(self,token):
            if token=='RUN':self.start=now[0]
        def receive(self,status,index):
            if status=='SERVE':
                now[0]+=.218
                return dict(down={'command_monotonic':self.start,'observed_match_monotonic':self.start+.018},
                    up={'command_monotonic':self.start+.2,'observed_match_monotonic':now[0]})
            return {}
    class Monitor:
        readings=[]
        def __init__(self,*a,**kw):pass
        def start(self):pass
        def stop(self):pass
        def sample(self):pass
    def window(m,a,b):return dict(duration_seconds=b-a,energy_j=50. if complete else None)
    with patch('perceptbudget.device_service.RecoveryLease',Lease), \
         patch('perceptbudget.device_service.PowerMonitor',Monitor), \
         patch('perceptbudget.device_service.frequency_check',return_value={'matches':True}), \
         patch('perceptbudget.device_service.energy_window',side_effect=window), \
         patch('perceptbudget.device_service.time.monotonic',side_effect=lambda:now[0]), \
         patch('perceptbudget.device_service.time.sleep',side_effect=lambda t:now.__setitem__(0,now[0]+t)):
        if complete:
            out=matched_hold_baseline(tmp_path,'test','baseline','gpu_emc',.2,rows,prepare=lambda:prepared.append(True))
            assert out['energy']['duration_seconds']==pytest.approx(7.5)
            assert len(out['schedule'])==5 and out['max_abs_down_schedule_error_ms']==pytest.approx(0.)
            assert prepared==[True]
        else:
            with pytest.raises(RuntimeError,match='power incomplete'):
                matched_hold_baseline(tmp_path,'test','baseline','gpu_emc',.2,rows,prepare=lambda:None)


def resume_fixture(tmp_path):
    cfg={'output_root':str(tmp_path),'H':50,'python_executable':'python-test'}
    root=tmp_path/'M2';root.mkdir()
    full=root/'full';full.mkdir()
    (full/'run.json').write_text(json.dumps({'config_snapshot':cfg}))
    rows=[]
    for state,idle in [('hot',0.),('short',.2),('long',1.)]:
        for index in range(20):
            rows.append(dict(state_history=state,index=index,profile='baseline',requested_idle_seconds=idle,
                frame_matches_reference=True,state_matches_reference=True,raw_action_max_abs_difference_vs_reference=0,
                frequency_check={'matches':True},idle_frequency_check={'matches':True},
                measured={'frame_sha256':'pixels','raw_state':[0]*8}))
    record=dict(status='completed',restored=True,requests=rows,initial_physics_sha256='physics',old_target_sha256='old',
        conditions=[dict(services=20,restored=True,batch_energy={'energy_j':10.},
                        matched_hold={'energy':{'energy_j':5.}}) for _ in range(3)])
    (full/'full_service.json').write_text(json.dumps(record))
    (full/'continue_control.json').write_text(json.dumps({'status':'invalid','windows':[{}]*5}))
    session=root/'old_session';session.mkdir()
    (session/'validation_index.json').write_text(json.dumps(dict(batches=[1],jobs=[
        dict(batch=1,mode='full_continue',run=str(full),exit_code=1)])))
    return cfg,full,session,record


def test_resume_preserves_complete_full_and_retries_only_continue_then_fast(tmp_path,capsys):
    from scripts.run_m2_validation import main
    cfg,full,session,record=resume_fixture(tmp_path)
    original=(full/'full_service.json').read_bytes()
    target=tmp_path/'M2'/'new_session';target.mkdir()
    (target/'run.json').write_text(json.dumps({'config_snapshot':cfg}))
    commands=[]
    def run(command,**kwargs):
        if command[0]=='sudo':return SimpleNamespace(returncode=0)
        commands.append(command)
        child=tmp_path/'M2'/('continue_new' if '--continue-validation' in command else 'fast_new')
        child.mkdir()
        (child/'summary.json').write_text(json.dumps({'status':'completed'}))
        if '--continue-validation' not in command:
            (child/'selected_fast_service.json').write_text(json.dumps(record))
        kwargs['stdout'].write('run_dir='+str(child)+'\n')
        return SimpleNamespace(returncode=0)
    with patch('sys.argv',['probe','--batch','1','--resume-session',str(session)]), \
         patch('perceptbudget.experiment.load_config',return_value=cfg), \
         patch('perceptbudget.experiment.create_run',return_value=target), \
         patch('scripts.run_m2_validation.subprocess.run',side_effect=run):
        assert main()==0
    assert len(commands)==2 and '--continue-validation' in commands[0]
    assert '--full-service' not in commands[0] and '--service-validation' not in commands[0]
    assert '--selected-fast-clocks' in commands[1]
    d=json.loads((target/'validation_index.json').read_text())
    assert d['status']=='completed' and d['jobs'][0]['full_service_run']==str(full)
    assert d['jobs'][1]['cross_mode_observation_pairing_passed']
    assert (full/'full_service.json').read_bytes()==original


def test_service_reuse_rejects_duplicate_requests_or_changed_baseline(tmp_path):
    from scripts.run_m2_validation import verified_service
    cfg,full,session,record=resume_fixture(tmp_path)
    assert verified_service(full,'full_service.json',cfg)
    with pytest.raises(ValueError,match='configuration'):
        verified_service(full,'full_service.json',dict(cfg,H=20))
    record['requests'][1]['index']=0
    (full/'full_service.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='duplicates'):
        verified_service(full,'full_service.json',cfg)


@pytest.mark.parametrize('history',[0.,'initial'])
def test_continue_phase_uses_shared_real_camera_guard(tmp_path,history):
    from perceptbudget.device_continue import continue_observation
    phase=np.zeros((2,4,3),dtype=np.uint8)
    good={'agentview_image':phase[:,:2].copy(),'robot0_eye_in_hand_image':phase[:,2:].copy()}
    bad={'agentview_image':phase[:,:2].copy()+1,'robot0_eye_in_hand_image':phase[:,2:].copy()}
    with patch('perceptbudget.device_service.fresh_observation',side_effect=[(bad,{}),(good,{})]):
        obs,record=continue_observation(None,history,{history:phase},tmp_path,'window5',validated=True)
    assert obs is good and record['real_camera_reads']==2
    assert (tmp_path/'window5_frame_attempt0.npz').exists()


def test_continue_only_validation_does_not_repeat_full_or_merge_partial_windows(capsys):
    from scripts.profile_clock_screen import main
    with patch('sys.argv',['probe','--continue-history','--continue-validation','--dry-run']), \
         patch('subprocess.run') as run:
        assert main()==0
    d=json.loads(capsys.readouterr().out)
    assert d['windows']==60 and not d['full_service_rerun'] and not d['merges_old_partial_windows']
    run.assert_not_called()
