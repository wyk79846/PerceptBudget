from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import json
import time

import numpy as np
import pytest


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


def test_live_service_validation_keeps_all_sixty_requests_and_reuses_model_for_continue(tmp_path):
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
        result=screen(adapter,{},'test',np.zeros((45,32)),torch,tmp_path,{'valid_prefix':665,'model_loads':1})
    assert result['status']=='completed' and len(result['requests'])==60
    assert len(result['conditions'])==3 and all(c['services']==20 for c in result['conditions'])
    assert len(lease_calls)==12
    assert all(c['max_raw_action_difference_vs_reference']==0 for c in result['conditions'])
    assert [r['index'] for r in result['requests']]==list(range(20))*3
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
