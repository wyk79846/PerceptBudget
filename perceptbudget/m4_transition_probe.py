"""Bounded active-control directional preparation windows, one resident model."""
from dataclasses import asdict
from contextlib import contextmanager
import json,time,gc,ctypes
from pathlib import Path
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor
from perceptbudget.device_clock_screen import energy_window,frequency_check
from perceptbudget.closed_loop import RuntimeClocks

WINDOW_SECONDS=.25


def plan():
    return [(i,mode) for i in range(5) for mode in (('hold','switch') if i%2==0 else ('switch','hold'))]



@contextmanager
def clock_session(directory):
    clocks=RuntimeClocks(directory)
    clocks.__enter__()
    try:
        yield clocks
    finally:
        clocks.close()


def screen(adapter,snapshot,prompt,previous,torch,directory,metadata):
    import numpy as np
    from scripts import eval_q as base,probe_delay_buffer as probe
    from perceptbudget.device_quality import install_render_buffer_cleanup
    from perceptbudget.libero_input import load_libero_norm_stats
    from perceptbudget.rtc_memory import memory_record
    directory=Path(directory);cfg=adapter.config
    install_render_buffer_cleanup(shared_egl=True)
    _,env_class=base._libero_api();task=base._load_tasks([0],['libero_spatial'])[0]
    stats=load_libero_norm_stats(cfg['norm_stats_path'])
    result=dict(metadata,status='running',trials=[],model_loads=1,scope='Paired active-control250ms direction windows, not instantaneous switch-only energy')
    path=directory/'m4_transitions.json';env=None;reference=None;fixed_snapshot=None
    try:
        env=base._make_env(env_class,task);env._pb_lock_fixtures=True
        for repeat,mode in plan():
            label=f'{repeat}_{mode}';trial=directory/label;trial.mkdir()
            base._seed_process(7,np,torch);obs,_=base._prepare_env_episode(env,task,np)
            scene=probe.snapshot_scene(env,obs)
            state=np.asarray(scene['sim_state'],dtype=np.float64)
            if reference is None:reference=state.copy();fixed_snapshot=base._make_snapshot(obs,stats,True)
            elif not np.array_equal(state,reference):raise RuntimeError('Physical initial state changed')
            gc.collect();ctypes.CDLL(None).malloc_trim(0)
            if memory_record(torch,label)['available_mib']<900:raise RuntimeError('RAM headroom below900MiB')
            monitor=PowerMonitor(.02,max_gap_seconds=.1);monitor.start();clocks=None;steps=[]
            try:
                with clock_session(trial) as clocks:
                    request={};actions=adapter.bootstrap(fixed_snapshot,task['instruction'],request)
                    actions=np.asarray(actions[:40],dtype=np.float32)
                    if len(actions)!=40 or not np.isfinite(actions).all():raise RuntimeError('Invalid action prefix')
                    origin=time.monotonic();deadline=origin;markers={};steps=[]
                    for index,action in enumerate(actions):
                        clocks.poll();tick=time.monotonic()
                        if index in (0,20):
                            target='gpu_emc' if index==0 and mode=='switch' else 'baseline'
                            if clocks.pending is not None:raise RuntimeError('Previous frequency transition incomplete')
                            clocks.switch(target);markers['down' if index==0 else 'up']=time.monotonic()
                        obs,_,done,_=env.step(action)
                        steps.append(dict(step=index,tick_monotonic=tick,action=action.tolist()))
                        deadline+=.05
                        if done or env.check_success():raise RuntimeError('Task terminated before complete measurement window')
                        if time.monotonic()>deadline:deadline=time.monotonic()
                        else:time.sleep(deadline-time.monotonic())
                    stop=time.monotonic()
                    while clocks.pending is not None:
                        clocks.poll()
                        if time.monotonic()>stop+3:raise RuntimeError('Frequency acknowledgement timeout')
                        time.sleep(.005)
                if time.monotonic()<markers['up']+WINDOW_SECONDS:raise RuntimeError('Incomplete post-up control')
            finally:
                monitor.stop()
                _write_json_atomically(trial/'power.json',[asdict(r) for r in monitor.readings])
                _write_json_atomically(trial/'control.json',locals().get('steps',[]))
            events=clocks.lease.events
            applied=[e for e in events if e.get('status')=='APPLIED']
            if len(applied)!=2:raise RuntimeError('Missing directional frequency acknowledgement')
            if not any(e.get('message')=='RESTORED' for e in events):raise RuntimeError('Missing frequency restoration')
            windows={}
            for phase,event in zip(('down','up'),applied):
                start=event['command_monotonic'];end=start+WINDOW_SECONDS
                if not origin<=start<end<=stop:raise RuntimeError('Directional window outside active control')
                windows[phase]=dict(**energy_window(monitor,start,end),
                    readback_seconds=event['observed_match_monotonic']-start,target=event['profile'],
                    scope='Entire250ms preparation+control window, includes post-switch hold')
                if windows[phase]['energy_j'] is None:raise RuntimeError('Incomplete direction power coverage')
                check=frequency_check(events,event['profile'],event['observed_match_monotonic'],end)
                if not check['matches']:raise RuntimeError('Direction actual frequency mismatch')
                windows[phase]['frequency_check']=check
            result['trials'].append(dict(repeat=repeat,mode=mode,windows=windows,steps=40,restored=True,
                source=str(trial),action_sha256=__import__('hashlib').sha256(actions.tobytes()).hexdigest()))
            _write_json_atomically(path,result);print('M4_TRANSITION_TRIAL='+label,flush=True)
        if len({r['action_sha256'] for r in result['trials']})!=1:raise RuntimeError('Paired controls differ')
        result['directional_costs']=directional_costs(result['trials'])
        result.update(status='completed',window_seconds=WINDOW_SECONDS,paired_input='Fixed snapshot at identical reset physics, fresh bootstrap per trial')
    except BaseException as exc:
        result.update(status='invalid',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if env is not None:env.close()
        _write_json_atomically(path,result)
    return result


def directional_costs(trials):
    """Absolute preparation windows, not baseline-subtracted instantaneous energy."""
    costs=[]
    for phase,source,target in (('down','baseline','gpu_emc'),('up','gpu_emc','baseline')):
        switched=[r['windows'][phase] for r in trials if r['mode']=='switch']
        controls=[r['windows'][phase] for r in trials if r['mode']=='hold']
        if len(switched)!=5 or len(controls)!=5:raise ValueError('Require five measured pairs')
        for w in switched+controls:
            if w['energy_j'] is None or not w['frequency_check']['matches']:
                raise ValueError('Incomplete directional measurement')
            if w['readback_seconds']>WINDOW_SECONDS:raise ValueError('Switch exceeded preparation window')
        costs.append(dict(source=source,target=target,seconds=WINDOW_SECONDS,
            energy_j=sum(w['energy_j'] for w in switched)/len(switched),
            baseline_window_energy_j=sum(w['energy_j'] for w in controls)/len(controls),
            max_readback_seconds=max(w['readback_seconds'] for w in switched),
            scope='Absolute250ms preparation and control, includes remaining hold'))
    return costs
