"""Paired fixed-duration Full routes with a common LOW start and end state."""
from dataclasses import asdict
import time

from perceptbudget.device_clock_screen import energy_window, frequency_check
from perceptbudget.experiment import _write_json_atomically
from perceptbudget.power import PowerMonitor

WINDOW_SECONDS=4.
PREHOLD_SECONDS=3.
TAIL_SECONDS=3.


def window_end(start, returned_low):
    end=start+WINDOW_SECONDS
    if returned_low+TAIL_SECONDS>end:
        raise RuntimeError('Full route leaves less than3s tail in the common4s window')
    return end


def switch(clocks,profile):
    if clocks.pending is not None:raise RuntimeError('Unexpected pending frequency change')
    if clocks.profile==profile:return None
    clocks.switch(profile)
    deadline=time.monotonic()+3.
    while clocks.pending is not None:
        clocks.poll()
        if time.monotonic()>deadline:raise TimeoutError('Frequency acknowledgement timeout')
        if clocks.pending is not None:time.sleep(.002)
    return dict(clocks.last_change)


def measure_routes(adapter,env,prompt,stats,previous_raw,queue,*,torch,np,directory,result,validation_options):
    from perceptbudget.closed_loop import RuntimeClocks
    from perceptbudget.device_service import absolute_pair_plan,measure_one,summarize_absolute_pairs
    path=directory/'full_service.json'
    result.update(scope='Fixed-input Full frequency routes, common4s window; not a closed-loop task',
        energy_scope='Absolute board energy from service start through return to LOW and common tail',
        single_request_energy_status='Common4s route window, at least3s post-return tail; inspect raw readings',
        prehold_seconds=PREHOLD_SECONDS,route_window_seconds=WINDOW_SECONDS,minimum_tail_seconds=TAIL_SECONDS,
        histories=[('settled_low',PREHOLD_SECONDS)],requested_extra_idle_seconds=PREHOLD_SECONDS,
        input_mode='Same fixed physics/images/state/old actions and pair noise; no env.step',
        calibrated_energy_field=None,reported_energy_field='absolute_pair_review',route_blocks=[])
    refs={}
    for block in absolute_pair_plan():
        b=block['block'];profile=block['profile']
        clock_dir=directory/f'route_clocks_{b}';clock_dir.mkdir()
        clocks=RuntimeClocks(clock_dir)
        monitor=PowerMonitor(.02,max_gap_seconds=.1)
        block_rows=[]
        try:
            clocks.__enter__();monitor.start()
            for pair_id in block['pair_ids']:
                switch(clocks,'gpu_emc')
                time.sleep(PREHOLD_SECONDS)
                torch.manual_seed(20261023+pair_id)
                up=switch(clocks,profile)
                start=up['command_monotonic'] if up else time.monotonic()
                measured=measure_one(adapter,env,prompt,stats,previous_raw,queue,torch=torch,np=np,
                    **dict(validation_options,diagnostic_label=f'route_b{b}_p{pair_id}'))
                ready=measured['ready_monotonic']
                down=switch(clocks,'gpu_emc')
                returned=down['observed_match_monotonic'] if down else ready
                end=window_end(start,returned)
                time.sleep(max(0.,end-time.monotonic()))
                monitor.sample();time.sleep(.025);monitor.sample();clocks.poll()
                route=energy_window(monitor,start,end)
                service=energy_window(monitor,start,ready)
                pre=energy_window(monitor,start-.5,start)
                tail=energy_window(monitor,end-.5,end)
                if any(e['energy_j'] is None for e in (route,service,pre,tail)):
                    raise RuntimeError('Full route power coverage incomplete')
                events=clocks.lease.events
                compute_start=up['observed_match_monotonic'] if up else start
                compute=frequency_check(events,profile,compute_start,ready)
                hold=frequency_check(events,'gpu_emc',returned,end)
                before=frequency_check(events,'gpu_emc',start-.5,start)
                if not all(c['matches'] for c in (compute,hold,before)):
                    raise RuntimeError('Full route frequency window mismatch')
                raw=np.asarray(measured['request']['actions_raw'],dtype=np.float32)
                key=(measured['frame_sha256'],measured['raw_state'])
                if pair_id not in refs:refs[pair_id]=(key,raw.copy())
                ref,refraw=refs[pair_id]
                difference=float(np.max(np.abs(raw-refraw)))
                frame_matches=key[0]==ref[0];state_matches=key[1]==ref[1]
                if not frame_matches or not state_matches or difference!=0:
                    raise RuntimeError('Full route pair input/output mismatch')
                packet=measured['packet'];payload=packet.pop('action_payload')
                np.savez_compressed(directory/f'route_b{b}_p{pair_id}.npz',actions=payload,raw_actions=raw)
                row=dict(pair_id=pair_id,pair_block=b,index=pair_id,profile=profile,state_history='settled_low3s',
                    requested_idle_seconds=PREHOLD_SECONDS,seed=20261023+pair_id,
                    measured=measured,energy=service,route_energy=route,service_ms=(ready-start)*1000,
                    route_seconds=end-start,post_return_tail_seconds=end-returned,
                    entry_change=up,return_change=down,
                    prehold_mean_power_w=pre['mean_power_w'],end_hold_mean_power_w=tail['mean_power_w'],
                    frequency_check=compute,post_low_frequency_check=hold,idle_frequency_check=before,
                    frame_matches_reference=frame_matches,state_matches_reference=state_matches,
                    raw_action_max_abs_difference_vs_reference=difference)
                result['requests'].append(row);block_rows.append(row)
                _write_json_atomically(path,result)
                print(f"FULL_ROUTE_REQUEST=b{b}/p{pair_id} profile={profile} route_j={route['energy_j']:.4f}",flush=True)
        finally:
            monitor.stop()
            _write_json_atomically(directory/f'route_power_block{b}.json',[asdict(r) for r in monitor.readings])
            if clocks.lease.proc is not None:clocks.close()
        restored='RESTORED' in (clock_dir/'runtime_clocks.jsonl').read_text().splitlines()
        result['route_blocks'].append(dict(block=b,profile=profile,requests=len(block_rows),restored=restored,
            clock_log=str(clock_dir/'runtime_clocks.jsonl'),power_log=str(directory/f'route_power_block{b}.json')))
        if not restored:raise RuntimeError('Full route frequency restoration missing')
        result['conditions'].append(dict(profile=profile,pair_block=b,restored=restored,
            requests=len(block_rows),mean_route_energy_j=sum(r['route_energy']['energy_j'] for r in block_rows)/len(block_rows)))
    result.update(status='completed',restored=True,
        camera_extra_reads=sum(r['measured'].get('acquisition',{}).get('real_camera_reads',1)-1 for r in result['requests']),
        absolute_pair_review=summarize_absolute_pairs(result['requests'],energy_field='route_energy'))
    _write_json_atomically(path,result)
