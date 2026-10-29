import json
from pathlib import Path
from perceptbudget.m4_wait_costs import extract


def test_wait_extraction_excludes_inference_transition_and_drain(tmp_path):
    rtc=tmp_path/'rtc';rtc.mkdir()
    readings=[dict(timestamp_monotonic=i*.1,power_mw=8000,error=None) for i in range(31)]
    (tmp_path/'summary.json').write_text(json.dumps({'episode':{'power':{'raw_readings':readings}}}))
    (rtc/'q5_requests.json').write_text(json.dumps([dict(observation_monotonic=0,ready_monotonic=.5)]))
    keys=('class/devfreq/17000000.gpu/cur_freq','kernel/debug/bpmp/debug/clk/emc/rate',
          'devices/system/cpu/cpufreq/policy0/scaling_cur_freq','devices/system/cpu/cpufreq/policy4/scaling_cur_freq')
    actual=dict(zip(keys,(816000000,2133000000,1497600,1497600)))
    clocks=[dict(status='SAMPLE',monotonic=i*.02,actual=actual) for i in range(151)]
    clocks.append(dict(status='APPLIED',profile='gpu_emc',command_monotonic=.8,observed_match_monotonic=.82))
    (rtc/'runtime_clocks.jsonl').write_text('\n'.join(json.dumps(c) for c in clocks))
    controls=[dict(event='control_step',tick_monotonic=i*.05,inference_active=i<10,late_hold=False) for i in range(41)]
    (rtc/'trace.jsonl').write_text('\n'.join(json.dumps(c) for c in controls))
    result=extract(tmp_path,.3)
    assert result['windows']
    assert all(w['start']>=1.12 and w['stop']<=2 for w in result['windows'])
    assert all(abs(w['energy_j']/w['seconds']-8)<1e-8 for w in result['windows'])
    assert result['transitions'][0]['energy_status']=='unresolved_at_100ms_power_poll'
