"""Opt-in replay of a recorded per-task RTC initialization, not fixed runtime delay."""
import json
from pathlib import Path


def reference_delay(directory, suite, task_id, config):
    directory=Path(directory)
    record=json.loads((directory/'recording.json').read_text())
    matches=[r for r in record['episodes'] if r['suite']==suite and r['task_id']==task_id]
    if len(matches)!=1:
        raise ValueError('RTC delay reference requires exactly one episode for this task')
    run=Path(matches[0]['run'])
    saved=json.loads((run/'run.json').read_text())['config_snapshot']
    for key in ('H','N','precision','state_enabled','checkpoint_path'):
        if saved.get(key)!=config.get(key):
            raise ValueError('RTC reference configuration differs: '+key)
    delay=saved['rtc_initial_delay_steps']
    if type(delay) is not int or not 0<delay<config['H']:
        raise ValueError('Invalid recorded RTC initial delay')
    return dict(initial_delay_steps=delay,source_run=str(run),
                scope='Initial history only; subsequent delays still update from actual execution')
