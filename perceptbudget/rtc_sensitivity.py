"""One-task diagnostic: RTC initial history sensitivity, plus unguided generation."""
import csv
import json
from pathlib import Path


def variants():
    conditions=['rtc_initial_d8','rtc_initial_d9','rtc_initial_d10','rtc_initial_d11','unguided']
    return [f'{c}_repeat{r+1}' for r in range(1)
            for c in conditions[r:]+conditions[:r]]


def configure(config, label):
    if label not in variants():
        raise ValueError('Unknown RTC sensitivity condition')
    condition,repeat=label.rsplit('_repeat',1)
    config.update(rtc_initial_delay_steps=9 if condition=='unguided' else int(condition.removeprefix('rtc_initial_d')),
        rtc_disable_guidance=condition=='unguided',rtc_sensitivity_condition=condition,
        rtc_sensitivity_repeat=int(repeat))


def summarize(episodes, directory):
    rows=[]
    for e in episodes:
        run=Path(e['run'])
        cfg=json.loads((run/'run.json').read_text())['config_snapshot']
        v=json.loads((run/'summary.json').read_text())['episode']
        rows.append(dict(condition=cfg['rtc_sensitivity_condition'],repeat=cfg['rtc_sensitivity_repeat'],
            initial_delay=cfg['rtc_initial_delay_steps'],outcome=e['outcome'],steps=e['steps'],
            seconds=v['real_seconds'],underflow_steps=len(v['underflow_steps']),run=str(run)))
    if len(rows)!=5 or {e['view'] for e in episodes}!=set(variants()):
        raise ValueError('Incomplete RTC sensitivity diagnostic')
    result=dict(rows=rows,scope='Same seed/init, one real-async run per condition; sensitivity screen, not repeatability or generalization proof',
        unguided_scope='Same context and expert weights; no old-action target or VJP; same scheduler and absolute crop; runtime differs')
    for condition in sorted({r['condition'] for r in rows}):
        result.setdefault('groups',[]).append(dict(condition=condition,
            successes=sum(r['outcome']=='success' for r in rows if r['condition']==condition),trials=1))
    directory=Path(directory)
    with (directory/'rtc_sensitivity.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (directory/'rtc_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
