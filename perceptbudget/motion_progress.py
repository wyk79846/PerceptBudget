"""One state-only observation request per episode of poor downward response.

This is a progress hint, not a contact/grasp classifier or a physical model.
Command units are normalized; state Z is in meters. The first eight downward
samples establish this episode's response scale, then a three-step window
detects a large reduction. No simulator object state or contact labels enter.
"""
from collections import deque
from statistics import median


def downward_plan(actions, origin, step, required_steps, minimum_command, prediction_id):
    offset=step-origin
    future=actions[offset:offset+required_steps] if offset>=0 else []
    commands=[max(-1.,min(1.,float(a[2]))) for a in future]
    return dict(prediction_id=prediction_id,control_step=step,required_steps=required_steps,
        available_steps=len(commands),commands_z=commands,
        allowed=len(commands)==required_steps and all(z<=-minimum_command for z in commands))


class DownwardProgress:
    def __init__(self, minimum_command=.1, *, continuation_steps=0):
        self.minimum_command=minimum_command
        self.continuation_steps=continuation_steps;self.planned_stop_skips=0;self.last_plan=None
        self.samples=[];self.reference=None;self.window=deque(maxlen=3)
        self.processed=0;self.active=False;self.recovered=0
        self.onsets=0;self.protected_onsets=0;self.pending=None
        self.latest=None;self.last_onset=None;self.last_coverage=None

    def update(self, rows, *, protected, continuation=None):
        for row in rows[self.processed:]:
            command=max(-1.,min(1.,float(row['action'][2])))
            before=float(row['before']['raw_state'][2]);after=float(row['after']['raw_state'][2])
            sample=dict(step=row['step'],command=command,before=before,after=after,
                protected=bool(protected(row)),prediction_id=(row.get('source') or {}).get('prediction_id'))
            self.window.append(sample)
            if self.reference is None and command<=-self.minimum_command:
                self.samples.append((before-after)/abs(command))
                if len(self.samples)==8:self.reference=median(self.samples)
            if self.reference is None or self.reference<=0 or len(self.window)<3:continue
            down=all(s['command']<=-self.minimum_command for s in self.window)
            gain=((self.window[0]['before']-self.window[-1]['after'])/
                  sum(abs(s['command']) for s in self.window)) if down else None
            relative=gain/self.reference if gain is not None else None
            self.latest=dict(last_control_step=row['step'],downward_window=down,
                relative_response=relative,grasp_protected=any(s['protected'] for s in self.window))
            if down and relative<.2 and not self.active:
                self.active=True;self.recovered=0;self.onsets+=1
                event=dict(event_id=self.onsets,observed_step=row['step']+1,
                    window_steps=[s['step'] for s in self.window],relative_response=relative,
                    reference_meters_per_command=self.reference,
                    source_prediction_id=sample['prediction_id'],grasp_protected=self.latest['grasp_protected'])
                self.last_onset=event
                if event['grasp_protected']:self.protected_onsets+=1
                else:self.pending=dict(event)
            if self.active:
                recovered=not down or relative>=.4
                self.recovered=self.recovered+1 if recovered else 0
                if self.recovered>=3:self.active=False;self.recovered=0
        self.processed=len(rows)
        if self.continuation_steps and self.pending is not None:
            self.last_plan=continuation()
            if not self.last_plan['allowed']:
                self.planned_stop_skips+=1
                self.pending=None
        return self.status()

    def committed(self, request, commit):
        if self.pending is None or request['observation_step']<self.pending['observed_step']:
            return None
        self.last_coverage=dict(event_id=self.pending['event_id'],observed_step=self.pending['observed_step'],
            prediction_id=request['prediction_id'],observation_step=request['observation_step'],
            commit_step=commit['commit_step'])
        self.pending=None
        return dict(self.last_coverage)

    def status(self):
        result=dict(processed_controls=self.processed,calibration_samples=len(self.samples),
            reference_meters_per_command=self.reference,active=self.active,recovery_samples=self.recovered,
            onsets=self.onsets,protected_onsets=self.protected_onsets,
            latest=dict(self.latest) if self.latest else None,
            pending=dict(self.pending) if self.pending else None,
            last_onset=dict(self.last_onset) if self.last_onset else None,
            last_coverage=dict(self.last_coverage) if self.last_coverage else None)
        if self.continuation_steps:
            result.update(continuation_steps=self.continuation_steps,planned_stop_skips=self.planned_stop_skips,
                last_continuation_review=dict(self.last_plan) if self.last_plan else None)
        return result
