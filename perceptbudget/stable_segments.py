"""Causal state features and separate simulator-only feedback labels."""
import math
import re


WINDOW = 5
FEATURE_NAMES = ([f'state_{i}' for i in range(8)] + [f'joint_velocity_{i}' for i in range(7)]
                 + [f'gripper_velocity_{i}' for i in range(2)] + [f'action_{i}' for i in range(7)]
                 + ['control_seconds'])


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def object_positions(snapshot):
    return {k[:-4]: list(v) for k, v in snapshot.get('observations', {}).items()
            if k.endswith('_pos') and not k.startswith('robot') and '_to_' not in k
            and isinstance(v, list) and len(v) == 3}


def finger_contacts(snapshot, objects):
    result = {key: set() for key in objects}
    for c in snapshot.get('contacts') or []:
        if c.get('distance', 1.) > 0:
            continue
        pair = [c.get('geom1') or '', c.get('geom2') or '']
        for obj in objects:
            if not any(name.startswith(obj+'_') for name in pair):
                continue
            for name in pair:
                if 'finger' in name:
                    match = re.search(r'finger\d+', name)
                    result[obj].add(match[0] if match else name)
    return result


def online_features(row):
    obs = row['after']['observations']
    result = (list(row['after']['raw_state']) + list(obs['robot0_joint_vel'])
              + list(obs['robot0_gripper_qvel']) + list(row['action'])
              + [row['after']['sim_time']-row['before']['sim_time']])
    if len(result) != len(FEATURE_NAMES) or not all(math.isfinite(x) for x in result):
        raise ValueError('Missing finite online state/action features')
    return result


def phase_flags(row, decision):
    """Use decisions available before this action; never episode outcome."""
    if decision is None or decision.get('decision_monotonic', math.inf) > row['action_start_monotonic']:
        return dict(eligible=False, ordinary_declared=False, known_key_now=False, reason='missing_causal_decision')
    critical = decision.get('critical_step')
    known = critical is not None and abs(critical-row['step']) <= 2
    wait = decision.get('wait_seconds', 0.)
    eligible = (not row.get('inference_active_at_action', True)
                and not row.get('late_hold') and row.get('critical_wait_seconds', 0.) == 0
                and wait > 0 and not decision.get('start_now', True) and not known)
    return dict(eligible=bool(eligible), ordinary_declared=bool(decision.get('m3_fast_formula', {}).get('ordinary')),
        known_key_now=known, critical_step=critical, reason=decision.get('reason'),
        scope='Continue waiting outside the existing2-step key confirmation margin; ordinary flag also retained')


class PhysicalTeacher:
    """Local physical evidence, not final task failure or deployable features.

    Label1 is a local loss/empty-lift cue; it is not proof refreshing helps.
    Label0 covers only open free motion or a currently consistent carried object.
    Ambiguous contact/manipulation is left unlabeled, not silently normal.
    """
    def __init__(self):
        self.history = []
        self.initial_z = {}
        self.held = {}
        self.close = None
        self.open_width = 0.
        self.events = []
        self.last_event = {}

    def observe(self, row):
        state = row['after']['raw_state'];before = row['before']['raw_state']
        positions = object_positions(row['after'])
        contacts = finger_contacts(row['after'], positions)
        g = row['action'][6];width = abs(state[6]-state[7])
        for name, p in positions.items():self.initial_z.setdefault(name, p[2])
        if g <= -.2:
            self.open_width = max(self.open_width, width)
            self.close = None;self.held.clear()
        elif g >= .2 and self.close is None:
            touched = {name for previous in self.history[-5:] for name, fingers in previous['contacts'].items() if fingers}
            self.close = dict(step=row['step'], eef_z=before[2], objects=touched,
                origin={name:list(p) for name,p in positions.items()})
        info = dict(step=row['step'],eef=list(state[:3]),positions=positions,contacts=contacts,g=g,width=width)
        self.history.append(info)
        recent = self.history[-WINDOW:]
        emitted = []
        for name, p in positions.items():
            rel = [p[i]-state[i] for i in range(3)]
            both = len(contacts[name]) >= 2
            if (g >= .2 and both and p[2]-self.initial_z[name] >= .02
                    and sum(len(r['contacts'].get(name,())) >= 2 for r in recent) >= 2):
                held = self.held.setdefault(name,dict(established=row['step'],peak_relative_z=rel[2],misses=0))
                held['peak_relative_z'] = max(held['peak_relative_z'],rel[2])
            if name not in self.held or g < .2:
                continue
            held = self.held[name]
            held['misses'] = 0 if contacts[name] else held['misses']+1
            prior = recent[max(0,len(recent)-3)]
            relative_drop = held['peak_relative_z']-rel[2]
            if (row['step']-held['established'] >= 2 and held['misses'] >= 2
                    and relative_drop >= .02 and state[2] >= prior['eef'][2]-.003):
                emitted.append(dict(kind='held_object_separation',object=name,
                    relative_drop_m=relative_drop,missing_contact_steps=held['misses'],
                    scope='Lifted object lost contact and moved down relative to a non-descending hand'))
        if self.close and g >= .2 and row['step']-self.close['step'] >= 4:
            lift = state[2]-self.close['eef_z']
            saturated = self.open_width > 0 and width <= .05*self.open_width
            no_contact = all(not any(r['contacts'].values()) for r in recent[-3:])
            touched = self.close['objects'] & positions.keys()
            not_following = touched and all(positions[name][2]-self.close['origin'][name][2] < .005 for name in touched)
            if saturated and no_contact and lift >= .02 and not_following and not self.held:
                emitted.append(dict(kind='empty_lift_feedback',objects=sorted(touched),eef_lift_m=lift,
                    width_fraction=width/self.open_width,
                    scope='Pre-close object contact, then empty-stop aperture and lift without object following; feedback cue'))
        fresh = []
        for event in emitted:
            key=event['kind'],event.get('object'),self.close['step'] if self.close else -1
            if key not in self.last_event:
                event=dict(event,step=row['step'],observation_step=row['step']+1)
                self.last_event[key]=row['step'];self.events.append(event);fresh.append(event)
        consistent = []
        if g >= .2:
            for name in self.held:
                if len(contacts.get(name,()))<2:continue
                rels=[[r['positions'][name][i]-r['eef'][i] for i in range(3)] for r in recent if name in r['positions']]
                if len(rels)==WINDOW and max(distance(p,rels[-1]) for p in rels) <= .01:
                    consistent.append(name)
        open_free = (g <= -.2 and len(recent)==WINDOW and all(not any(r['contacts'].values()) for r in recent)
            and all(max(distance(r['positions'][name],p) for r in recent if name in r['positions']) <= .005
                    for name,p in positions.items()))
        normal = bool(open_free or consistent) and not emitted
        return dict(events=fresh,normal=normal,
            normal_kind='open_free_motion' if open_free else 'consistent_carry' if consistent else None,
            held_objects=list(self.held),contacts={k:len(v) for k,v in contacts.items() if v},
            close_step=self.close['step'] if self.close else None,
            ambiguous_closed_without_confirmed_object=bool(g >= .2 and not consistent and not fresh))
