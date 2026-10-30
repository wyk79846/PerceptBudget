"""Executed command hint only; does not infer physical contact or grasp success."""

def command_state(rows,threshold=.2):
    last=next((r for r in reversed(rows) if r.get('source') is not None and abs(r['action'][6])>=threshold),None)
    state='unknown' if last is None else 'closed' if last['action'][6]>0 else 'open'
    return dict(state=state,allowed=state=='open',last_command_step=None if last is None else last['step'],
                last_command=None if last is None else float(last['action'][6]),threshold=threshold)
