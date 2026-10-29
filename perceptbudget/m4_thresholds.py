"""Offline availability and energy boundaries; never installs runtime thresholds."""
from perceptbudget.m4_candidates import hold_energy


def total_energy(window,cell,waits,post_profile,down):
    duration=cell['mean_service_seconds'];remaining=window-duration
    if remaining<0:return None
    energy=cell['mean_absolute_service_energy_j']
    if cell['profile']==post_profile:
        return energy+hold_energy(waits,post_profile,0.,remaining)
    if (cell['profile'],post_profile)!=('baseline','gpu_emc'):raise ValueError('Unmeasured direction')
    if remaining<down['seconds']:return None
    return energy+down['energy_j']+hold_energy(waits,post_profile,down['seconds'],remaining-down['seconds'])


def service_deadline(cell,control_period,margin):
    # Same initial prior as M4: ready samples + one control cycle + eta.
    return cell['p95_service_seconds']+control_period+margin


def crossings(points):
    """Locate every sign change; do not assume a single energy threshold."""
    result=[]
    for a,b in zip(points,points[1:]):
        x,y=a;xx,yy=b
        if y==0:result.append(x)
        elif y*yy<0:result.append(x+(xx-x)*(-y)/(yy-y))
    return result


def scan(low,high,waits,down,*,control_period=.07,margin=.025,maximum=10.,step=.001):
    low_ready=service_deadline(low,control_period,margin)
    high_ready=service_deadline(high,control_period,margin)
    # Returning low inside the common window consumes a metered preparation block.
    high_return=high_ready+down['seconds']
    rows=[];differences=[]
    for i in range(1,round(maximum/step)+1):
        t=round(i*step,9)
        low_energy=total_energy(t,low,waits,'gpu_emc',down)
        high_energy=total_energy(t,high,waits,'baseline',down)
        return_energy=total_energy(t,high,waits,'gpu_emc',down)
        low_ok=t>=low_ready;high_ok=t>=high_ready;return_ok=t>=high_return
        if low_ok and return_ok:differences.append((t,return_energy-low_energy))
        choices=[(e,label) for label,e,ok in [('low_full_low',low_energy,low_ok),
            ('high_full_high',high_energy,high_ok),('high_full_low',return_energy,return_ok)] if ok and e is not None]
        rows.append(dict(window_seconds=t,low_j=low_energy,high_keep_j=high_energy,high_return_j=return_energy,
            low_feasible=low_ok,high_feasible=high_ok,high_return_feasible=return_ok,
            lowest=min(choices)[1] if choices else 'none'))
    return dict(low_ready_seconds=low_ready,high_ready_seconds=high_ready,
        high_return_code_seconds=high_return,high_return_readback_estimate_seconds=high_ready+down['max_readback_seconds'],
        equal_low_end_state_crossings_seconds=crossings(differences),
        min_high_return_minus_low_j=min((y for x,y in differences),default=None),rows=rows)
