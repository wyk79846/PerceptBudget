from perceptbudget.fast_gripper_gate import command_state
from perceptbudget.m3_service_choice import choose_service
from test_m3_service_choice import fixture


def row(step,grip,source=True):
    return dict(step=step,action=[0.]*6+[grip],source={'prediction_id':0} if source else None)


def test_executed_commands_not_future_commands_or_underflow():
    assert command_state([])['state']=='unknown'
    assert not command_state([])['allowed']
    assert command_state([row(0,-1),row(1,.1)])['allowed']
    gate=command_state([row(0,-1),row(1,1),row(2,-1,False)])
    assert gate['state']=='closed' and gate['last_command_step']==1
    assert command_state([row(0,1),row(1,-1)])['allowed']


def test_closed_and_unknown_force_full_open_keeps_energy_choice():
    costs,d=fixture('gpu_emc')
    for rows,mode in [([], 'Full'),([row(1,1)],'Full'),([row(1,-1)],'Fast')]:
        result=choose_service(dict(d,fast_gripper_gate=command_state(rows)),observation_step=64,previous_mode='Full',
            cost_model=costs,quantiles={'Full':.55,'Fast':.35},latest_start_cap=1.95)
        assert result['mode']==mode
