import numpy as np
import torch

from perceptbudget.grip_occupancy import GripOccupancyMonitor, GripOccupancyCNN


IMAGE=np.zeros((256,256,3),dtype=np.uint8)


def monitor_with_predictions(probabilities):
    values=iter(probabilities)
    return GripOccupancyMonitor(predictor=lambda _:torch.tensor([next(values)]),confidence=.95)


def control(monitor,step,z=0.,g=1.):
    state=[0.,0.,z,0.,0.,0.,.01,-.01]
    action=[0.,0.,0.,0.,0.,0.,g]
    monitor.on_control(action,state,step=step)
    return state,action


def sample(monitor,step,z=0.,g=1.):
    state,action=control(monitor,step,z,g)
    return monitor.observe(IMAGE,state,action,step=step)


def test_closure_immediately_images_but_empty_approach_is_not_an_alarm():
    monitor=monitor_with_predictions([[.01,.99],[.01,.99],[.01,.99],[.01,.99]])
    control(monitor,2,g=-1.)
    control(monitor,3)
    assert monitor.due(3)
    assert not sample(monitor,3)['alarm']
    assert not monitor.due(4)
    assert not sample(monitor,8)['alarm']  # Closing near the table is not failed carry.
    assert not sample(monitor,13,z=.03)['alarm']
    row=sample(monitor,18,z=.04)
    assert row['alarm'] and row['carrying_window']
    assert not monitor.due(23)  # One alarm per closure; no repeated replanning loop.


def test_unknown_breaks_empty_confirmation_and_release_starts_new_cycle():
    monitor=monitor_with_predictions([[.01,.99],[.5,.5],[.01,.99],[.01,.99],[.01,.99]])
    control(monitor,1,g=-1.)
    assert not sample(monitor,2,z=.03)['alarm']
    row=sample(monitor,7,z=.03)
    assert row['predicted_state']=='unknown' and row['confirmation_count']==0
    assert not sample(monitor,12,z=.03)['alarm']
    assert sample(monitor,17,z=.03)['alarm']
    control(monitor,18,z=.03,g=-1.)
    assert not monitor.due(19)
    row=sample(monitor,19,z=.04)
    assert row['close_step']==19 and not row['carrying_window'] and not row['alarm']


def test_held_and_carrying_descent_do_not_trigger_empty_alarm():
    monitor=monitor_with_predictions([[.99,.01],[.99,.01],[.99,.01]])
    control(monitor,1,g=-1.)
    for step,z in [(2,.04),(7,.02),(12,-.01)]:
        row=sample(monitor,step,z)
        assert row['carrying_window'] and not row['alarm']


def test_cpu_checkpoint_loading_does_not_change_vla_rng(tmp_path):
    path=tmp_path/'model.pt'
    torch.save(dict(state_dict=GripOccupancyCNN().state_dict(),confidence=.95,input_size=128),path)
    state=torch.random.get_rng_state()
    monitor=GripOccupancyMonitor(path)
    assert torch.equal(torch.random.get_rng_state(),state)
    assert all(p.device.type=='cpu' for p in monitor.model.parameters())
