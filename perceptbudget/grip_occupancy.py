"""Small, optional RGB-only classifier; independent of the VLA and FlashRT."""
import torch
from torch import nn


class GripOccupancyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks=[];previous=3
        for width in (8,16,32,64):
            blocks.extend([nn.Conv2d(previous,width,3,stride=2,padding=1,bias=False),
                           nn.BatchNorm2d(width),nn.ReLU()])
            previous=width
        self.features=nn.Sequential(*blocks)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),
                                nn.Linear(256,32),nn.ReLU(),nn.Linear(32,2))

    def forward(self,images):
        return self.head(self.features(images))


def occupancy_state(probabilities,confidence):
    """Return held/empty/unknown; confidence is calibrated, not a visibility oracle."""
    score,index=probabilities.max(-1)
    return [(("held","empty")[int(i)] if float(p)>=confidence else 'unknown')
            for p,i in zip(score,index)]


class GripOccupancyMonitor:
    """RGB occupancy during a commanded grasp, independent of simulator labels.

    Start imaging on closure, but alarm only after actual upward arm motion.
    Unknown predictions break confirmation; an opening command resets the cycle.
    CPU execution keeps the CNN off the VLA's GPU stream during Continue.
    """
    def __init__(self, checkpoint=None, *, interval=5, confirmation=2, predictor=None,
                 confidence=None):
        import time
        self.interval=interval;self.confirmation=confirmation
        self.close=None;self.last_control=None;self.last_step=None
        self.bad=0;self.alarmed=False;self.samples=[]
        self.predictor=predictor
        self.model=None
        if predictor is None:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            with torch.random.fork_rng(devices=[]):
                self.model=GripOccupancyCNN().eval()
            self.model.load_state_dict(saved['state_dict'])
            self.confidence=float(saved['confidence'])
            self.input_size=int(saved['input_size'])
        else:
            self.confidence=float(confidence);self.input_size=128
        self.loaded_monotonic=time.monotonic()

    def on_control(self, action, state=None, *, step=None):
        g=float(action[6]);z=float(state[2])
        if g<=-.2:
            self.close=None;self.bad=0;self.alarmed=False;self.last_step=None
        elif g>=.2 and self.close is None:
            self.close=dict(step=step,z=self.last_control['z'] if self.last_control else z,
                            carrying_window=False)
        if self.close is not None and g>=.2 and z-self.close['z']>=.02:
            self.close['carrying_window']=True
        self.last_control=dict(step=step,z=z,g=g)

    def due(self, step):
        return bool(self.close is not None and not self.alarmed
                    and (self.last_step is None or step-self.last_step>=self.interval))

    def grasp_check_due(self):
        return False

    def observe(self, image, state, action, *, step, eligible=True):
        import time
        import numpy as np
        from PIL import Image
        started=time.monotonic()
        if self.close is None or float(action[6])<.2:
            raise ValueError('Occupancy images require an active closed-gripper cycle')
        rgb=np.asarray(Image.fromarray(image).resize((self.input_size,self.input_size),
                       Image.Resampling.BILINEAR))
        inputs=torch.from_numpy(rgb.transpose(2,0,1).copy()).float().unsqueeze(0)/255.
        prepared=time.monotonic()
        with torch.inference_mode():
            probs=(self.predictor(inputs) if self.predictor is not None
                   else self.model(inputs).softmax(1))
        probabilities=probs[0].tolist()
        predicted=occupancy_state(probs,self.confidence)[0]
        stage=bool(self.close['carrying_window'])
        self.bad=self.bad+1 if eligible and stage and predicted=='empty' else 0
        alarm=bool(not self.alarmed and self.bad>=self.confirmation)
        if alarm:self.alarmed=True
        row=dict(step=step,eligible=bool(eligible),alarm=alarm,
            reason='cnn_confirmed_empty_carry' if alarm else 'cnn_'+predicted,
            predicted_state=predicted,probabilities=probabilities,
            confidence_threshold=self.confidence,confirmation_count=self.bad,
            close_step=self.close['step'],close_z_m=self.close['z'],
            eef_lift_m=float(state[2])-self.close['z'],carrying_window=stage,
            gripper_command=float(action[6]),input_kind='raw_wrist_rgb',device='cpu',
            input_shape=list(rgb.shape),preprocess_seconds=prepared-started,
            forward_seconds=time.monotonic()-prepared,detector_seconds=time.monotonic()-started)
        self.samples.append(row);self.last_step=step
        return row
