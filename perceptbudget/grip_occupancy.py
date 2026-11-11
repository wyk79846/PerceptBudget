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
