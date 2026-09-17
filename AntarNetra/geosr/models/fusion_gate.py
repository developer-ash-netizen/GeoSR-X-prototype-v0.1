import torch
import torch.nn as nn

class FusionGate(nn.Module):
    def __init__(self, channels=4, features=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels*2,features,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(features,features,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(features,1,1), nn.Sigmoid())

    def forward(self, detail, consistency):
        gate = self.net(torch.cat([detail, consistency],1))
        out = gate*detail + (1-gate)*consistency
        disagreement = torch.mean(torch.abs(detail-consistency),1,keepdim=True)
        return out, gate, disagreement
