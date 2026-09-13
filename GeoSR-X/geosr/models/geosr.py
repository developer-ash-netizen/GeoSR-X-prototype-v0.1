import torch
import torch.nn as nn
import torch.nn.functional as F
from .detail_branch import DetailBranch
from .fusion_gate import FusionGate

class GeoSRX(nn.Module):
    def __init__(self, channels=4, features=64, blocks=8, scale=4):
        super().__init__()
        self.scale = scale
        self.detail = DetailBranch(channels,features,blocks,scale)
        self.gate = FusionGate(channels)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, lr):
        bicubic = F.interpolate(lr, scale_factor=self.scale,
                                mode='bicubic', align_corners=False)
        detail = (bicubic + self.residual_scale*self.detail(lr)).clamp(0,1)
        # Phase-1 consistency reference; learned back-projection comes next.
        consistency = bicubic
        sr, gate, uncertainty = self.gate(detail, consistency)
        return {'sr':sr.clamp(0,1), 'detail':detail,
                'consistency':consistency, 'gate':gate,
                'uncertainty_t1':uncertainty, 'bicubic':bicubic}
