import torch.nn.functional as F

class ObservationConsistency:
    def __init__(self, scale=4):
        self.scale = scale
    def __call__(self, sr):
        return F.interpolate(sr, scale_factor=1/self.scale, mode='area')
