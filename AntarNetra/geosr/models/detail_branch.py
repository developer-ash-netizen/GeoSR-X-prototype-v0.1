import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c,c,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(c,c,3,1,1))
    def forward(self,x):
        return x + self.body(x)

class DetailBranch(nn.Module):
    # Phase-1 lightweight branch; replaceable by SwinIR/Swin2MoSE later.
    def __init__(self, in_channels=4, features=64, blocks=8, scale=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, features, 3,1,1)
        self.body = nn.Sequential(*[ResidualBlock(features) for _ in range(blocks)])
        self.tail_body = nn.Conv2d(features,features,3,1,1)
        self.up = nn.Sequential(
            nn.Conv2d(features, features*scale*scale,3,1,1),
            nn.PixelShuffle(scale), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(features,in_channels,3,1,1)

    def forward(self,x):
        f = self.head(x)
        f = f + self.tail_body(self.body(f))
        return self.out(self.up(f))
