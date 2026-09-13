import argparse, torch
import torch.nn.functional as F
from geosr.data.raster import read_raster, write_raster

p=argparse.ArgumentParser()
p.add_argument('--input',required=True); p.add_argument('--output',required=True)
p.add_argument('--scale',type=int,default=4)
a=p.parse_args()
arr,profile=read_raster(a.input)
x=torch.from_numpy(arr).unsqueeze(0)
with torch.no_grad():
    y=F.interpolate(x,scale_factor=a.scale,mode='bicubic',align_corners=False)
write_raster(a.output,y[0].numpy(),profile,a.scale)
