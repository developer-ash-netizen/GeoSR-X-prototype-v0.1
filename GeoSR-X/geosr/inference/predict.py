import argparse, torch
import torch.nn.functional as F
from geosr.data.raster import read_raster, write_raster
from geosr.models.geosr import GeoSRX

p=argparse.ArgumentParser()
p.add_argument('--input',required=True); p.add_argument('--output',required=True)
p.add_argument('--checkpoint',required=True); p.add_argument('--uncertainty-output')
p.add_argument('--channels',type=int,default=4); p.add_argument('--scale',type=int,default=4)
a=p.parse_args()
device='cuda' if torch.cuda.is_available() else 'cpu'
arr,profile=read_raster(a.input)
x=torch.from_numpy(arr).float().unsqueeze(0).to(device)
m=GeoSRX(a.channels,scale=a.scale).to(device)
ckpt=torch.load(a.checkpoint,map_location=device)
m.load_state_dict(ckpt['model']); m.eval()
with torch.no_grad(): out=m(x)
write_raster(a.output,out['sr'][0].cpu().numpy(),profile,a.scale)
if a.uncertainty_output:
    u=F.interpolate(out['uncertainty_t1'],size=out['sr'].shape[-2:],mode='bilinear',align_corners=False)
    write_raster(a.uncertainty_output,u[0].cpu().numpy(),profile,a.scale)
