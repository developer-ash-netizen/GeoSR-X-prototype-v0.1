import argparse
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from geosr.data.dataset import PairedRasterDataset
from geosr.models.geosr import GeoSRX
from geosr.losses.combined import geosr_loss

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/base.yaml'); a=p.parse_args()
cfg=yaml.safe_load(open(a.config,encoding='utf-8'))
device=cfg['training']['device']
if device=='cuda' and not torch.cuda.is_available(): device='cpu'
ds=PairedRasterDataset(cfg['data']['train_lr'],cfg['data']['train_hr'])
loader=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=True,
                  num_workers=cfg['training']['num_workers'])
m=GeoSRX(cfg['model']['channels'],cfg['model']['features'],
         cfg['model']['blocks'],cfg['model']['scale']).to(device)
opt=torch.optim.AdamW(m.parameters(),lr=cfg['training']['lr'],weight_decay=1e-4)
Path('checkpoints').mkdir(exist_ok=True)
for epoch in range(cfg['training']['epochs']):
    m.train(); total=0
    for b in tqdm(loader,desc=f'Epoch {epoch+1}'):
        lr,hr=b['lr'].to(device),b['hr'].to(device)
        opt.zero_grad(set_to_none=True)
        out=m(lr)
        loss,parts=geosr_loss(out,lr,hr,cfg['loss']['pixel'],
                              cfg['loss']['consistency'],cfg['loss']['spectral'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        total+=loss.item()
    print(f'Epoch {epoch+1}: {total/len(loader):.6f}')
    torch.save({'epoch':epoch+1,'model':m.state_dict(),
                'optimizer':opt.state_dict(),'config':cfg},
               'checkpoints/geosr_x_latest.pt')
