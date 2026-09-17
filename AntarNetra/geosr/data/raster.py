from pathlib import Path
import numpy as np
import rasterio
import torch

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    return arr, profile

def write_raster(path, arr, profile, scale=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(driver='GTiff', dtype='float32', count=arr.shape[0],
             height=arr.shape[1], width=arr.shape[2], compress='deflate')
    if scale != 1:
        p['transform'] = p['transform'] * p['transform'].scale(1/scale, 1/scale)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(arr.astype(np.float32))

def normalize_percentile(x, low=2, high=98):
    out = torch.empty_like(x)
    for c in range(x.shape[0]):
        lo = torch.quantile(x[c], low/100)
        hi = torch.quantile(x[c], high/100)
        out[c] = ((x[c]-lo)/(hi-lo+1e-6)).clamp(0,1)
    return out
