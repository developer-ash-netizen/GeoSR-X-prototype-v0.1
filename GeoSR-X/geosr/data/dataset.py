from pathlib import Path
import rasterio
import torch
from torch.utils.data import Dataset

class PairedRasterDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr = sorted(Path(lr_dir).glob('*.tif'))
        self.hr_dir = Path(hr_dir)
        if not self.lr:
            raise FileNotFoundError(f'No .tif files found in {lr_dir}')

    def __len__(self):
        return len(self.lr)

    def _read(self, p):
        with rasterio.open(p) as src:
            return torch.from_numpy(src.read().astype('float32'))

    def __getitem__(self, i):
        lr = self.lr[i]
        hr = self.hr_dir / lr.name
        if not hr.exists():
            raise FileNotFoundError(f'Missing HR pair: {hr}')
        return {'lr': self._read(lr), 'hr': self._read(hr), 'name': lr.name}
