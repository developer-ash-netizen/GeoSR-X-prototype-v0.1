import torch, rasterio
print('GeoSR-X environment check')
print('PyTorch:',torch.__version__)
print('CUDA:',torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
print('Rasterio:',rasterio.__version__)
