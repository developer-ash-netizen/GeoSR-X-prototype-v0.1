import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def psnr(pred,target):
    return float(peak_signal_noise_ratio(target,pred,data_range=1.0))

def ssim(pred,target):
    return float(np.mean([
        structural_similarity(target[c],pred[c],data_range=1.0)
        for c in range(pred.shape[0])
    ]))

def sam(pred,target,eps=1e-8):
    p=pred.transpose(1,2,0); t=target.transpose(1,2,0)
    cos=np.sum(p*t,-1)/(np.linalg.norm(p,-1)*np.linalg.norm(t,-1)+eps)
    cos=np.clip(cos,-1+1e-6,1-1e-6)
    return float(np.mean(np.arccos(cos)))

def ergas(pred,target,scale=4,eps=1e-8):
    mse=np.mean((pred-target)**2,axis=(1,2))
    mean=np.mean(target,axis=(1,2))
    return float(100/scale*np.sqrt(np.mean(mse/(mean**2+eps))))
