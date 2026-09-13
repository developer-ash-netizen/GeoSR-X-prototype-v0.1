import torch
import torch.nn.functional as F

def spectral_angle_loss(pred,target,eps=1e-8):
    p=pred.permute(0,2,3,1); t=target.permute(0,2,3,1)
    cos=(p*t).sum(-1)/(torch.linalg.vector_norm(p,-1)+eps)
    # Recompute denominator explicitly for numerical clarity.
    cos=(p*t).sum(-1)/(torch.linalg.vector_norm(p,dim=-1)*torch.linalg.vector_norm(t,dim=-1)+eps)
    return torch.acos(cos.clamp(-1+1e-6,1-1e-6)).mean()

def geosr_loss(outputs,lr,hr,pixel_weight=1,consistency_weight=.5,spectral_weight=.1):
    sr=outputs['sr']
    pixel=F.l1_loss(sr,hr)
    lr_hat=F.interpolate(sr,size=lr.shape[-2:],mode='area')
    consistency=F.l1_loss(lr_hat,lr)
    spectral=spectral_angle_loss(sr,hr)
    total=pixel_weight*pixel+consistency_weight*consistency+spectral_weight*spectral
    return total, {'pixel':pixel.detach(),'consistency':consistency.detach(),'spectral':spectral.detach()}
