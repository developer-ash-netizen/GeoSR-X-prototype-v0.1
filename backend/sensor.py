"""Sensor model: the forward operator A_d and its adjoint.

    LR_d = pool_s( blur_psf( shift_d( HR ) ) ) + noise

`shift_d` is the registration shift of date d relative to the target date (in LR pixels).
This module is the single place the degradation lives, so the fidelity constraint (L3a),
the self-consistency check (L6) and leave-one-date-out validation (Tier B) all use the
same operator. To use ESA's `opensr-degradation`, subclass SensorModel and override
forward()/adjoint() -- nothing else in the pipeline changes.
"""
import numpy as np
from scipy import ndimage as ndi


class SensorModel:
    def __init__(self, scale=4, psf_sigma=1.5):
        self.scale = scale
        self.psf_sigma = psf_sigma

    def forward(self, hr, shift_lr=(0.0, 0.0)):
        """hr: (B,H,W) -> lr: (B,H/s,W/s)"""
        s = self.scale
        x = hr
        sy, sx = shift_lr
        if sy or sx:
            # native sample q sees target-frame position q + shift
            x = ndi.shift(x, (0, -sy * s, -sx * s), order=1, mode="nearest")
        x = ndi.gaussian_filter(x, (0, self.psf_sigma, self.psf_sigma), mode="nearest")
        B, H, W = x.shape
        return x.reshape(B, H // s, s, W // s, s).mean((2, 4))

    def adjoint(self, lr, shift_lr=(0.0, 0.0)):
        """Back-projection: (B,h,w) -> (B,H,W)"""
        s = self.scale
        up = np.repeat(np.repeat(lr, s, axis=-2), s, axis=-1)
        up = ndi.gaussian_filter(up, (0, self.psf_sigma, self.psf_sigma), mode="nearest")
        sy, sx = shift_lr
        if sy or sx:
            up = ndi.shift(up, (0, sy * s, sx * s), order=1, mode="nearest")
        return up

    def upsample(self, lr):
        """Bicubic upsample aligned to pixel centres."""
        s = self.scale
        return ndi.zoom(lr, (1, s, s), order=3, grid_mode=True, mode="nearest")
