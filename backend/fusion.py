"""L2 TEMPORAL FUSION + L3a FIDELITY PATH.

Multi-frame super-resolution by weighted iterative back-projection (IBP) through the sensor
model. This is the *measured* path: every HR value it produces is pulled toward agreement
with real observations, and the data-consistency projection at the end makes
degrade(SR) == LR the operating constraint rather than a soft loss term.

In the full system the attention-based fusion network replaces the IBP initialiser; the
interface (weights in, HR estimate out, degrade-consistency preserved) is unchanged.
"""
import numpy as np
from scipy import ndimage as ndi


def _fill_nearest(x, valid):
    if valid.all():
        return x
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return x[:, idx[0], idx[1]]


def fuse(sensor, lr_norm, aligned_norm, shifts, weights_ref, weights_native, ref_idx,
         n_iter=12, step=1.0, fidelity_projection=2, mode="temporal"):
    """
    lr_norm       (N,B,h,w) radiometrically-matched native-frame LR stack
    aligned_norm  (N,B,h,w) same, resampled into target frame (for the initial estimate)
    weights_ref   (N,h,w)   evidence weights in the target frame (init composite)
    weights_native(N,h,w)   evidence weights in each date's native frame (data term)
    Returns HR estimate (B,H,W) and per-iteration RMS residual history.
    """
    N, B, h, w = lr_norm.shape
    s = sensor.scale
    wsum = weights_ref.sum(0)
    comp = (aligned_norm * weights_ref[:, None]).sum(0) / np.maximum(wsum, 1e-6)[None]
    comp = _fill_nearest(comp, wsum > 1e-3)
    X = sensor.upsample(comp)
    hist = []
    for it in range(n_iter):
        acc = np.zeros_like(X); den = np.zeros((1,) + X.shape[1:], X.dtype)
        sq = 0.0; cnt = 0.0
        for d in range(N):
            wd = weights_native[d]
            if wd.max() <= 1e-3:
                continue
            sh = tuple(shifts[d])
            res = (lr_norm[d] - sensor.forward(X, sh)) * wd[None]
            acc += sensor.adjoint(res, sh)
            den += sensor.adjoint(wd[None], sh)
            sq += float(((res / np.maximum(wd[None], 1e-3)) ** 2 * wd[None]).sum()); cnt += float(wd.sum() * B)
        X = X + step * acc / np.maximum(den, 0.05)
        hist.append(float(np.sqrt(sq / max(cnt, 1))))
    # hard-constraint style projection onto {X : A_ref X = LR_ref} where the target date is observed
    wref = weights_native[ref_idx]
    for _ in range(fidelity_projection):
        res = (lr_norm[ref_idx] - sensor.forward(X, (0, 0))) * (wref > 0.5)[None]
        X = X + sensor.adjoint(res, (0, 0))
    return np.clip(X, 0.0, 1.0), hist


def native_weights(compat, quality, shifts, valid_native):
    """Move target-frame evidence weights into each date's own frame."""
    N = compat.shape[0]
    out = np.zeros_like(compat, dtype=np.float32)
    for d in range(N):
        w = compat[d].astype(np.float32) * float(quality[d])
        wn = ndi.shift(w, (-shifts[d][0], -shifts[d][1]), order=1, mode="nearest")
        out[d] = np.clip(wn, 0, 1) * valid_native[d]
    return out
