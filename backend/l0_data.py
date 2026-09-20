"""L0 DATA FOUNDATION: per-date quality, sub-pixel co-registration, radiometric matching.

Alignment is verified *before* any metric is trusted: every date gets a post-registration
correlation score and dates that fail are dropped from fusion.

Adapters: in production `cubo` supplies the stack and `s2cloudless` the mask
(see pipeline_io.py). The pipeline only needs (lr, cloud) arrays.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation


def quality_scores(cloud):
    frac = 1.0 - cloud.reshape(cloud.shape[0], -1).mean(1)
    return frac  # clean fraction per date in [0,1]


def pick_target(clean_frac, min_clean=0.9):
    N = len(clean_frac)
    for d in range(N - 1, -1, -1):
        if clean_frac[d] >= min_clean:
            return d
    return int(np.argmax(clean_frac))


def _hp(img, m, sig=3.0):
    mf = m.astype(np.float64)
    bl = ndi.gaussian_filter(img * mf, sig) / np.maximum(ndi.gaussian_filter(mf, sig), 1e-3)
    return (img - bl) * mf


def _lk_refine(ref, mv, m, sh, iters=6):
    """Translation Lucas-Kanade refinement on high-passed, lightly smoothed images (sub-pixel)."""
    sh = np.array(sh, float)
    for _ in range(iters):
        w = ndi.shift(mv, sh, order=3, mode="nearest")
        wm = ndi.shift(m.astype(float), sh, order=1, mode="constant", cval=0.0) > 0.999
        mm = wm & m
        mm = ndi.binary_erosion(mm, iterations=2)
        if mm.sum() < 100:
            break
        gy, gx = np.gradient(ndi.gaussian_filter(w, 0.8))
        r = (ref - w)[mm]
        g = np.stack([gy[mm], gx[mm]], 1)
        A = g.T @ g + 1e-9 * np.eye(2)
        d = np.linalg.solve(A, -g.T @ r)
        sh = sh + d
        if np.abs(d).max() < 1e-3:
            break
    return sh


def coregister(lr, valid, ref_idx, upsample=100, min_ncc=0.55):
    """Sub-pixel shift of every date to the target-date frame (LR pixels).

    Stage 1: masked cross-correlation (skimage) with 1/100 px upsampling.
    Stage 2: Lucas-Kanade refinement.
    Returns shifts (N,2) with the ndi.shift convention (ndi.shift(date, shift) ~ target),
    post-alignment NCC per date and a `usable` flag. Dates below `min_ncc` are dropped.
    """
    N, B, h, w = lr.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    shifts = np.zeros((N, 2)); ncc = np.ones(N)
    refimg = lr[ref_idx].mean(0)
    for d in range(N):
        if d == ref_idx:
            continue
        m = valid[d] & valid[ref_idx]
        a = _hp(refimg, m); b = _hp(lr[d].mean(0), m)
        sh, _, _ = phase_cross_correlation(a * win, b * win, upsample_factor=upsample, normalization=None)
        sh = _lk_refine(a, b, m, sh)
        shifts[d] = sh
    aligned, avalid = align_stack(lr, valid, shifts)
    for d in range(N):
        if d == ref_idx:
            continue
        m = avalid[d] & avalid[ref_idx]
        m[:4] = m[-4:] = False; m[:, :4] = m[:, -4:] = False
        if m.sum() < 50:
            ncc[d] = 0.0; continue
        a = ndi.gaussian_filter(aligned[d].mean(0), 0.7)[m]; b = ndi.gaussian_filter(aligned[ref_idx].mean(0), 0.7)[m]
        a = a - a.mean(); b = b - b.mean()
        ncc[d] = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
    usable = ncc >= min_ncc
    usable[ref_idx] = True
    return shifts, ncc, usable


def align_stack(stack, valid, shifts):
    N = stack.shape[0]
    out = np.empty_like(stack)
    ov = np.empty_like(valid)
    for d in range(N):
        sh = tuple(shifts[d])
        out[d] = ndi.shift(stack[d], (0, *sh), order=3, mode="nearest")
        v = ndi.shift(valid[d].astype(np.float32), sh, order=1, mode="constant", cval=0.0)
        ov[d] = v > 0.999          # only trust pixels whose whole interpolation support is valid
        ov[d] = ndi.binary_erosion(ov[d], iterations=1, border_value=0) | (v > 0.999)
    return out, ov


def match_radiometry(lr, aligned, avalid, ref_idx):
    """Per-date per-band gain/offset to the target date via trimmed least squares."""
    N, B = lr.shape[:2]
    gain = np.ones((N, B)); off = np.zeros((N, B))
    for d in range(N):
        if d == ref_idx:
            continue
        m = avalid[d] & avalid[ref_idx]
        if m.sum() < 100:
            continue
        for b in range(B):
            x = aligned[d, b][m]; y = aligned[ref_idx, b][m]
            keep = np.ones_like(x, bool)
            g, o = 1.0, 0.0
            for _ in range(4):
                A = np.c_[x[keep], np.ones(keep.sum())]
                g, o = np.linalg.lstsq(A, y[keep], rcond=None)[0]
                r = y - (g * x + o)
                s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
                keep = np.abs(r - np.median(r[keep])) < 2.5 * s
            gain[d, b], off[d, b] = g, o
    ln = lr * gain[:, :, None, None] + off[:, :, None, None]
    an = aligned * gain[:, :, None, None] + off[:, :, None, None]
    return ln.astype(np.float32), an.astype(np.float32), gain, off
