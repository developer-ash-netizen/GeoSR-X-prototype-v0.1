"""L1 EVIDENCE TENSOR -- computed BEFORE any super-resolution.

  E1 temporal support      number of clean, state-compatible dates observing the pixel
  E2 disagreement          RMS temporal deviation (in noise sigmas) after removing structured change
  E3 change flag           is the variation STRUCTURED (single coherent step) or RANDOM (noise/outlier)?
  E4 spectral coherence    stability of the band relationship across compatible dates

C2 (change-aware evidence) lives here: a single-step fit separates real change from noise,
so real change is *protected* (dates on the other side of the step are simply not evidence
about the target state) instead of being penalised as disagreement.
"""
import warnings
import numpy as np
from scipy import ndimage as ndi


def estimate_noise(al, valid):
    """Robust per-band temporal noise sigma (median over pixels of per-pixel MAD)."""
    N, B, h, w = al.shape
    x = np.where(valid[:, None], al, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(x, axis=0)
        mad = np.nanmedian(np.abs(x - med[None]), axis=0) * 1.4826
    sig = np.array([np.nanmedian(mad[b][np.isfinite(mad[b])]) for b in range(B)])
    return np.maximum(sig, 1e-4)


def compute_evidence(al, valid, ref_idx, tol_px=0.3, z_out=4.0, min_step_sigma=6.0):
    N, B, h, w = al.shape
    sigma = estimate_noise(al, valid)
    M = valid.astype(np.float32)
    n = M.sum(0)

    x = np.where(valid[:, None], al, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, np.nanmean(al, axis=0))

    # tolerance grows with each date's own local gradient: sub-pixel sampling differences at
    # edges are SIGNAL, not disagreement. (Per-date, not from the median image: the median of
    # mixed-state pixels is spatially noisy and would inflate tolerance exactly where change is.)
    sm = ndi.gaussian_filter(al, (0, 0, 0.7, 0.7), mode="nearest")
    gy = ndi.sobel(sm, axis=2) / 8.0; gx = ndi.sobel(sm, axis=3) / 8.0
    grad = np.sqrt(gx ** 2 + gy ** 2)                           # (N,B,h,w)
    sig_eff = np.sqrt(sigma[None, :, None, None] ** 2 + (tol_px * grad) ** 2)

    dev = al - med[None]
    z = dev / sig_eff                                            # tolerance-scaled: used for outliers + E2
    zrms = np.sqrt((z ** 2).mean(1))                             # (N,h,w)

    # ---- structured-change test: best single step on NOISE-scaled deviations -------------
    # (statistical significance is judged against the fit's own residual, so no gradient
    #  tolerance is needed here; random sampling residuals are not date-ordered, so they do
    #  not masquerade as a step.)
    z0 = dev / sigma[None, :, None, None]
    Mz = M[:, None] * z0
    S1 = Mz.sum(0); Q = (M[:, None] * z0 ** 2).sum(0)           # (B,h,w)
    nn = np.maximum(n, 1)
    sse0 = (Q - S1 ** 2 / nn).sum(0)

    cum_n = np.cumsum(M, axis=0); cum_S = np.cumsum(Mz, axis=0)
    best = np.full((h, w), np.inf, np.float32)
    bestk = np.zeros((h, w), np.int16); step = np.zeros((h, w), np.float32)
    n1b = np.ones((h, w), np.float32); n2b = np.ones((h, w), np.float32)
    for k in range(2, N - 1):
        n1 = cum_n[k - 1]; n2 = n - n1
        ok = (n1 >= 2) & (n2 >= 2)
        S1a = cum_S[k - 1]; S2a = S1 - S1a
        sse = (Q - S1a ** 2 / np.maximum(n1, 1) - S2a ** 2 / np.maximum(n2, 1)).sum(0)
        m1 = S1a / np.maximum(n1, 1); m2 = S2a / np.maximum(n2, 1)
        st = np.sqrt(((m2 - m1) ** 2).mean(0))
        upd = ok & (sse < best)
        best = np.where(upd, sse, best); bestk = np.where(upd, k, bestk)
        step = np.where(upd, st, step); n1b = np.where(upd, n1, n1b); n2b = np.where(upd, n2, n2b)

    r2 = np.where(sse0 > 1e-6, 1 - best / np.maximum(sse0, 1e-6), 0.0)
    resid_rms = np.maximum(np.sqrt(np.maximum(best, 0) / (nn * B)), 1.0)
    tstat = step / (resid_rms * np.sqrt(1 / n1b + 1 / n2b))
    structured = (n >= 4) & np.isfinite(best) & (r2 > 0.6) & (step > min_step_sigma) & (tstat > 4.0)

    side_ref = ref_idx >= bestk                                    # target-date state
    dates = np.arange(N)[:, None, None]
    left = (dates < bestk[None])
    same_side = (~left) == side_ref[None]
    compat = valid & np.where(structured[None], same_side, zrms <= z_out)

    # E2: RMS disagreement (tolerance-scaled sigmas) after removing the protected step
    Mv = M[:, None]
    nl = np.maximum((M * left).sum(0), 1); nr = np.maximum((M * ~left).sum(0), 1)
    ml = (Mv * left[:, None] * z).sum(0) / nl; mr = (Mv * (~left)[:, None] * z).sum(0) / nr
    mall = (Mv * z).sum(0) / nn
    fit = np.where(structured[None, None], np.where(left[:, None], ml[None], mr[None]), mall[None])
    sse_z = (Mv * (z - fit) ** 2).sum((0, 1))
    E2 = np.sqrt(sse_z / (nn * B))
    E2 = np.where(n >= 2, E2, 3.0).astype(np.float32)
    E1 = compat.sum(0).astype(np.float32)

    # E4 spectral coherence over compatible dates
    v = al / (np.linalg.norm(al, axis=1, keepdims=True) + 1e-6)
    C = compat[:, None].astype(np.float32)
    vm = (v * C).sum(0) / np.maximum(C.sum(0), 1)
    vm = vm / (np.linalg.norm(vm, axis=0, keepdims=True) + 1e-6)
    ang = np.arccos(np.clip((v * vm[None]).sum(1), -1, 1))       # (N,h,w)
    mang = (ang * compat).sum(0) / np.maximum(E1, 1)
    E4 = np.where(E1 >= 2, np.exp(-mang / 0.06), 0.5).astype(np.float32)

    # sub-pixel sampling diversity among compatible dates (needed for a genuine multi-date gain)
    return dict(E1=E1, E2=E2, E3=structured, E3_r2=r2.astype(np.float32), E3_step=step, E3_k=bestk,
                E4=E4, compat=compat, sigma=sigma, sig_eff=sig_eff, med=med)


def sampling_diversity(compat, shifts, n_cells=2):
    """How many distinct sub-pixel phase cells (n_cells x n_cells) the compatible dates cover, per pixel."""
    fr = np.mod(shifts, 1.0)
    cell = (np.floor(fr * n_cells).astype(int)) % n_cells
    cid = cell[:, 0] * n_cells + cell[:, 1]
    out = np.zeros(compat.shape[1:], np.float32)
    for c in range(n_cells * n_cells):
        sel = cid == c
        if sel.any():
            out += compat[sel].any(0)
    return out
