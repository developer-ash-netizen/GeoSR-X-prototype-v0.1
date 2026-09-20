"""L4 evidence gate, L5 provenance + hallucination risk, L6 self-consistency."""
import numpy as np
from scipy import ndimage as ndi
from .regions import rmean, up

MULTI, SINGLE, PRIOR = 0, 1, 2
PROV_NAMES = {MULTI: "multi-date observed", SINGLE: "single-date observed", PRIOR: "prior-generated"}


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ L4
def evidence_gate(E1, E2n, E3, E4, alpha_max=1.0):
    """alpha = f(E1..E4). Gates on L1 ONLY -- never looks at L3 outputs.

    strong evidence  -> admit the prior path's detail (it is checkable against real data)
    weak evidence    -> stay on the fidelity path (no unverifiable detail)
    structured change with thin support -> alpha = 0 (a new structure seen on few dates gets no invented texture)
    """
    E1n = np.clip(E1 / 4.0, 0, 1)
    e2 = np.exp(-np.maximum(E2n - 1.0, 0) / 1.5)
    score = 0.45 * E1n + 0.25 * e2 + 0.30 * E4
    a = _smoothstep((score - 0.55) / 0.35)
    a = np.where(E3 & (E1 < 3), 0.0, a)
    return (a * alpha_max).astype(np.float32)


# ------------------------------------------------------------------ L5
def region_features(lab, n, s, ev, E2n, alpha_lr, div, fid, prior, sr, sig_noise):
    """Region-level evidence + audit signals. Returns dict of arrays (n,)."""
    E1 = rmean(up(ev["E1"], s), lab, n); E2 = rmean(up(E2n, s), lab, n)
    E3f = rmean(up(ev["E3"].astype(np.float32), s), lab, n); E4 = rmean(up(ev["E4"], s), lab, n)
    dv = rmean(up(div, s), lab, n)
    al = rmean(up(alpha_lr, s), lab, n)

    hf_fid = np.abs(fid - ndi.gaussian_filter(fid, (0, 2, 2), mode="nearest")).mean(0)
    dp = np.abs(prior - fid).mean(0)                    # UNWEIGHTED prior deviation: independent of the gate
    dp_gate = np.abs(al_map(alpha_lr, s) * (prior - fid)).mean(0)   # what actually got blended in
    hf_fid_r = rmean(hf_fid, lab, n); dp_r = rmean(dp, lab, n); dpg_r = rmean(dp_gate, lab, n)
    fp = dpg_r / (dpg_r + hf_fid_r + 1e-4)              # share of the detail that came from the prior

    hf_sr = np.abs(sr - ndi.gaussian_filter(sr, (0, 1.5, 1.5), mode="nearest")).mean(0)
    tex = rmean(hf_sr, lab, n)
    size = np.bincount(lab.ravel(), minlength=n).astype(np.float32)
    # ---- hallucination risk: INDEPENDENT signals only (alpha is deliberately not an input, P2)
    r_t = np.clip((E2 - 1.0) / 2.0, 0, 1)                                   # temporal disagreement
    r_s = np.clip((1.0 - E4) / 0.8, 0, 1)                                   # spectral inconsistency
    r_x = np.clip(dp_r / (dp_r + hf_fid_r + 1e-4) * 1.6 - 0.2, 0, 1)        # texture novelty (prior vs observation)
    risk = 0.4 * r_t + 0.3 * r_s + 0.3 * r_x
    return dict(E1=E1, E2=E2, E3f=E3f, E4=E4, div=dv, alpha=al, fp=fp, r_t=r_t, r_s=r_s, r_x=r_x,
                risk=risk, tex=tex, size=size)


def al_map(alpha_lr, s):
    return up(alpha_lr, s)[None]


def provenance_head(rf, e1_multi=3.0, fp_prior=0.40, min_cells=2):
    """Lightweight rule head over the evidence tensor (transparent thresholds, trained/tuned
    separately from the SR model so provenance can never destabilise SR training)."""
    prov = np.full(len(rf["E1"]), SINGLE, np.int8)
    prov[(rf["E1"] >= e1_multi) & (rf["div"] >= min_cells - 0.5)] = MULTI
    prov[(rf["fp"] >= fp_prior) | (rf["E1"] < 0.5)] = PRIOR
    return prov


# ------------------------------------------------------------------ L6
def self_consistency(sensor, sr, lr_ref, ref_valid, sig_noise, lab, n, s):
    """Forward-degrade SR through the sensor model, compare to the real target-date observation."""
    pred = sensor.forward(sr, (0, 0))
    r = np.sqrt((((lr_ref - pred) / sig_noise[:, None, None]) ** 2).mean(0))       # (h,w) RMS in sigmas
    r = np.where(ref_valid, r, 0.0)
    w = ref_valid.astype(np.float32)
    num = rmean(up(r * w, s), lab, n); den = rmean(up(w, s), lab, n)
    res = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0.0)
    inflate = 1.0 + 0.5 * np.clip(res - 1.5, 0, 4)
    return res, inflate, r
