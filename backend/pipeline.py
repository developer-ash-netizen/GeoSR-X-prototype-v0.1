"""Orchestrator: L0 -> L8 for one scene. Stage callbacks drive the UI progress rail."""
import time
import numpy as np
from scipy import ndimage as ndi
from .sensor import SensorModel
from . import l0_data as L0, l1_evidence as L1, fusion as F
from . import l4_l6 as L
from .regions import make_regions, rmean, up
from .l7_calibration import classify_regions, featmat, CLASSES

STAGES = [
    ("L0", "Data foundation", "cloud mask, quality, sub-pixel co-registration"),
    ("L1", "Evidence tensor", "E1 support, E2 disagreement, E3 change, E4 spectral"),
    ("L2", "Temporal fusion", "multi-date fusion through the sensor model"),
    ("L3", "Fidelity + prior paths", "constrained path and learned-detail path"),
    ("L4", "Evidence gate", "alpha from L1 only"),
    ("L5", "Provenance + risk", "3-class provenance, independent hallucination risk"),
    ("L6", "Self-consistency", "forward-degrade SR against the observation"),
    ("L7", "Calibration", "region-level split-conformal bound"),
    ("L8", "Analysis-ready output", "GeoTIFF + sidecar"),
    ("L9", "Validation", "Tier A / B / C"),
]


class Result:
    pass


def _tick(cb, stage, status, info=None, t0=None):
    if cb:
        cb(stage, status, info or {}, (time.time() - t0) if t0 else None)


def prepare(scene, sensor, target=None, cb=None, res=None):
    """L0 + L1. Returns a namespace with everything downstream needs."""
    R = res or Result()
    t0 = time.time(); _tick(cb, "L0", "run")
    valid = ~scene.cloud
    R.clean = L0.quality_scores(scene.cloud)
    R.ref = int(target) if target is not None else L0.pick_target(R.clean)
    R.shifts, R.ncc, R.usable = L0.coregister(scene.lr, valid, R.ref)
    valid = valid & R.usable[:, None, None]
    R.valid = valid
    aligned, avalid = L0.align_stack(scene.lr, valid, R.shifts)
    R.lr_norm, R.aligned, R.gain, R.off = L0.match_radiometry(scene.lr, aligned, avalid, R.ref)
    R.avalid = avalid
    if scene.true_shifts is not None:
        R.reg_err = np.abs(R.shifts - scene.true_shifts).max(1)
    else:
        R.reg_err = None
    _tick(cb, "L0", "done", dict(clean=R.clean.round(3).tolist(), ncc=R.ncc.round(3).tolist(), ref=R.ref,
                                 usable=R.usable.tolist()), t0)
    t0 = time.time(); _tick(cb, "L1", "run")
    R.ev = L1.compute_evidence(R.aligned, avalid, R.ref)
    ev = R.ev
    R.E2n = ev["E2"] / max(float(np.median(ev["E2"])), 1e-3)
    R.div = L1.sampling_diversity(ev["compat"], R.shifts)
    _tick(cb, "L1", "done", dict(E1_mean=float(ev["E1"].mean()), E3_frac=float(ev["E3"].mean()),
                                 sigma=ev["sigma"].round(5).tolist()), t0)
    return R


def fidelity(scene, sensor, R, naive=False, exclude=None, n_iter=12):
    """L2+L3a. `naive=True` ignores E3 (temporal averaging of every clean date) for the C2 comparison."""
    compat = R.avalid.copy() if naive else R.ev["compat"].copy()
    if exclude is not None:
        compat[exclude] = False
    q = R.clean * R.usable
    wref = compat.astype(np.float32) * q[:, None, None]
    wn = F.native_weights(compat, q, R.shifts, R.valid.astype(np.float32))
    X, hist = F.fuse(sensor, R.lr_norm, R.aligned, R.shifts, wref, wn, R.ref, n_iter=n_iter)
    return X, hist


def process(scene, prior, calib, sensor=None, target=None, cb=None, gate="evidence", n_regions=300):
    sensor = sensor or SensorModel(scale=scene.scale)
    s = scene.scale
    R = prepare(scene, sensor, target, cb)
    ev = R.ev

    t0 = time.time(); _tick(cb, "L2", "run")
    R.fid, R.hist = fidelity(scene, sensor, R)
    R.bicubic = np.clip(sensor.upsample(scene.lr[R.ref]), 0, 1)
    _tick(cb, "L2", "done", dict(residual=[round(h, 5) for h in R.hist]), t0)

    t0 = time.time(); _tick(cb, "L3", "run")
    R.prior = prior.predict(R.fid) if prior is not None else R.fid.copy()
    R.fid_resid = float(np.sqrt(((sensor.forward(R.fid) - R.lr_norm[R.ref]) ** 2)[:, R.valid[R.ref]].mean()))
    _tick(cb, "L3", "done", dict(fidelity_rms=R.fid_resid), t0)

    t0 = time.time(); _tick(cb, "L4", "run")
    if gate == "ungated":
        R.alpha_lr = np.ones_like(ev["E1"])
    else:
        R.alpha_lr = L.evidence_gate(ev["E1"], R.E2n, ev["E3"], ev["E4"])
    R.lab, R.n = make_regions(R.fid, n_segments=n_regions)
    a_reg = rmean(up(R.alpha_lr, s), R.lab, R.n)                       # region-level blend weight
    a_hr = ndi.gaussian_filter(a_reg[R.lab], 1.0)
    R.alpha_hr = np.clip(a_hr, 0, 1)
    R.sr = np.clip(R.fid + R.alpha_hr[None] * (R.prior - R.fid), 0, 1).astype(np.float32)
    _tick(cb, "L4", "done", dict(alpha_mean=float(R.alpha_hr.mean())), t0)

    t0 = time.time(); _tick(cb, "L5", "run")
    R.rf = L.region_features(R.lab, R.n, s, ev, R.E2n, R.alpha_lr, R.div, R.fid, R.prior, R.sr, ev["sigma"])
    R.prov = L.provenance_head(R.rf)
    R.cls = classify_regions(R.sr, R.lab, R.n)
    shares = [float((R.prov == k).mean()) for k in range(3)]
    _tick(cb, "L5", "done", dict(shares=shares), t0)

    t0 = time.time(); _tick(cb, "L6", "run")
    ref_valid = R.valid[R.ref]
    R.res_r, infl, R.res_lr = L.self_consistency(sensor, R.sr, R.lr_norm[R.ref], ref_valid, ev["sigma"], R.lab, R.n, s)
    R.Fm = featmat(R.rf, R.res_r)
    _tick(cb, "L6", "done", dict(residual_median=float(np.median(R.res_r))), t0)

    t0 = time.time(); _tick(cb, "L7", "run")
    if calib is not None and calib.unc is not None and calib.q:
        R.u0 = calib.u0(R.Fm, infl)
        R.bound = calib.bound(R.u0, R.cls, 0.9)
    else:
        R.u0 = np.full(R.n, np.nan); R.bound = np.full(R.n, np.nan)
    R.infl = infl
    _tick(cb, "L7", "done", dict(bound_mean=float(np.nanmean(R.bound)) if np.isfinite(R.bound).any() else None), t0)

    R.sensor = sensor
    if scene.gt is not None:
        R.mae = np.stack([rmean(np.abs(R.sr[b] - scene.gt[b]), R.lab, R.n) for b in range(4)]).mean(0)
    return R
