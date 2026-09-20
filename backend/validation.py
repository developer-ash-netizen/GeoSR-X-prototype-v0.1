"""L9 VALIDATION -- three independent tiers, no single point of failure.

  Tier A  absolute quality vs real HR pairs   (here: synthetic HR truth; opensr-test/Sen2Venus/WorldStrat on real data)
  Tier B  leave-one-date-out                  (no external data; consistency, NOT absolute resolution gain)
  Tier C  downstream delta                    (field-boundary F1/IoU on LR vs SR)
plus a change-preservation test for C2.

Hallucination / omission / improvement here are edge-based PROXIES in the spirit of ESA opensr-test,
not that library's exact metrics; swap `tier_a_metrics` for opensr-test when real HR pairs exist.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
from . import pipeline as P


# ---------------------------------------------------------------- Tier A
def _edges(x, thr=0.02):
    lum = ndi.gaussian_filter(x.mean(0), 0.7)
    gy, gx = np.gradient(lum)
    return np.hypot(gx, gy) > thr


def tier_a_metrics(gt, x, bic):
    rng = float(gt.max() - gt.min())
    mse = float(((gt - x) ** 2).mean())
    psnr = 10 * np.log10(rng ** 2 / max(mse, 1e-12))
    ss = float(np.mean([ssim(gt[b], x[b], data_range=rng) for b in range(gt.shape[0])]))
    a = gt.reshape(4, -1); b = x.reshape(4, -1)
    cosang = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)
    sam = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean())
    eg, ex = _edges(gt), _edges(x)
    dg = ndi.binary_dilation(eg, iterations=1); dx = ndi.binary_dilation(ex, iterations=1)
    omission = float(1 - (eg & dx).sum() / max(eg.sum(), 1))          # real edges missing from SR
    halluc = float(1 - (ex & dg).sum() / max(ex.sum(), 1))            # SR edges that are not real
    mse_b = float(((gt - bic) ** 2).mean()); psnr_b = 10 * np.log10(rng ** 2 / max(mse_b, 1e-12))
    return dict(psnr=round(psnr, 2), ssim=round(ss, 4), sam_deg=round(sam, 3), omission=round(omission, 3),
                hallucination=round(halluc, 3), improvement_db=round(psnr - psnr_b, 2))


# ---------------------------------------------------------------- Tier B
def tier_b(scene, R, prior):
    sensor = R.sensor
    N = len(R.clean)
    cands = [d for d in range(N) if d != R.ref and R.usable[d] and R.clean[d] >= 0.7]
    before = [d for d in cands if d < R.ref]
    if not cands:
        return None
    h = before[-1] if before else cands[-1]
    fid_wo, _ = P.fidelity(scene, sensor, R, exclude=h)
    pri = prior.predict(fid_wo) if prior is not None else fid_wo
    sr_wo = np.clip(fid_wo + R.alpha_hr[None] * (pri - fid_wo), 0, 1)
    sh = tuple(R.shifts[h])
    obs = R.lr_norm[h]
    q = R.clean * R.usable
    wn = ndi.shift(R.ev["compat"][h].astype(np.float32), (-sh[0], -sh[1]), order=1, mode="nearest")
    m = (wn > 0.99) & R.valid[h]
    m = ndi.binary_erosion(m, iterations=2)
    if m.sum() < 200:
        return None
    preds = {
        "GeoSR-X (gated), date hidden": sensor.forward(sr_wo, sh),
        "fidelity path only, date hidden": sensor.forward(fid_wo, sh),
        "single-date bicubic, degraded": sensor.forward(np.clip(sensor.upsample(scene.lr[R.ref]), 0, 1), sh),
        "single-date copy (registered)": ndi.shift(R.lr_norm[R.ref], (0, -sh[0], -sh[1]), order=3, mode="nearest"),
    }
    sig = R.ev["sigma"]
    rows = []
    for k, p in preds.items():
        e = (p - obs)[:, m]
        rows.append(dict(method=k, rmse=round(float(np.sqrt((e ** 2).mean())), 5),
                         rmse_sigma=round(float(np.sqrt(((e / sig[:, None]) ** 2).mean())), 2)))
    return dict(hidden_date=int(h), n_pixels=int(m.sum()), rows=rows,
                caveat="Validates temporal/radiometric consistency against a real observation the model never saw. "
                       "It does NOT measure absolute resolution gain against a true 2.5 m reference (that is Tier A).")


# ---------------------------------------------------------------- Tier C
def _field_edges(x):
    ndvi = (x[3] - x[2]) / (x[3] + x[2] + 1e-6)
    return canny(ndvi, sigma=1.6, low_threshold=0.80, high_threshold=0.93, use_quantiles=True)


def _gt_boundaries(parcel):
    H, W = parcel.shape
    b = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = parcel[: H - dy, : W - dx]; c = parcel[dy:, dx:]
        b[: H - dy, : W - dx] |= (a != c) & (a >= 0) & (c >= 0)
    return b


def tier_c(scene, products):
    gtb = _gt_boundaries(scene.parcel)
    zone = ndi.binary_dilation(scene.parcel >= 0, iterations=2)
    out = []
    for name, x in products.items():
        pe = _field_edges(x) & zone
        tol_p = ndi.binary_dilation(gtb, iterations=2); tol_g = ndi.binary_dilation(pe, iterations=2)
        prec = float((pe & tol_p).sum() / max(pe.sum(), 1)); rec = float((gtb & tol_g).sum() / max(gtb.sum(), 1))
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        a = ndi.binary_dilation(pe, iterations=1); b = ndi.binary_dilation(gtb, iterations=1)
        iou = float((a & b).sum() / max((a | b).sum(), 1))
        out.append(dict(product=name, precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3), iou=round(iou, 3)))
    return out


# ---------------------------------------------------------------- C2: change preservation
def change_test(scene, R, naive_fid):
    gt_change = scene.change_mask
    if gt_change is None or not gt_change.any():
        return None
    s = scene.scale
    flag = np.repeat(np.repeat(R.ev["E3"], s, 0), s, 1)
    core = ndi.binary_erosion(gt_change, iterations=3)
    rec = float(flag[core].mean()) if core.any() else float("nan")
    fp = float(flag[~ndi.binary_dilation(gt_change, iterations=6)].mean())

    def rmse(x):
        return float(np.sqrt(((x - scene.gt)[:, core] ** 2).mean()))
    rows = [
        dict(method="single-date bicubic", rmse=round(rmse(R.bicubic), 4)),
        dict(method="naive temporal fusion (no E3)", rmse=round(rmse(naive_fid), 4)),
        dict(method="GeoSR-X change-aware (E3 protects change)", rmse=round(rmse(R.sr), 4)),
    ]
    return dict(change_recall=round(rec, 3), false_flag_rate=round(fp, 4), rows=rows, n_core_px=int(core.sum()))


def run_validation(scene, R, prior, cb=None):
    V = {}
    if scene.gt is not None:
        naive, _ = P.fidelity(scene, R.sensor, R, naive=True)
        R.naive = naive
        variants = {
            "single-date bicubic": R.bicubic,
            "naive temporal fusion": naive,
            "fidelity path only": R.fid,
            "prior path only (ungated)": R.prior,
            "GeoSR-X (evidence-gated)": R.sr,
        }
        V["tier_a"] = [dict(product=k, **tier_a_metrics(scene.gt, v, R.bicubic)) for k, v in variants.items()]
        V["tier_c"] = tier_c(scene, {k: variants[k] for k in ("single-date bicubic", "fidelity path only", "prior path only (ungated)", "GeoSR-X (evidence-gated)")})
        V["change"] = change_test(scene, R, naive)
        cov = float((R.mae <= R.bound).mean()) if np.isfinite(R.bound).any() else None
        V["scene_coverage90"] = cov
        cls_names = ["water", "vegetation", "built-up", "bare / other"]
        V["scene_coverage_by_class"] = {cls_names[c]: (float((R.mae[R.cls == c] <= R.bound[R.cls == c]).mean()), int((R.cls == c).sum()))
                                        for c in range(4) if (R.cls == c).any()}
        V["prov_error"] = {n: (float(R.mae[R.prov == k].mean()) if (R.prov == k).any() else None)
                           for k, n in enumerate(["multi-date observed", "single-date observed", "prior-generated"])}
    V["tier_b"] = tier_b(scene, R, prior)
    return V
