"""Synthetic Sentinel-2-like scene generator.

There is no network in this build environment and no Indian HR reference data, so the
prototype ships a generator that produces a *known* 2.5 m ground truth and a realistic
multi-date 10 m stack from it: sub-pixel shifts, PSF blur, per-date atmosphere,
noise, clouds + shadows (some undetected), thin haze, phenology drift, and *real change*
(new buildings + a flooded field) partway through the series.

Smallholder-scale parcels (~15-40 px = 40-100 m), narrow roads (2-3 px = 5-7 m) and
ponds are included deliberately: they are what makes 10 m -> 2.5 m hard.
Real data enters through `scene_from_geotiffs()` in pipeline_io.py.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .sensor import SensorModel

BANDS = ["B02", "B03", "B04", "B08"]
WATER, VEG, URBAN, BARE = 0, 1, 2, 3
CLASS_NAMES = {WATER: "water", VEG: "vegetation", URBAN: "built-up", BARE: "bare / other"}

SPEC = {
    "veg": (0.035, 0.075, 0.045, 0.42),
    "soil": (0.10, 0.14, 0.18, 0.26),
    "dry": (0.07, 0.11, 0.14, 0.30),
    "roof_a": (0.20, 0.21, 0.22, 0.24),
    "roof_b": (0.30, 0.31, 0.32, 0.33),
    "roof_c": (0.13, 0.14, 0.15, 0.18),
    "road": (0.10, 0.105, 0.11, 0.12),
    "water": (0.060, 0.055, 0.035, 0.012),
}


def sp(name):
    return np.array(SPEC[name], dtype=np.float32)[:, None, None]


@dataclass
class Scene:
    lr: np.ndarray                 # (N,B,h,w) reflectance, native frame of each date
    cloud: np.ndarray              # (N,h,w) bool, DETECTED contamination (what a cloud mask would flag)
    dates: list
    scale: int = 4
    gsd_lr: float = 10.0
    origin: tuple = (500000.0, 1400000.0)   # UTM easting/northing of top-left corner (m)
    epsg: int = 32644
    band_names: list = field(default_factory=lambda: list(BANDS))
    # ---- synthetic-only ground truth (None for real data)
    gt: np.ndarray = None          # (B,H,W) HR reflectance at target date
    gt_class: np.ndarray = None    # (H,W)
    parcel: np.ndarray = None      # (H,W) parcel ids (-1 outside farmland)
    change_mask: np.ndarray = None # (H,W) bool: pixels whose state differs between early and target dates
    true_shifts: np.ndarray = None # (N,2)
    seed: int = None
    name: str = "scene"


def _fractal(rng, H, W, sigma):
    n = ndi.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma, mode="wrap")
    return (n - n.mean()) / (n.std() + 1e-9)


def _parcels(rng, mask, n):
    ys, xs = np.nonzero(mask)
    lo = np.array([ys.min(), xs.min()], float)
    hi = np.array([ys.max(), xs.max()], float)
    pts = rng.uniform(lo, hi, (n, 2))
    th = rng.uniform(0, np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    st = np.array([1.0, rng.uniform(1.0, 2.0)])
    tf = lambda p: (p @ R) * st
    _, idx = cKDTree(tf(pts)).query(tf(np.c_[ys, xs].astype(float)))
    return ys, xs, idx


def _build_state_a(rng, H, W):
    yy, xx = np.mgrid[:H, :W]
    n_macro = 9
    pts = rng.uniform(0, H, (n_macro, 2))
    _, macro = cKDTree(pts).query(np.c_[yy.ravel(), xx.ravel()])
    macro = macro.reshape(H, W)
    kinds = [WATER, URBAN, BARE] + [VEG] * 6
    rng.shuffle(kinds)
    cls = np.zeros((H, W), np.int8)
    for m, k in enumerate(kinds):
        cls[macro == m] = k

    img = np.zeros((4, H, W), np.float32)
    parcel = -np.ones((H, W), np.int32)
    pid = 0
    micro = _fractal(rng, H, W, 1.2)

    for m, k in enumerate(kinds):
        mask = macro == m
        if k == VEG:
            ys, xs, idx = _parcels(rng, mask, n=int(rng.integers(14, 24)))
            for p in range(idx.max() + 1):
                sel = idx == p
                if sel.sum() == 0:
                    continue
                py, px = ys[sel], xs[sel]
                w = rng.uniform(0.0, 1.0)
                base = (w * sp("veg") + (1 - w) * (sp("soil") if rng.random() < .5 else sp("dry")))[:, 0, 0]
                phi = rng.uniform(0, np.pi)
                per = rng.uniform(3.0, 6.0)
                amp = rng.uniform(0.0, 0.22) if rng.random() < 0.7 else 0.0
                stripe = 1 + amp * np.sin(2 * np.pi * (px * np.cos(phi) + py * np.sin(phi)) / per)
                img[:, py, px] = base[:, None] * stripe[None, :]
                parcel[py, px] = pid
                pid += 1
        elif k == URBAN:
            img[:, mask] = sp("soil")[:, :, 0] * 0.9
            py0 = int(rng.integers(20, 30)); px0 = int(rng.integers(20, 30))
            oy, ox = int(rng.integers(0, py0)), int(rng.integers(0, px0))
            road = np.zeros((H, W), bool)
            for y in range(oy, H, py0):
                road[y:y + int(rng.integers(2, 4)), :] = True
            for x in range(ox, W, px0):
                road[:, x:x + int(rng.integers(2, 4))] = True
            u = mask & road
            img[:, u] = sp("road")[:, :, 0]
            lab, nb = ndi.label(mask & ~road)
            for b in range(1, nb + 1):
                ys, xs = np.nonzero(lab == b)
                if len(ys) < 60:
                    continue
                for _ in range(int(rng.integers(1, 4))):
                    hh, ww = int(rng.integers(5, 12)), int(rng.integers(5, 14))
                    cy = int(rng.choice(ys)); cx = int(rng.choice(xs))
                    y0, x0 = max(cy - hh // 2, 0), max(cx - ww // 2, 0)
                    roof = np.array(SPEC[str(rng.choice(["roof_a", "roof_b", "roof_c"]))], np.float32)
                    box = np.zeros((H, W), bool)
                    box[y0:y0 + hh, x0:x0 + ww] = True
                    region = box & mask & ~road
                    img[:, region] = roof[:, None]
        elif k == WATER:
            img[:, mask] = sp("water")[:, :, 0]
        else:  # BARE / scrub
            v = _fractal(rng, H, W, 3.0)
            w = np.clip(0.5 + 0.5 * v, 0, 1) * 0.6
            base = (1 - w)[None] * sp("soil") + w[None] * sp("veg")
            for b in range(4):
                img[b][mask] = base[b][mask]

    # small ponds inside farmland
    for _ in range(3):
        cy, cx = rng.integers(20, H - 20, 2)
        if cls[cy, cx] == VEG:
            r = rng.uniform(4, 8)
            pond = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
            img[:, pond] = sp("water")[:, :, 0]
            cls[pond] = WATER
            parcel[pond] = -1

    # field bunds: lighten parcel boundaries slightly (subtle, narrow)
    b = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = parcel[: H - dy, : W - dx]; c = parcel[dy:, dx:]
        edge = (a != c) & (a >= 0) & (c >= 0)
        b[: H - dy, : W - dx] |= edge
    img[:, b] = 0.6 * img[:, b] + 0.4 * sp("soil")[:, :, 0]

    img *= (1 + 0.05 * micro)[None]
    return np.clip(img, 0.003, 0.9), cls, parcel


def _apply_change(rng, img, cls, parcel):
    """Structured change: new buildings on a scrub/farm pad + a flooded field."""
    H, W = cls.shape
    B = img.copy(); C = cls.copy()
    change = np.zeros((H, W), bool)
    yy, xx = np.mgrid[:H, :W]
    # 1) new building cluster
    y0 = x0 = 10
    for _ in range(60):
        y0, x0 = int(rng.integers(8, H - 50)), int(rng.integers(8, W - 50))
        pad = (slice(y0, y0 + 34), slice(x0, x0 + 40))
        if np.isin(cls[pad], (VEG, BARE)).mean() > 0.9:
            break
    pad = (slice(y0, y0 + 34), slice(x0, x0 + 40))
    B[:, pad[0], pad[1]] = sp("road")
    for (dy, dx, hh, ww, roof) in ((3, 3, 12, 14, "roof_b"), (3, 22, 10, 13, "roof_a"), (19, 5, 11, 12, "roof_c"), (19, 22, 10, 14, "roof_b")):
        B[:, y0 + dy:y0 + dy + hh, x0 + dx:x0 + dx + ww] = sp(roof)
    C[pad] = URBAN
    change[pad] = True
    # 2) flooded field: elliptical blob over farmland
    blob = np.zeros((H, W), bool)
    for _ in range(80):
        cy, cx = int(rng.integers(30, H - 30)), int(rng.integers(30, W - 30))
        th = rng.uniform(0, np.pi)
        u = (yy - cy) * np.cos(th) + (xx - cx) * np.sin(th)
        v = -(yy - cy) * np.sin(th) + (xx - cx) * np.cos(th)
        blob = (u / 26.0) ** 2 + (v / 15.0) ** 2 < 1
        if (cls[blob] == VEG).mean() > 0.8 and not (change & blob).any():
            break
    if (change & blob).any():
        blob[:] = False
    B[:, blob] = sp("water")[:, :, 0] * (1 + 0.15 * rng.standard_normal(int(blob.sum())))[None]
    C[blob] = WATER
    change |= blob
    return np.clip(B, 0.003, 0.9), C, change


def make_scene(seed=7, size=256, n_dates=8, cloud_level=1.0, noise=0.004, with_change=True, scale=4):
    rng = np.random.default_rng(seed)
    H = W = size
    imgA, clsA, parcel = _build_state_a(rng, H, W)
    if with_change:
        imgB, clsB, change = _apply_change(rng, imgA, clsA, parcel)
    else:
        imgB, clsB, change = imgA, clsA, np.zeros((H, W), bool)
    k_change = n_dates // 2
    sensor = SensorModel(scale=scale)
    h = H // scale

    veg_mask = clsA == VEG
    lr = np.zeros((n_dates, 4, h, h), np.float32)
    cloud_det = np.zeros((n_dates, h, h), bool)
    shifts = np.zeros((n_dates, 2))
    dates = []
    cl_frac = np.clip(rng.uniform(0.0, 0.22, n_dates) * cloud_level, 0, 0.7)
    cl_frac[-1] = min(cl_frac[-1], 0.02)
    if n_dates > 2:
        cl_frac[1] = max(cl_frac[1], 0.22 * cloud_level)
    haze_date = n_dates - 3
    for d in range(n_dates):
        state = imgB if (with_change and d >= k_change) else imgA
        st = state.copy()
        phi = 0.05 * (d - (n_dates - 1)) / n_dates          # gentle phenology drift
        st[3, veg_mask] *= (1 + phi)
        sh = np.zeros(2) if d == n_dates - 1 else rng.uniform(-0.9, 0.9, 2)
        shifts[d] = sh
        obs = sensor.forward(st, tuple(sh))
        if d != n_dates - 1:
            g = 1 + 0.025 * rng.standard_normal(4); o = 0.003 * rng.standard_normal(4)
            obs = obs * g[:, None, None] + o[:, None, None]
        obs = obs + noise * rng.standard_normal(obs.shape).astype(np.float32)
        if cl_frac[d] > 0.003:
            f = _fractal(rng, h, h, 4.0)
            cloud = f > np.quantile(f, 1 - cl_frac[d])
            shadow = ndi.shift(cloud.astype(float), (3, 4), order=0) > 0.5
            shadow &= ~cloud
            ctone = 0.55 + 0.12 * _fractal(rng, h, h, 1.5)
            for b in range(4):
                obs[b] = np.where(cloud, ctone + 0.02 * rng.standard_normal((h, h)), obs[b])
                obs[b] = np.where(shadow, obs[b] * 0.55, obs[b])
            det = ndi.binary_dilation(cloud, iterations=2) | (shadow & (rng.random((h, h)) > 0.35))
            cloud_det[d] = det
        if d == haze_date:  # undetected thin haze patch
            cy, cx = rng.integers(10, h - 10, 2)
            yy, xx = np.mgrid[:h, :h]
            hz = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 6.0 ** 2)))
            obs += 0.05 * hz[None]
        lr[d] = np.clip(obs, 0, 1)
        dates.append(f"2026-{2 + d // 2:02d}-{(d % 2) * 14 + 3:02d}")
    gt = np.clip(imgB if with_change else imgA, 0, 1).copy()
    gtc = clsB if with_change else clsA
    return Scene(lr=lr, cloud=cloud_det, dates=dates, scale=scale, gt=gt.astype(np.float32),
                 gt_class=gtc, parcel=parcel, change_mask=change, true_shifts=shifts,
                 seed=seed, name=f"synthetic-{seed}")
