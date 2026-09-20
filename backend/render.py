"""Layer rendering to PNG (no external assets, works offline)."""
import io
import numpy as np
from PIL import Image
from matplotlib import colormaps
from skimage.segmentation import find_boundaries
from .l4_l6 import MULTI, SINGLE, PRIOR

PROV_RGB = {MULTI: (27, 158, 140), SINGLE: (232, 169, 29), PRIOR: (196, 35, 88)}


def _png(a):
    im = Image.fromarray(a)
    b = io.BytesIO(); im.save(b, "PNG", optimize=False)
    return b.getvalue()


def rgb(x, bands=(2, 1, 0), lo=0.0, hi=0.26, gamma=0.8):
    im = np.moveaxis(x[list(bands)], 0, -1)
    im = np.clip((im - lo) / (hi - lo), 0, 1) ** gamma
    return _png((im * 255).astype(np.uint8))


def cmap(v, vmin, vmax, name="viridis"):
    t = np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    return _png((colormaps[name](t)[..., :3] * 255).astype(np.uint8))


def provenance(prov_hr):
    out = np.zeros(prov_hr.shape + (3,), np.uint8)
    for k, c in PROV_RGB.items():
        out[prov_hr == k] = c
    return _png(out)


def regions(lab):
    b = find_boundaries(lab, mode="inner")
    a = np.zeros(lab.shape + (4,), np.uint8)
    a[b] = (255, 255, 255, 200)
    return _png(a)


def change_overlay(mask_hr):
    a = np.zeros(mask_hr.shape + (4,), np.uint8)
    a[mask_hr] = (255, 60, 40, 170)
    return _png(a)


def render_layer(name, scene, R, lr_idx=None):
    s = scene.scale
    up = lambda m: np.repeat(np.repeat(m, s, 0), s, 1)
    if name in ("sr", "bicubic", "fid", "prior"):
        return rgb(getattr(R, name))
    if name == "gt" and scene.gt is not None:
        return rgb(scene.gt)
    if name == "sr_fcc":
        return rgb(R.sr, bands=(3, 2, 1), hi=0.45)
    if name == "provenance":
        return provenance(R.prov[R.lab])
    if name == "risk":
        return cmap(R.rf["risk"][R.lab], 0, 0.8, "inferno")
    if name == "uncertainty":
        return cmap(R.bound[R.lab], 0, np.nanpercentile(R.bound, 98), "magma")
    if name == "alpha":
        return cmap(R.alpha_hr, 0, 1, "cividis")
    if name == "E1":
        return cmap(up(R.ev["E1"]), 0, len(R.clean), "viridis")
    if name == "E2":
        return cmap(up(R.E2n), 0, 4, "plasma")
    if name == "E3":
        return cmap(up(R.ev["E3"].astype(float)), 0, 1, "Reds")
    if name == "E4":
        return cmap(up(R.ev["E4"]), 0, 1, "viridis")
    if name == "err" and scene.gt is not None:
        return cmap(np.abs(R.sr - scene.gt).mean(0), 0, 0.06, "magma")
    if name == "regions":
        return regions(R.lab)
    if name == "truth_change" and scene.change_mask is not None:
        return change_overlay(scene.change_mask)
    if name == "lr":
        i = int(lr_idx or 0)
        x = np.repeat(np.repeat(scene.lr[i], s, 1), s, 2)
        img = np.moveaxis(x[[2, 1, 0]], 0, -1)
        img = (np.clip(img / 0.26, 0, 1) ** 0.8 * 255).astype(np.uint8)
        m = np.repeat(np.repeat(scene.cloud[i], s, 0), s, 1)
        img[m] = (0.55 * img[m] + 0.45 * np.array([60, 120, 255])).astype(np.uint8)
        return _png(img)
    raise KeyError(name)
