"""Region (super-pixel) helpers. Provenance, risk and calibration all operate at REGION level:
neighbouring pixels are spatially correlated, so regions are far closer to exchangeable units
than pixels are (the honest assumption behind split-conformal, stated in the architecture)."""
import numpy as np
from skimage.segmentation import slic


def make_regions(img_hr, n_segments=300, compactness=14):
    """img_hr (B,H,W) -> label map (H,W) int32, count."""
    im = np.moveaxis(img_hr[[2, 1, 0]] if img_hr.shape[0] >= 3 else img_hr[:1].repeat(3, 0), 0, -1)
    lo, hi = np.percentile(im, 1), np.percentile(im, 99)
    im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
    lab = slic(im, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1, sigma=0.5)
    _, lab = np.unique(lab, return_inverse=True)
    lab = lab.reshape(im.shape[:2]).astype(np.int32)
    return lab, int(lab.max() + 1)


def rmean(x, lab, n):
    """Mean of x (H,W) per region."""
    return np.bincount(lab.ravel(), weights=x.ravel().astype(np.float64), minlength=n) / np.maximum(
        np.bincount(lab.ravel(), minlength=n), 1)


def up(x, s):
    return np.repeat(np.repeat(x, s, axis=-2), s, axis=-1)


def paint(vals, lab):
    return vals[lab]
