"""Real-data entry point: build a Scene from uploaded multi-date GeoTIFFs.

Expected: one GeoTIFF per acquisition, first four bands = B02,B03,B04,B08 (10 m), same grid.
Reflectance either 0-1 or scaled x10000 (auto-detected). Date parsed from filename (YYYYMMDD / YYYY-MM-DD).
Cloud mask: a simple brightness/whiteness heuristic. In production replace with s2cloudless
(`from s2cloudless import S2PixelCloudDetector`) -- only the boolean (N,h,w) mask is consumed.
"""
import io, re
import numpy as np
import tifffile
from scipy import ndimage as ndi
from .synth import Scene


def heuristic_cloud_mask(x):
    b = x.mean(0)
    spread = (x.max(0) - x.min(0)) / (b + 1e-6)
    m = (b > 0.22) & (spread < 0.45) & (x[3] > 0.12)
    m = ndi.binary_opening(m, iterations=1)
    return ndi.binary_dilation(m, iterations=2)


def _geo(tf):
    p = tf.pages[0]
    scale = p.tags.get(33550); tie = p.tags.get(33922); keys = p.tags.get(34735)
    gsd = float(scale.value[0]) if scale else 10.0
    origin = (float(tie.value[3]), float(tie.value[4])) if tie else (0.0, 0.0)
    epsg = 32644
    if keys:
        v = list(keys.value)
        for i in range(4, len(v) - 3, 4):
            if v[i] == 3072:
                epsg = int(v[i + 3])
    return gsd, origin, epsg


def scene_from_geotiffs(files, scale=4):
    """files: list of (filename, bytes)"""
    items = []
    for name, data in files:
        with tifffile.TiffFile(io.BytesIO(data)) as tf:
            arr = tf.asarray(); gsd, origin, epsg = _geo(tf)
        if arr.ndim == 2:
            raise ValueError(f"{name}: need >= 4 bands")
        if arr.shape[0] > 16:
            arr = np.moveaxis(arr, -1, 0)
        if arr.shape[0] < 4:
            raise ValueError(f"{name}: need >= 4 bands (B02,B03,B04,B08), got {arr.shape[0]}")
        arr = arr[:4].astype(np.float32)
        if arr.max() > 2.0:
            arr = arr / 10000.0
        m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", name)
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else name
        items.append((date, arr, gsd, origin, epsg))
    items.sort(key=lambda t: t[0])
    shapes = {a.shape for _, a, *_ in items}
    if len(shapes) != 1:
        raise ValueError("all dates must share the same grid size")
    if len(items) < 3:
        raise ValueError("need at least 3 acquisitions")
    lr = np.stack([a for _, a, *_ in items]).clip(0, 1)
    h, w = lr.shape[2:]
    hh, ww = (h // 4) * 4, (w // 4) * 4
    lr = lr[:, :, :hh, :ww]
    cloud = np.stack([heuristic_cloud_mask(x) for x in lr])
    _, _, gsd, origin, epsg = items[0]
    return Scene(lr=lr, cloud=cloud, dates=[d for d, *_ in items], scale=scale, gsd_lr=gsd, origin=origin, epsg=epsg,
                 name="uploaded-stack")
