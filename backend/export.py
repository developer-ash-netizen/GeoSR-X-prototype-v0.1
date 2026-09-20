"""L8 ANALYSIS-READY OUTPUT: georeferenced GeoTIFF + sidecar (error budget, coverage report, lineage)."""
import json, os, time
import numpy as np
import tifffile
from .synth import CLASS_NAMES
from .l4_l6 import PROV_NAMES
from .l7_calibration import CLASSES

VERSION = "geosrx-proto-0.4"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_geotiff(path, scene, R):
    s = scene.scale
    gsd = scene.gsd_lr / s
    H, W = R.sr.shape[1:]
    x0, y0 = scene.origin
    prov_hr = R.prov[R.lab].astype(np.float32)
    unc_hr = R.bound[R.lab].astype(np.float32)
    data = np.concatenate([R.sr.astype(np.float32), prov_hr[None], unc_hr[None]], 0)
    names = [f"SR_{b}" for b in scene.band_names] + ["PROVENANCE_CLASS", "UNCERTAINTY_P90_MAE"]
    keys = [1, 1, 0, 4,  1024, 0, 1, 1,  1025, 0, 1, 1,  3072, 0, 1, int(scene.epsg),  3076, 0, 1, 9001]
    items = "".join(f'<Item name="DESCRIPTION" sample="{i}" role="description">{n}</Item>' for i, n in enumerate(names))
    items += ('<Item name="PROVENANCE_CLASSES">0=multi-date observed;1=single-date observed;2=prior-generated</Item>'
              '<Item name="UNCERTAINTY_MEANING">calibrated 90% upper bound on region-mean absolute reflectance error</Item>'
              f'<Item name="GENERATOR">{VERSION}</Item>')
    xml = f"<GDALMetadata>{items}</GDALMetadata>"
    extratags = [
        (33550, "d", 3, (gsd, gsd, 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, float(x0), float(y0), 0.0), True),
        (34735, "H", len(keys), tuple(keys), True),
        (42112, "s", 0, xml, True),
    ]
    tifffile.imwrite(path, data, photometric="minisblack", planarconfig="separate", extratags=extratags,
                     metadata=None, description=json.dumps({"bands": names, "generator": VERSION}))
    return names


def sidecar(scene, R, V, calib_report, opts):
    n = R.n
    ebud = {}
    for c, cname in enumerate(CLASSES):
        sel = R.cls == c
        if sel.any():
            ebud[cname] = dict(regions=int(sel.sum()), mean_p90_bound=float(np.nanmean(R.bound[sel])),
                               mean_u0=float(np.nanmean(R.u0[sel])))
    prov = {PROV_NAMES[k]: dict(area_fraction=float(((R.prov[R.lab]) == k).mean()),
                                mean_p90_bound=float(np.nanmean(R.bound[R.prov == k])) if (R.prov == k).any() else None)
            for k in range(3)}
    d = dict(
        generator=VERSION, created=time.strftime("%Y-%m-%dT%H:%M:%S"), scene=scene.name,
        crs=f"EPSG:{scene.epsg}", pixel_size_m=scene.gsd_lr / scene.scale, scale_factor=scene.scale,
        bands=[f"SR_{b}" for b in scene.band_names] + ["PROVENANCE_CLASS", "UNCERTAINTY_P90_MAE"],
        provenance_classes=PROV_NAMES,
        error_budget=dict(
            registration=dict(method="masked xcorr + Lucas-Kanade", shifts_px=R.shifts, ncc=R.ncc,
                              max_true_error_px=(float(np.max(R.reg_err)) if R.reg_err is not None else None)),
            radiometric=dict(gain=R.gain, offset=R.off, noise_sigma=R.ev["sigma"]),
            fidelity=dict(rms_residual_target_date=R.fid_resid),
            uncertainty_by_class=ebud, by_provenance=prov),
        coverage_report=dict(nominal=0.9, held_out_test=calib_report,
                             note="Region-level split-conformal; coverage measured on scenes disjoint from calibration."),
        lineage=dict(options=opts, dates=scene.dates, target_date=scene.dates[R.ref], target_index=int(R.ref),
                     dates_used=[bool(u) for u in R.usable], clean_fraction=R.clean,
                     stages=["L0 cloud/quality/co-registration", "L1 evidence tensor", "L2 fusion", "L3 fidelity+prior",
                             "L4 evidence gate", "L5 provenance+risk", "L6 self-consistency", "L7 conformal", "L8 export"]),
        limitations=[
            "Prototype validated on synthetic data with known 2.5 m truth; no Indian HR reference performance is claimed.",
            "Prior path is a gradient-boosted stand-in for a Swin/RRDB or SEN2SR network.",
            "Conformal guarantees assume exchangeable regions; spatial correlation makes this approximate.",
            "Hallucination is measured, located and labelled, not eliminated."],
    )
    return _jsonable(d)
