"""Train + calibrate + verify. Produces artifacts/{prior.pkl, calib.pkl, calib_report.json}.

Disjoint scene splits (seeds), so nothing is verified on data it was fit on:
  prior       : 101-106   (fidelity output -> HR truth)
  uncertainty : 201-206   (region-error regressor)
  conformal   : 301-308   (split-conformal quantiles)
  test        : 401-408   (coverage measurement -> reliability diagram)
"""
import os, json, time
import numpy as np
from .synth import make_scene
from .sensor import SensorModel
from . import pipeline as P
from .prior import PriorModel
from .l7_calibration import Calibrator, CLASSES

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
SPLITS = dict(prior=range(101, 107), unc=range(201, 207), cal=range(301, 309), test=range(401, 409))


def _scene(seed):
    return make_scene(seed)


def _region_data(seeds, prior, log, tag):
    F, mae, cls, infl = [], [], [], []
    for sd in seeds:
        sc = _scene(sd)
        R = P.process(sc, prior, None)
        F.append(R.Fm); mae.append(R.mae); cls.append(R.cls); infl.append(R.infl)
        log(f"  [{tag}] seed {sd}: {R.n} regions")
    return np.concatenate(F), np.concatenate(mae), np.concatenate(cls), np.concatenate(infl)


def train_all(log=print, force=False):
    os.makedirs(ART, exist_ok=True)
    pp, cp, rp = (os.path.join(ART, f) for f in ("prior.pkl", "calib.pkl", "calib_report.json"))
    t0 = time.time()
    sensor = SensorModel()
    log("[1/4] training prior path (fidelity output -> HR truth)")
    pairs = []
    for sd in SPLITS["prior"]:
        sc = _scene(sd)
        R = P.prepare(sc, sensor)
        fid, _ = P.fidelity(sc, sensor, R)
        pairs.append((fid, sc.gt)); log(f"  seed {sd}")
    prior = PriorModel().fit(pairs); prior.save(pp)
    log("[2/4] uncertainty model")
    Fu, mu, cu, iu = _region_data(SPLITS["unc"], prior, log, "unc")
    cal = Calibrator(); cal.fit_uncertainty(Fu, mu)
    log("[3/4] split-conformal calibration")
    Fc, mc, cc, ic = _region_data(SPLITS["cal"], prior, log, "cal")
    cal.fit_conformal(cal.u0(Fc, ic), mc, cc)
    log("[4/4] measuring coverage on held-out test scenes")
    Ft, mt, ct, it = _region_data(SPLITS["test"], prior, log, "test")
    rep = cal.verify(cal.u0(Ft, it), mt, ct)
    rep["class_names"] = CLASSES; rep["n_cal_regions"] = int(len(mc))
    cal.save(cp)
    json.dump(rep, open(rp, "w"), indent=1)
    log(f"done in {time.time()-t0:.0f}s. coverage@90: {rep['overall'][5]:.3f} (nominal 0.90), global-only: {rep['global_only'][5]:.3f}")
    return prior, cal, rep


def load_or_train(log=print):
    pp, cp, rp = (os.path.join(ART, f) for f in ("prior.pkl", "calib.pkl", "calib_report.json"))
    try:
        if all(os.path.exists(p) for p in (pp, cp, rp)):
            return PriorModel.load(pp), Calibrator.load(cp), json.load(open(rp))
    except Exception as e:
        log(f"artifact load failed ({e}); retraining")
    return train_all(log)


if __name__ == "__main__":
    train_all()
