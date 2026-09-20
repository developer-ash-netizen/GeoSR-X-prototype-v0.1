"""L7 CALIBRATION -- split-conformal, REGION level, per land-cover class.

  1. an uncertainty model predicts region-mean absolute error from evidence/audit features (trained on split A)
  2. split-conformal on split B (held out from both SR training and split A) turns that prediction
     into a bound with finite-sample coverage under exchangeability of *regions*:
         bound_i = q_c(level) * u0_i ,   q_c = conformal quantile of (MAE_i / u0_i) within class c
  3. coverage is then MEASURED on split C and published as a reliability diagram.

Exchangeability caveat (stated, not hidden): regions are far more independent than pixels but not
perfectly so; classes with too few calibration regions fall back to the pooled quantile.
"""
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor

LEVELS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
CLASSES = ["water", "vegetation", "built-up", "bare / other"]


def classify_regions(sr, lab, n):
    from .regions import rmean
    m = np.stack([rmean(sr[b], lab, n) for b in range(4)])
    blue, green, red, nir = m
    ndvi = (nir - red) / (nir + red + 1e-6); ndwi = (green - nir) / (green + nir + 1e-6)
    c = np.full(n, 3, np.int8)                       # bare / other
    c[(ndvi < 0.12) & (ndvi > -0.3)] = 2             # built-up (flat spectra)
    c[ndvi >= 0.3] = 1                               # vegetation
    c[ndwi > 0.3] = 0                                # water
    return c


def featmat(rf, res):
    return np.c_[rf["E1"], rf["E2"], rf["E3f"], rf["E4"], rf["r_t"], rf["r_s"], rf["r_x"], res,
                 rf["tex"], np.log(rf["size"])].astype(np.float32)


def _cq(scores, level):
    s = np.sort(scores); n = len(s)
    k = int(np.ceil((n + 1) * level))
    return float(s[min(k, n) - 1]) if k <= n else float(s[-1] * 1.25)


class Calibrator:
    def __init__(self):
        self.unc = None; self.q = {}; self.qpool = {}; self.n_cal = {}; self.report = None

    def fit_uncertainty(self, F, mae):
        self.unc = HistGradientBoostingRegressor(max_iter=120, max_depth=4, learning_rate=0.06,
                                                 l2_regularization=2.0, min_samples_leaf=15, random_state=0)
        self.unc.fit(F, np.log(mae + 1e-4))

    def u0(self, F, infl=1.0):
        return np.exp(self.unc.predict(F)) * infl

    def fit_conformal(self, u0, mae, cls):
        sc = mae / u0
        self.qpool = {l: _cq(sc, l) for l in LEVELS}
        for c in range(4):
            sel = cls == c
            self.n_cal[c] = int(sel.sum())
            self.q[c] = {l: (_cq(sc[sel], l) if sel.sum() >= 40 else self.qpool[l]) for l in LEVELS}

    def bound(self, u0, cls, level=0.9):
        q = np.array([self.q[int(c)][level] for c in cls])
        return u0 * q

    def verify(self, u0, mae, cls):
        """Empirical coverage on held-out regions: overall and per class."""
        out = {"levels": LEVELS, "overall": [], "per_class": {}, "n_test": int(len(mae))}
        for l in LEVELS:
            out["overall"].append(float((mae <= self.bound(u0, cls, l)).mean()))
        for c in range(4):
            sel = cls == c
            out["per_class"][CLASSES[c]] = {
                "n_cal": self.n_cal.get(c, 0), "n_test": int(sel.sum()),
                "coverage": [float((mae[sel] <= self.bound(u0[sel], cls[sel], l)).mean()) if sel.sum() else None for l in LEVELS],
                "q90": self.q[c][0.9]}
        # global (non-class-adaptive) baseline for contrast
        qp = np.array([self.qpool[l] for l in LEVELS])
        out["global_only"] = [float((mae <= u0 * self.qpool[l]).mean()) for l in LEVELS]
        self.report = out
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
