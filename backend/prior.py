"""L3b PRIOR PATH -- learned detail synthesis from real high-resolution pairs.

Stand-in for the Swin/RRDB (or SEN2SR) network so the prototype trains in seconds on a laptop:
a gradient-boosted regressor maps a 5x5 neighbourhood of the fidelity-path image to the
missing high-frequency residual, trained on (fidelity output, HR truth) pairs.

Interface for swapping in a real network: `predict(fid: (B,H,W)) -> (B,H,W)` returning the
prior-path image. Everything downstream (gate, provenance, risk, calibration) treats it as
a black box, which is the point of P2.
"""
import os, pickle
import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingRegressor


def _features(x):
    """x: (B,H,W) -> (B*H*W, F)"""
    B, H, W = x.shape
    lm = ndi.gaussian_filter(x, (0, 2.0, 2.0), mode="nearest")
    pad = np.pad(x, ((0, 0), (2, 2), (2, 2)), mode="reflect")
    patches = []
    for dy in range(5):
        for dx in range(5):
            patches.append(pad[:, dy:dy + H, dx:dx + W] - lm)
    P = np.stack(patches, -1)                                    # (B,H,W,25)
    others = []
    for b in range(B):
        others.append(np.broadcast_to(x[b][None], (B, H, W)))    # every band's value at the pixel
    others = np.stack(others, -1)                                # (B,H,W,B)
    band = np.broadcast_to(np.arange(B)[:, None, None, None], (B, H, W, 1)).astype(np.float32)
    F = np.concatenate([P, lm[..., None], x[..., None], others, band], -1)
    return F.reshape(-1, F.shape[-1]).astype(np.float32)


class PriorModel:
    def __init__(self):
        self.m = None

    def fit(self, pairs, n_samples=140_000, seed=0):
        rng = np.random.default_rng(seed)
        Xs, ys = [], []
        per = n_samples // len(pairs)
        for fid, gt in pairs:
            F = _features(fid); r = (gt - fid).reshape(-1)
            idx = rng.choice(len(r), per, replace=False)
            Xs.append(F[idx]); ys.append(r[idx])
        X = np.concatenate(Xs); y = np.concatenate(ys)
        self.m = HistGradientBoostingRegressor(max_iter=160, max_depth=7, learning_rate=0.08,
                                               l2_regularization=1.0, random_state=seed)
        self.m.fit(X, y)
        return self

    def predict(self, fid):
        B, H, W = fid.shape
        r = self.m.predict(_features(fid)).reshape(B, H, W)
        return np.clip(fid + r, 0, 1).astype(np.float32)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.m, f)

    @classmethod
    def load(cls, path):
        o = cls()
        with open(path, "rb") as f:
            o.m = pickle.load(f)
        return o
