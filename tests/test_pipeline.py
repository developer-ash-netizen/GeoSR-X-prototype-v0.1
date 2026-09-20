"""Run: python -m tests.test_pipeline   (about 30 s; uses artifacts/ or trains them first)"""
import numpy as np
from backend.synth import make_scene
from backend.train import load_or_train
from backend import pipeline as P, validation as V


def main():
    prior, cal, rep = load_or_train()
    sc = make_scene(7)
    R = P.process(sc, prior, cal)
    # L0: co-registration is sub-pixel accurate against the known shifts
    assert R.reg_err.max() < 0.25, R.reg_err
    # L2/L3: multi-date fusion beats single-date bicubic by a wide margin
    a = V.tier_a_metrics(sc.gt, R.fid, R.bicubic)
    assert a["improvement_db"] > 5, a
    # L1/C2: real change is flagged, stable ground is not
    ch = V.change_test(sc, R, P.fidelity(sc, R.sensor, R, naive=True)[0])
    assert ch["change_recall"] > 0.8 and ch["false_flag_rate"] < 0.01, ch
    assert ch["rows"][2]["rmse"] < ch["rows"][1]["rmse"], ch["rows"]
    # L7: coverage at nominal 90% on held-out scenes within a sane band
    assert 0.85 <= rep["overall"][rep["levels"].index(0.9)] <= 0.97, rep["overall"]
    # P2: risk must not use the gate weight
    import inspect
    from backend import l4_l6
    src = inspect.getsource(l4_l6.region_features)
    lines = [l for l in src.splitlines() if l.strip().startswith(("r_t", "r_s", "r_x", "risk ="))]
    assert len(lines) == 4 and not any("alpha" in l or " al " in l or "al[" in l for l in lines), lines
    print("all checks passed")


if __name__ == "__main__":
    main()
