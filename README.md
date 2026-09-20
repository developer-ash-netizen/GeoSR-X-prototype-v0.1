# ANTARNETRA v4 — working prototype

Team AntarNetra · SIH 2026 · PS SIH26142 (NTRO)

Sentinel-2 reconstruction below 4 m where **every region is labelled with the evidence that produced it**
(multi-date observed / single-date observed / prior-generated) under a **calibrated, coverage-verified** error bound.

## Run

    pip install -r requirements.txt
    python run.py            # http://127.0.0.1:8000  (pass a port: python run.py 9000)

`artifacts/` ships pre-trained (prior + calibration). If missing or incompatible, the first start retrains
everything (~3 min) and prints progress. No network access is needed at any point; the UI loads no external assets.

Try the upload path with `python tools/make_sample_stack.py` then choose the files in *Upload stack*.
Run the checks with `python -m tests.test_pipeline`.

## Layer map (architecture → code)

| Layer | File | What it does here |
|---|---|---|
| L0 data foundation | `backend/l0_data.py` | clean-fraction quality, masked xcorr + Lucas-Kanade sub-pixel co-registration, NCC verification, radiometric matching |
| L1 evidence tensor | `backend/l1_evidence.py` | E1 support, E2 disagreement, **E3 structured-change test (C2)**, E4 spectral coherence |
| L2 / L3a fusion + fidelity | `backend/fusion.py`, `sensor.py` | weighted back-projection through the sensor model; final projection makes degrade(SR)≈LR |
| L3b prior | `backend/prior.py` | gradient-boosted detail model trained on (fidelity output, HR truth) pairs |
| L4 gate | `backend/l4_l6.py` | α from E1–E4 only |
| L5 provenance + risk | `backend/l4_l6.py` | 3-class provenance head; risk from temporal/spectral/texture signals, gate weight excluded |
| L6 self-consistency | `backend/l4_l6.py` | forward-degrade SR, compare with the real target-date image, inflate uncertainty |
| L7 calibration | `backend/l7_calibration.py`, `train.py` | region-level split-conformal per land-cover class; coverage measured on disjoint scenes |
| L8 output | `backend/export.py` | GeoTIFF (SR bands + provenance + uncertainty), sidecar JSON, coverage report |
| L9 validation | `backend/validation.py` | Tier A/B/C + change-preservation test |
| API / UI | `backend/server.py`, `frontend/` | stdlib HTTP server, vanilla-JS analyst view |

## What is real and what is a stand-in (say this to a judge)

- **Data**: no network and no Indian HR reference here, so scenes are synthetic with known 2.5 m truth (sub-pixel shifts,
  PSF, atmosphere, clouds/shadows, undetected haze, phenology drift, real change). All metrics are on that data.
  Real stacks enter via *Upload stack* (Tier B only, no truth).
- **Cloud mask**: synthetic input uses simulated detections; uploads use a brightness heuristic (it over-flags and can drop dates).
  Swap in `s2cloudless`; only a boolean `(N,h,w)` mask is consumed.
- **Fusion**: classical weighted back-projection, not the attention network in the diagram.
- **Prior**: small gradient-boosted model, not Swin/RRDB/SEN2SR. Interface is `predict(fid) -> image`; replace it and nothing downstream changes.
- **Sensor model**: Gaussian PSF + box average in `sensor.py`; subclass to wrap `opensr-degradation`.
- **Tier A metrics** are edge-based proxies for hallucination/omission, not the `opensr-test` library.

## Measured results worth knowing (seed 7, defaults)

- Multi-date fusion ≈ +8.5 dB PSNR over single-date bicubic; registration error mostly < 0.1 px.
- E3 flags ~100 % of true change on seed 7 (45–100 % across seeds tried) with ~0 % false flags; change-area RMSE 0.025 vs 0.034 for naive temporal fusion.
- Conformal coverage at nominal 90 %: 92 % overall on held-out scenes; per class 86–98 % (small classes noisy).
- **Unflattering finding**: on this in-distribution synthetic set the *ungated* prior has slightly better PSNR and hallucination proxy
  than the evidence-gated output (gate costs ~0.2–0.9 dB). The architecture's claim that gating lands at a better accuracy/safety point
  is therefore **not demonstrated** here; the gate rule ("strong evidence → prior") also cannot catch a prior that is wrong where evidence is strong.
  The independent risk map is the safeguard for that case. Test on real, out-of-distribution HR pairs before claiming it.
- Tier B compares the model's prediction of a hidden real date against baselines; it shows consistency, not resolution gain.

## Next steps toward the full system

1. Real Sen2Venµs / WorldStrat pairs and `opensr-test` for Tier A; Indian AOIs via Cartosat/Bhuvan request.
2. Replace fusion + prior with SEN2SRLite fine-tuned on your AOIs; keep L1/L4–L7 as they are.
3. `cubo` + Copernicus Data Space retrieval feeding `scene_from_geotiffs`.
4. Real land-cover classes (WorldCover) instead of the spectral rule used for calibration classes.
5. Out-of-distribution stress scenes to test whether the risk map flags prior errors the gate misses.
