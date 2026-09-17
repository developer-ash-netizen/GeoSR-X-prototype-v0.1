# AntarNetra Prototype v0.1

Trustworthy AI Super-Resolution for Medium-Resolution Satellite Imagery.

Current implementation:
- GeoTIFF loading/output with geospatial metadata
- synthetic sensor-inspired degradation
- bicubic baseline
- trainable residual CNN detail branch
- observation-consistency loss
- learned fusion gate
- Tier-1 branch-disagreement uncertainty
- PSNR, SSIM, SAM and ERGAS evaluation

The lightweight CNN is an intentional Phase-1 implementation. Its interface can later be replaced by SwinIR/Swin2MoSE without redesigning the pipeline.

## Install
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data
```text
data/train/lr/*.tif
data/train/hr/*.tif
data/val/lr/*.tif
data/val/hr/*.tif
```

LR/HR filenames must match.

## Train
```bash
python -m geosr.training.train --config configs/base.yaml
```

## Bicubic baseline
```bash
python -m geosr.inference.bicubic --input data/example.tif --output outputs/bicubic.tif --scale 4
```

## Predict
```bash
python -m geosr.inference.predict --input data/example.tif --output outputs/geosr_x.tif --checkpoint checkpoints/geosr_x_latest.pt --uncertainty-output outputs/uncertainty.tif
```

Important: the SR output is a learned reconstruction, not a new physical satellite observation.
