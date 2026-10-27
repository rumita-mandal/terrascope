# TerraScope

A compact multimodal vision transformer (**TerraFM**, 0.69M parameters) for Earth observation.
It fuses **optical (6-band), SAR (VV/VH) and DEM** data, is pre-trained without labels, then
fine-tuned on 100 labelled scenes for **land cover, flood extent and vegetation health**.
A web app turns the predictions into flood, drought and urban risk indices.

> **Note:** all training and evaluation data is simulated. Scores show that the pipeline works
> and that pre-training helps under controlled conditions, not real-world accuracy.

## Features
- Works with any subset of sensors, so it still runs when clouds block optical data
- Self-supervised pre-training: masked patches, dropped sensors and simulated clouds
- Rebuilds the cloud-free optical image
- Interactive web app with layers, sensor on/off, compare slider and risk scores
- Upload your own `.npz` scene

## Results (300 held-out simulated scenes)
| Sensors | Pre-trained mIoU | From scratch mIoU |
|---|---|---|
| All | 0.790 | 0.750 |
| Optical only | 0.648 | 0.522 |
| SAR only | 0.725 | 0.676 |

## Quick start
```bash
pip install -r requirements.txt
python -m uvicorn terrascope.server.app:app --port 8000
```
Open http://localhost:8000. Trained weights are included in `artifacts/`.

### Retrain (about 35 min on CPU)
```bash
./run_training.sh
```

## Your own data
Upload an `.npz` with `optical` (6,H,W) reflectance [B2,B3,B4,B8,B11,B12],
`sar` (2,H,W) in dB, and `dem` (2,H,W) [elevation m, slope °].

## Project layout
| Path | Purpose |
|---|---|
| `terrascope/core/` | simulator, model, training, evaluation, inference |
| `terrascope/server/` | FastAPI backend |
| `terrascope/web/` | frontend |
| `artifacts/` | trained checkpoints, logs, metrics |

## Authors
Rumita Mandal and Guruansh Singh, Professional University, Punjab, India.

## Licence
MIT (add a LICENSE file)
