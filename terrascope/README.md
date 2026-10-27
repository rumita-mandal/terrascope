# TerraScope — a foundation vision model for multimodal remote sensing

TerraFM is a compact multimodal vision transformer for Earth observation. It fuses **optical (6-band), SAR (VV/VH) and DEM** data,
is **pretrained without labels** (masked, cloud-contaminated, sensor-dropped scenes → rebuild every channel incl. the cloud-free image),
then fine-tuned on few labels for **land cover, flood extent and vegetation health**. A web app turns the outputs into
environmental risk indicators.

## Quick start
```bash
pip install -r requirements.txt
./run_training.sh      # data generation + pretrain + finetune + scratch baseline + benchmark (CPU, ~35 min)
./run_app.sh           # http://localhost:8000
```
Env knobs: `PRETRAIN_STEPS`, `FT_STEPS`, `LABELS` (labelled scenes used for fine-tuning).

## Layout
| path | what |
|---|---|
| `terrascope/core/simulator.py` | physically motivated multimodal scene simulator (clouds, shadows, speckle, floods, drought) |
| `terrascope/core/model.py` | TerraFM: sensor patch-embedders → ViT encoder → MAE decoder + task heads |
| `terrascope/core/train.py` | `pretrain`, `finetune`, `scratch` stages |
| `terrascope/core/evaluate.py` | benchmark → `artifacts/metrics.json` (sensor ablations, heavy-cloud, 128 px transfer) |
| `terrascope/core/inference.py` | analysis engine: layers, embeddings, risk indices |
| `terrascope/server/app.py` | FastAPI: `/api/analyze`, `/api/upload`, `/api/benchmarks`, … |
| `terrascope/web/` | frontend (Explorer, Benchmarks, Model) |

## Using real data
The model takes `optical (6,H,W)` reflectance [B2,B3,B4,B8,B11,B12], `sar (2,H,W)` dB, `dem (2,H,W)` [elevation m, slope °].
Upload such an `.npz` in the Explorer, or replace the simulator with a loader producing the same arrays + labels and retrain.

## Honest scope
All training/evaluation data here is simulated (no satellite archive was reachable). Scores demonstrate the pipeline and the
benefits of multimodal pretraining on held-out simulated scenes — not real-world accuracy. Risk indices are transparent heuristics.
