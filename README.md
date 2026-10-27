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
| Optical only | 0.648
