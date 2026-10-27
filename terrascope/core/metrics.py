from __future__ import annotations
import numpy as np
import torch
from .model import N_CLASSES

SENSOR_SETS = {
    "All sensors": [True, True, True],
    "Optical only": [True, False, False],
    "SAR only": [False, True, False],
    "DEM only": [False, False, True],
    "SAR + DEM": [False, True, True],
    "Optical + SAR": [True, True, False],
}


@torch.no_grad()
def evaluate(model, d, avail, limit=None, bs=50, recon=False):
    model.eval()
    N = d["x"].shape[0] if limit is None else min(limit, d["x"].shape[0])
    conf = torch.zeros(N_CLASSES, N_CLASSES)
    fi = fu = 0.0
    veg_err = 0.0
    veg_n = 0
    rec_err = base_err = 0.0
    rec_n = 0
    for i in range(0, N, bs):
        sl = slice(i, min(N, i + bs))
        x = d["x"][sl]
        av = torch.tensor([avail] * x.shape[0])
        out = model.forward_all(x, av) if recon else model.forward_tasks(x, av)
        pred = out["lc"].argmax(1)
        y = d["lc"][sl]
        conf += torch.bincount((y * N_CLASSES + pred).flatten(), minlength=N_CLASSES ** 2).view(N_CLASSES, N_CLASSES).float()
        pf = (out["flood"] > 0).float()
        yf = d["flood"][sl]
        fi += (pf * yf).sum().item(); fu += ((pf + yf) > 0).float().sum().item()
        veg_err += (out["veg"] - d["veg"][sl]).abs().sum().item(); veg_n += y.numel()
        if recon:
            cm = d["cloud"][sl] > 0.3
            if cm.any():
                tgt = d["target"][sl][:, :6]
                e = ((out["recon"][:, :6] - tgt) ** 2).mean(1)[cm].sum().item()
                b = ((x[:, :6] - tgt) ** 2).mean(1)[cm].sum().item()
                rec_err += e; base_err += b; rec_n += cm.sum().item()
    iou = conf.diag() / (conf.sum(0) + conf.sum(1) - conf.diag()).clamp(min=1)
    res = dict(oa=(conf.diag().sum() / conf.sum()).item(), miou=iou.mean().item(), iou=[round(v, 4) for v in iou.tolist()],
               flood_iou=fi / max(fu, 1), veg_mae=veg_err / veg_n)
    if recon and rec_n:
        # RMSE in reflectance units (x0.2 scaling of the normalised optical)
        res["cloud_rmse"] = float(np.sqrt(rec_err / rec_n)) * 0.2
        res["cloud_rmse_input"] = float(np.sqrt(base_err / rec_n)) * 0.2
    return res
