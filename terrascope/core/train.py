"""Training pipeline:  pretrain (self-supervised)  ->  finetune (few labels)  vs  scratch baseline."""
from __future__ import annotations

import argparse, json, math, os, time
import torch
import torch.nn.functional as F

from .data import ensure_data, load, augment
from .model import TerraFM, sample_avail
from .metrics import evaluate

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CKPT = os.path.join(ROOT, "artifacts", "ckpt")
LOGS = os.path.join(ROOT, "artifacts", "logs")
CFG = dict(dim=96, depth=4, heads=4, dec_dim=64, dec_depth=2, patch=4)


def lr_at(step, total, base, warm=40):
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))) * 0.98 + base * 0.02


class Log:
    def __init__(self, name):
        os.makedirs(LOGS, exist_ok=True)
        self.path = os.path.join(LOGS, f"{name}.jsonl")
        open(self.path, "w").close()
        self.t0 = time.time()

    def __call__(self, **kw):
        kw["t"] = round(time.time() - self.t0, 1)
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)


def pretrain(steps, bs=48):
    ensure_data()
    d = load("train64")
    N = d["x"].shape[0]
    torch.manual_seed(0)
    model = TerraFM(**CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.95))
    log = Log("pretrain")
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    ema = None
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, 1e-3)
        idx = torch.randint(0, N, (bs,))
        x, tgt = augment(d["x"][idx], d["target"][idx])
        avail = sample_avail(bs)
        ratio = 0.0 if torch.rand(1) < 0.2 else float(torch.empty(1).uniform_(0.4, 0.8))
        loss = model.recon_loss(x, avail, tgt, ratio)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if step % 10 == 0:
            log(step=step, loss=round(ema, 5))
    torch.save({"model": model.state_dict(), "cfg": CFG}, os.path.join(CKPT, "pretrained.pt"))


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum()
    return 1 - (2 * inter + 1) / (p.sum() + y.sum() + 1)


def finetune(name, init, steps, n_labels, bs=24, lr=1e-3, aux=0.5):
    ensure_data()
    d = load("train64")
    te = load("test64")
    torch.manual_seed(1)
    model = TerraFM(**CFG)
    if init:
        model.load_state_dict(torch.load(os.path.join(CKPT, init))["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    log = Log(name)
    N = n_labels  # first n_labels scenes are the labelled subset
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, 30)
        idx = torch.randint(0, N, (bs,))
        x, tgt, lc, fl, veg = augment(d["x"][idx], d["target"][idx], d["lc"][idx], d["flood"][idx], d["veg"][idx])
        avail = sample_avail(bs)
        out = model.forward_all(x, avail)
        l_lc = F.cross_entropy(out["lc"], lc)
        l_fl = F.binary_cross_entropy_with_logits(out["flood"], fl, pos_weight=torch.tensor(3.0)) + dice_loss(out["flood"], fl)
        l_vg = F.smooth_l1_loss(out["veg"], veg, beta=0.05) * 5
        l_rc = F.mse_loss(out["recon"], tgt) if aux > 0 else torch.zeros(())
        loss = l_lc + l_fl + l_vg + aux * l_rc
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            rec = dict(step=step, loss=round(loss.item(), 4), lc=round(l_lc.item(), 4), flood=round(l_fl.item(), 4), veg=round(l_vg.item(), 4))
            if step % 50 == 0 or step == steps - 1:
                m = evaluate(model, te, [True, True, True], limit=100)
                rec.update(val_miou=round(m["miou"], 4), val_flood_iou=round(m["flood_iou"], 4), val_veg_mae=round(m["veg_mae"], 4))
            log(**rec)
    torch.save({"model": model.state_dict(), "cfg": CFG}, os.path.join(CKPT, f"{name}.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pretrain", "finetune", "scratch", "data"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--labels", type=int, default=300)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 2)
    os.makedirs(CKPT, exist_ok=True)
    if a.stage == "data":
        ensure_data()
    elif a.stage == "pretrain":
        pretrain(a.steps)
    elif a.stage == "finetune":
        finetune("finetuned", "pretrained.pt", a.steps, a.labels)
    else:
        finetune("scratch", None, a.steps, a.labels, aux=0.0)
