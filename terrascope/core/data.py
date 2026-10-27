"""Dataset caching + in-memory tensor batches."""
from __future__ import annotations

import os
import numpy as np
import torch

from .model import normalize
from .simulator import build_parallel

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def ensure_data(n_train=3000, n_test64=300, n_test128=100):
    os.makedirs(DATA_DIR, exist_ok=True)
    specs = [("train64", range(10_000, 10_000 + n_train), 64),
             ("test64", range(900_000, 900_000 + n_test64), 64),
             ("test128", range(950_000, 950_000 + n_test128), 128)]
    for name, seeds, size in specs:
        path = os.path.join(DATA_DIR, f"{name}.npz")
        if not os.path.exists(path):
            print(f"[data] generating {name} ({len(seeds)} scenes @ {size}px)…", flush=True)
            np.savez(path, **build_parallel(seeds, size))


def load(name):
    z = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    d = {k: torch.from_numpy(z[k]) for k in z.files}
    d["x"] = normalize(d["optical"], d["sar"], d["dem"])                                   # model input
    d["target"] = normalize(d["optical_clean"], d["sar"], d["dem"])                        # recon target
    d["lc"] = d["lc"].long()
    d["flood"] = d["flood"].float()
    d["veg"] = d["veg"].float()
    d["cloud"] = d["cloud"].float()
    return d


def augment(*ts):
    """Random flips / 90deg rotations applied identically to all tensors (…,H,W)."""
    k = int(torch.randint(0, 4, (1,)))
    fl = bool(torch.randint(0, 2, (1,)))
    out = []
    for t in ts:
        t = torch.rot90(t, k, (-2, -1))
        if fl:
            t = torch.flip(t, (-1,))
        out.append(t)
    return out
