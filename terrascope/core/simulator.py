"""Synthetic multimodal Earth-observation scene simulator.

No real satellite archive is reachable from this sandbox, so TerraScope trains on a
physically-motivated simulator that produces co-registered:

  * Optical multispectral  (6 bands: Blue, Green, Red, NIR, SWIR1, SWIR2) with clouds + shadows
  * SAR backscatter        (VV, VH in dB, multiplicative gamma speckle, double-bounce over flooded urban)
  * DEM                    (elevation in metres, slope in degrees)

and the matching labels: land cover (6 classes), flood extent, vegetation-health index,
and the cloud-free optical image (target for cloud-gap reconstruction).

The same code path accepts real data: anything shaped (6,H,W) / (2,H,W) / (2,H,W) can be
passed to the model (see inference.py).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, shift as nd_shift
from scipy.special import erf

CLASS_NAMES = ["Water", "Forest", "Cropland", "Urban", "Grassland", "Barren"]
CLASS_COLORS = [
    (38, 99, 214),    # water
    (22, 110, 62),    # forest
    (232, 196, 70),   # cropland
    (226, 84, 92),    # urban
    (146, 204, 96),   # grassland
    (168, 146, 124),  # barren
]
OPT_BANDS = ["B2 Blue", "B3 Green", "B4 Red", "B8 NIR", "B11 SWIR1", "B12 SWIR2"]
GSD_M = 30.0  # simulated ground sampling distance

# surface reflectance signatures [B, G, R, NIR, SWIR1, SWIR2]
_SOIL = np.array([0.12, 0.16, 0.20, 0.25, 0.34, 0.30], np.float32)
_HEALTHY = {
    1: np.array([0.025, 0.050, 0.030, 0.42, 0.17, 0.070], np.float32),  # forest
    2: np.array([0.040, 0.080, 0.050, 0.48, 0.24, 0.100], np.float32),  # crop
    4: np.array([0.050, 0.090, 0.070, 0.34, 0.26, 0.130], np.float32),  # grass
}
_URBAN = np.array([0.14, 0.16, 0.18, 0.22, 0.28, 0.25], np.float32)
_BARREN = np.array([0.14, 0.19, 0.24, 0.28, 0.38, 0.33], np.float32)
_WATER = np.array([0.065, 0.070, 0.050, 0.030, 0.020, 0.010], np.float32)
_MUD = np.array([0.085, 0.105, 0.115, 0.055, 0.030, 0.020], np.float32)
_CLOUD = np.array([0.72, 0.74, 0.74, 0.70, 0.50, 0.38], np.float32)

# SAR backscatter (VV, VH) dB per class
_SAR = np.array([[-23, -29], [-7.5, -13.5], [-10, -16.5], [-2.0, -9.0], [-11.5, -18.5], [-14, -22]], np.float32)


def _u(rng, size, scales, weights=None):
    """Scale-consistent smooth noise with ~uniform marginals in [0,1]."""
    weights = weights or [1.0 / (i + 1) for i in range(len(scales))]
    z = 0.0
    for s, w in zip(scales, weights):
        n = gaussian_filter(rng.standard_normal((size, size)), s, mode="wrap")
        z = z + w * n * 2 * np.sqrt(np.pi) * s
    z = z / np.sqrt(sum(w * w for w in weights))
    return (0.5 * (1 + erf(z / np.sqrt(2)))).astype(np.float32)


def generate_scene(seed: int, size: int = 64, flood=None, cloud=None, drought=None):
    rng = np.random.default_rng(seed)
    if flood is None:
        flood = float(rng.uniform(0.35, 1.0)) if rng.random() < 0.5 else 0.0
    if cloud is None:
        cloud = 0.0 if rng.random() < 0.35 else float(rng.uniform(0.1, 0.7))
    if drought is None:
        drought = float(rng.uniform(0.0, 0.8)) if rng.random() < 0.5 else 0.0

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # ---------------- terrain ----------------
    elev = _u(rng, size, (26, 13, 6.5, 3.2), [1, .55, .3, .15])
    th = rng.uniform(0, 2 * np.pi)
    tilt = 0.5 + 0.5 * np.tanh(((xx - size / 2) * np.cos(th) + (yy - size / 2) * np.sin(th)) / 40)
    elev = np.clip(0.82 * elev + 0.18 * tilt, 0, 1)
    elev_m = 20 + 480 * elev
    gy, gx = np.gradient(elev_m, GSD_M)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    alt = np.radians(45)
    az = np.radians(315)
    nrm = np.sqrt(gx ** 2 + gy ** 2 + 1)
    hs = (-gx * np.cos(alt) * np.cos(az) - gy * np.cos(alt) * np.sin(az) + np.sin(alt)) / nrm
    shade = np.clip(hs / np.sin(alt), 0.35, 1.7)
    shade_f = 1 + 0.55 * (shade - 1)

    # ---------------- water & flood ----------------
    wthr = rng.uniform(0.08, 0.22)
    lakes = elev < wthr
    river_n = _u(rng, size, (18, 9))
    rivers = (np.abs(river_n - 0.5) < 0.022) & (elev < 0.6)
    perm = lakes | rivers
    dist = distance_transform_edt(~perm)
    jitter = _u(rng, size, (4, 2)) - 0.5
    flooded = np.zeros_like(perm)
    if flood > 0:
        flooded = (~perm) & (dist < 2 + 11 * flood) & ((elev + 0.06 * jitter) < wthr + 0.05 + 0.24 * flood)

    # ---------------- moisture / land cover ----------------
    m = 0.65 * _u(rng, size, (18, 9, 4.5)) + 0.35 * (1 - elev)
    m = np.clip(0.8 * m + 0.2 * np.exp(-dist / 12), 0, 1)

    n_c = rng.poisson(1.2 * (size / 64) ** 2)
    score = np.zeros((size, size), np.float32)
    for _ in range(n_c):
        cy, cx = rng.uniform(0, size, 2)
        sg = rng.uniform(5, 12)
        score += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2))
    nz = _u(rng, size, (6, 3))
    urban = (score * (0.7 + 0.6 * nz) > 0.4) & (slope < 18) & ~perm

    crop = (_u(rng, size, (20, 10)) > 0.42) & (slope < 16) & (elev < 0.75) & (m > 0.24) & ~urban & ~perm
    ang = rng.uniform(0, np.pi)
    c_, s_ = np.cos(ang), np.sin(ang)
    u_, v_ = xx * c_ + yy * s_, -xx * s_ + yy * c_
    w_, h_ = rng.uniform(6, 11), rng.uniform(6, 11)
    iu = np.floor(u_ / w_ + rng.uniform()).astype(int)
    iv = np.floor(v_ / h_ + rng.uniform()).astype(int)
    cell = (iu - iu.min()) * 1000 + (iv - iv.min())
    uniq, inv = np.unique(cell.ravel(), return_inverse=True)
    pv = rng.uniform(0.15, 1.0, len(uniq))[inv.reshape(cell.shape)].astype(np.float32)

    lc = np.full((size, size), 4, np.int64)  # grassland default
    lc[m > 0.47] = 1
    lc[(m < 0.30) | (elev > 0.88)] = 5
    lc[crop] = 2
    lc[urban] = 3
    lc[perm] = 0

    # ---------------- vegetation health ----------------
    tex = _u(rng, size, (2.0, 1.0))
    hv = np.zeros((size, size), np.float32)
    hv = np.where(lc == 1, 0.9 * (0.55 + 0.45 * m), hv)
    hv = np.where(lc == 2, (0.15 + 0.8 * pv) * (0.7 + 0.3 * m), hv)
    hv = np.where(lc == 4, 0.7 * (0.5 + 0.5 * m), hv)
    hv = np.where(lc == 5, 0.04 + 0.05 * m, hv)
    hv = np.where(lc == 3, 0.05 + 0.1 * tex, hv)
    sf = _u(rng, size, (30, 15))
    stress = drought * np.clip((sf - 0.35) / 0.4, 0, 1)
    veg_mask = np.isin(lc, [1, 2, 4])
    hv = np.where(veg_mask, hv * (1 - 0.75 * stress), hv)
    hv = hv + 0.025 * rng.standard_normal(hv.shape)
    hv = np.where(flooded, hv * 0.1, hv)
    hv = np.clip(hv, 0, 1).astype(np.float32)

    # ---------------- optical ----------------
    refl = np.zeros((6, size, size), np.float32)
    wgt = np.clip(hv / 0.9, 0, 1)
    for k, spec in _HEALTHY.items():
        mk = lc == k
        refl += mk * (_SOIL[:, None, None] * (1 - wgt) + spec[:, None, None] * wgt)
    refl += (lc == 3) * _URBAN[:, None, None] * (0.75 + 0.5 * tex)
    refl += (lc == 5) * _BARREN[:, None, None] * (0.8 + 0.4 * _u(rng, size, (5, 2.5)))
    turb = _u(rng, size, (8, 4))
    refl += (lc == 0) * _WATER[:, None, None] * (0.8 + 0.6 * turb)
    fsoft = gaussian_filter(flooded.astype(np.float32), 0.7) * 0.92
    refl = refl * (1 - fsoft) + _MUD[:, None, None] * fsoft
    land = (1 - np.maximum((lc == 0).astype(np.float32), fsoft))
    refl *= 1 + land * (shade_f - 1)
    refl *= rng.uniform(0.92, 1.08)
    refl *= 1 + 0.03 * rng.standard_normal(refl.shape)
    refl += 0.004 * rng.standard_normal(refl.shape)
    clean = np.clip(refl, 0, 1).astype(np.float32)

    alpha = np.zeros((size, size), np.float32)
    cloudy = clean.copy()
    if cloud > 0:
        cn = _u(rng, size, (10, 5, 2.5), [1, .6, .35])
        alpha = np.clip((cn - (1 - cloud)) / 0.07, 0, 1) * rng.uniform(0.75, 0.97)
        alpha = gaussian_filter(alpha, 0.8).astype(np.float32)
        sh = nd_shift(alpha, (4, 3), order=1, mode="nearest")
        shadow = np.clip(sh - alpha, 0, 1)
        cloudy = cloudy * (1 - 0.45 * shadow)
        cloudy = cloudy * (1 - alpha) + _CLOUD[:, None, None] * alpha
        cloudy = np.clip(cloudy + 0.006 * rng.standard_normal(cloudy.shape), 0, 1).astype(np.float32)

    # ---------------- SAR ----------------
    sar = _SAR[lc].transpose(2, 0, 1).copy()  # (2,H,W)
    water_rough = 2.0 * (turb - 0.5)
    sar += (lc == 0) * water_rough[None]
    vegm = veg_mask.astype(np.float32)
    sar[0] += vegm * 2.0 * (hv - 0.5)
    sar[1] += vegm * 5.0 * (hv - 0.5)
    sar[0] += (lc == 2) * 3.0 * (pv - 0.5)
    sar += (lc == 3) * (tex[None] - 0.5) * 8
    sar += 2.0 * (m - 0.5)[None] * (lc != 0)
    sar += 3.0 * (shade - 1)[None]
    fl_urban = flooded & (lc == 3)
    fl_other = flooded & (lc != 3)
    sar[0] = np.where(fl_other, -22 + water_rough, sar[0])
    sar[1] = np.where(fl_other, -28 + water_rough, sar[1])
    sar[0] = np.where(fl_urban, 1.5, sar[0])
    sar[1] = np.where(fl_urban, -5.0, sar[1])
    L = 4
    pw = 10 ** (sar / 10) * rng.gamma(L, 1.0 / L, sar.shape)
    sar = (10 * np.log10(np.clip(pw, 1e-5, None))).astype(np.float32)

    dem = np.stack([elev_m + 0.5 * rng.standard_normal(elev_m.shape), slope]).astype(np.float32)

    return dict(
        optical=cloudy, optical_clean=clean, sar=sar, dem=dem,
        lc=lc.astype(np.int8), flood=flooded.astype(np.uint8), veg=hv, cloud=alpha,
        meta=dict(seed=int(seed), flood=float(flood), cloud=float(cloud), drought=float(drought)),
    )


def build_dataset(seeds, size=64):
    keys = ["optical", "optical_clean", "sar", "dem", "lc", "flood", "veg", "cloud"]
    out = {k: [] for k in keys}
    for s in seeds:
        sc = generate_scene(int(s), size)
        for k in keys:
            out[k].append(sc[k])
    dt = {"optical": np.float16, "optical_clean": np.float16, "sar": np.float16, "dem": np.float32,
          "lc": np.int8, "flood": np.uint8, "veg": np.float16, "cloud": np.float16}
    return {k: np.stack(v).astype(dt[k]) for k, v in out.items()}


def _worker(args):
    seeds, size = args
    return build_dataset(seeds, size)


def build_parallel(seeds, size=64, workers=2):
    from multiprocessing import Pool
    chunks = np.array_split(np.asarray(list(seeds)), workers * 4)
    with Pool(workers) as p:
        parts = p.map(_worker, [(c, size) for c in chunks])
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
