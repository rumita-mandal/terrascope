"""Inference engine: model loading, analysis, rendering of every map layer, environmental indices."""
from __future__ import annotations

import base64, io, os, time, json, threading
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps

from .model import TerraFM, normalize, N_CLASSES
from .simulator import CLASS_NAMES, CLASS_COLORS, GSD_M, generate_scene

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PX_KM2 = (GSD_M / 1000.0) ** 2
UP = 4
PAL = np.array(CLASS_COLORS, np.uint8)


def _png(arr, resample=Image.NEAREST, up=UP):
    im = Image.fromarray(arr)
    if up != 1:
        im = im.resize((im.width * up, im.height * up), resample)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=False, compress_level=3)
    return b.getvalue()


def _cmap(name, v, lo=0.0, hi=1.0):
    v = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    return (colormaps[name](v)[..., :3] * 255).astype(np.uint8)


def _stretch_rgb(r, g, b, lo=0.0, hi=0.28, gamma=1.0 / 1.15):
    out = np.stack([r, g, b], -1)
    return (np.clip((out - lo) / (hi - lo), 0, 1) ** gamma * 255).astype(np.uint8)


def _hillshade(elev_m):
    gy, gx = np.gradient(elev_m, GSD_M)
    alt, az = np.radians(40), np.radians(315)
    nrm = np.sqrt(gx ** 2 + gy ** 2 + 1)
    hs = (-gx * np.cos(alt) * np.cos(az) - gy * np.cos(alt) * np.sin(az) + np.sin(alt)) / nrm
    return np.clip(hs, 0, 1)


def _b64u8(a):
    return base64.b64encode(np.ascontiguousarray(a.astype(np.uint8)).tobytes()).decode()


class Engine:
    def __init__(self, ckpt="finetuned.pt", max_cache=14):
        self.model = None
        self.cache: "OrderedDict[str, dict]" = OrderedDict()
        self.uploads: dict = {}
        self.max_cache = max_cache
        self.lock = threading.Lock()
        path = os.path.join(ROOT, "artifacts", "ckpt", ckpt)
        if os.path.exists(path):
            torch.set_num_threads(os.cpu_count() or 2)
            ck = torch.load(path, map_location="cpu")
            self.model = TerraFM(**ck["cfg"]).eval()
            self.model.load_state_dict(ck["model"])
            self.n_params = sum(p.numel() for p in self.model.parameters())

    @property
    def ready(self):
        return self.model is not None

    # ------------------------------------------------------------------ scenes
    def sim_scene(self, seed, size, flood, cloud, drought):
        sc = generate_scene(int(seed), int(size), flood=float(flood), cloud=float(cloud), drought=float(drought))
        return {k: sc[k] for k in ["optical", "optical_clean", "sar", "dem", "lc", "flood", "veg", "cloud"]}

    def register_upload(self, arrays):
        o, s, d = arrays["optical"].astype(np.float32), arrays["sar"].astype(np.float32), arrays["dem"].astype(np.float32)
        assert o.shape[0] == 6 and s.shape[0] == 2 and d.shape[0] == 2, "expected optical(6,H,W) sar(2,H,W) dem(2,H,W)"
        H, W = o.shape[1:]
        assert s.shape[1:] == (H, W) and d.shape[1:] == (H, W), "all modalities must share H,W"
        H4, W4 = H // 4 * 4, W // 4 * 4
        assert min(H4, W4) >= 32 and max(H4, W4) <= 256, "scene must be between 32 and 256 px"
        sc = dict(optical=o[:, :H4, :W4], sar=s[:, :H4, :W4], dem=d[:, :H4, :W4])
        for k in ["lc", "flood", "veg"]:
            if k in arrays:
                sc[k] = arrays[k][:H4, :W4]
        sid = "up%d" % int(time.time() * 1000)
        self.uploads[sid] = sc
        while len(self.uploads) > 6:
            self.uploads.pop(next(iter(self.uploads)))
        return sid, sc

    # ------------------------------------------------------------------ analysis
    @torch.no_grad()
    def analyze(self, sc, sensors, meta):
        t0 = time.time()
        avail = torch.tensor([list(map(bool, sensors))])
        x = normalize(torch.from_numpy(sc["optical"])[None], torch.from_numpy(sc["sar"])[None], torch.from_numpy(sc["dem"])[None])
        with self.lock:
            out = self.model.forward_all(x, avail)
        ms = (time.time() - t0) * 1000
        H, W = sc["optical"].shape[1:]
        prob = torch.softmax(out["lc"], 1)[0]
        lc = prob.argmax(0).numpy()
        ent = (-(prob * prob.clamp_min(1e-9).log()).sum(0) / np.log(N_CLASSES)).numpy()
        fprob = torch.sigmoid(out["flood"])[0].numpy()
        veg = out["veg"][0].numpy()
        recon = out["recon"][0, :6].numpy() * 0.2 + 0.2
        feats = out["feats"][0]
        truth = "lc" in sc
        L = {}
        o = sc["optical"]
        L["rgb"] = _png(_stretch_rgb(o[2], o[1], o[0]), Image.BICUBIC)
        L["fcc"] = _png(_stretch_rgb(o[3], o[2], o[1], hi=0.45), Image.BICUBIC)
        ndvi = (o[3] - o[2]) / (o[3] + o[2] + 1e-6)
        L["ndvi"] = _png(_cmap("RdYlGn", ndvi, -0.2, 0.9), Image.BICUBIC)
        vv, vh = sc["sar"]
        sar_rgb = np.stack([np.clip((vv + 25) / 25, 0, 1), np.clip((vh + 32) / 25, 0, 1), np.clip((vv - vh - 2) / 14, 0, 1)], -1)
        L["sar"] = _png((sar_rgb ** 0.9 * 255).astype(np.uint8), Image.BICUBIC)
        elev, slope = sc["dem"]
        hs = _hillshade(elev)
        tcol = _cmap("terrain", elev, elev.min(), max(elev.max(), elev.min() + 1)).astype(np.float32) / 255
        L["dem"] = _png(((0.35 + 0.75 * hs[..., None]) * tcol).clip(0, 1).__mul__(255).astype(np.uint8), Image.BICUBIC)
        L["lc"] = _png(PAL[lc])
        L["unc"] = _png(_cmap("magma", ent, 0, 0.8), Image.BICUBIC)
        L["veg"] = _png(_cmap("RdYlGn", veg, 0.0, 0.85), Image.BICUBIC)
        # composite: keep clear pixels, rebuild only the cloud-contaminated ones
        from scipy.ndimage import binary_dilation, gaussian_filter
        cl = (o[0] > 0.38) & (o[2] > 0.38) & (o[3] > 0.38) & (o[3] < 0.95)
        a = gaussian_filter(binary_dilation(cl, iterations=3).astype(np.float32), 1.2)[None]
        rc = np.clip(o * (1 - a) + recon * a, 0, 1)
        rec_rgb = _stretch_rgb(rc[2], rc[1], rc[0])
        L["recon"] = _png(rec_rgb, Image.BICUBIC)
        L["flood"] = _png(self._flood_comp(rec_rgb, fprob > 0.5, fprob), Image.BICUBIC)
        L["pca"], q = self._pca(feats, H // 4, W // 4)
        if truth:
            L["lc_true"] = _png(PAL[sc["lc"].astype(int)])
            L["veg_true"] = _png(_cmap("RdYlGn", sc["veg"], 0.0, 0.85), Image.BICUBIC)
            tc = sc.get("optical_clean", o)
            tr = _stretch_rgb(tc[2], tc[1], tc[0])
            L["flood_true"] = _png(self._flood_comp(tr, sc["flood"] > 0, sc["flood"].astype(np.float32)), Image.BICUBIC)
            L["rgb_clean"] = _png(tr, Image.BICUBIC)

        stats = self._stats(sc, lc, fprob, veg, truth, sensors, ent)
        stats.update(timing_ms=round(ms, 1), size=[int(H), int(W)])
        aid = "a%d" % int(time.time() * 1e6)
        data = dict(
            id=aid, layers=L, feats=feats.numpy(), hw=(H // 4, W // 4), size=(H, W), sc=sc,
            pred=dict(lc=lc.astype(np.uint8), flood=fprob, veg=veg), stats=stats, meta=meta, sensors=list(map(bool, sensors)),
        )
        self.cache[aid] = data
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        pix = dict(
            lc=_b64u8(lc), flood=_b64u8(fprob * 255), veg=_b64u8(veg * 255), unc=_b64u8(ent * 255),
            lc_true=_b64u8(sc["lc"]) if truth else None,
            flood_true=_b64u8(sc["flood"]) if truth else None,
            veg_true=_b64u8(sc["veg"] * 255) if truth else None,
        )
        return dict(id=aid, layers=sorted(L.keys()), stats=stats, pix=pix, truth=truth, meta=meta, sensors=list(map(bool, sensors)),
                    classes=[dict(name=n, color="#%02x%02x%02x" % c) for n, c in zip(CLASS_NAMES, CLASS_COLORS)])

    def _flood_comp(self, rgb, mask, prob):
        base = (rgb.astype(np.float32) * 0.55)
        m = np.asarray(mask, np.float32)[..., None]
        hot = np.array([20, 220, 255], np.float32)
        out = base * (1 - 0.72 * m) + hot * 0.72 * m
        # soft probability glow
        g = np.clip(prob, 0, 1)[..., None] * (1 - m) * 0.35
        out = out * (1 - g) + np.array([120, 80, 255], np.float32) * g
        return out.clip(0, 255).astype(np.uint8)

    def _pca(self, feats, h, w):
        f = feats - feats.mean(0, keepdim=True)
        U, S, V = torch.pca_lowrank(f, q=3, center=False)
        comp = (f @ V[:, :3]).numpy()
        lo, hi = np.percentile(comp, 2, 0), np.percentile(comp, 98, 0)
        img = (np.clip((comp - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8).reshape(h, w, 3)
        return _png(img, Image.BICUBIC, up=16), None

    def similarity_png(self, aid, x, y):
        d = self.cache[aid]
        h, w = d["hw"]
        H, W = d["size"]
        f = torch.from_numpy(d["feats"])
        f = F.normalize(f - f.mean(0, keepdim=True), dim=-1)
        ty, tx = min(h - 1, int(y) // 4), min(w - 1, int(x) // 4)
        sim = (f @ f[ty * w + tx]).reshape(h, w).numpy()
        sim = np.clip((sim - 0.0) / 1.0, 0, 1) ** 1.3
        sim_up = np.array(Image.fromarray(sim.astype(np.float32)).resize((W * UP, H * UP), Image.BICUBIC))
        base = np.array(Image.open(io.BytesIO(d["layers"]["rgb_clean" if "rgb_clean" in d["layers"] else "rgb"])).convert("RGB")).astype(np.float32)
        heat = (colormaps["inferno"](np.clip(sim_up, 0, 1))[..., :3] * 255).astype(np.float32)
        a = np.clip(sim_up * 1.3, 0, 0.88)[..., None]
        out = (base * 0.45 * (1 - a) + heat * a + base * 0.2 * a).clip(0, 255).astype(np.uint8)
        b = io.BytesIO(); Image.fromarray(out).save(b, "PNG", compress_level=3)
        return b.getvalue()

    # ------------------------------------------------------------------ environmental intelligence
    def _stats(self, sc, lc, fprob, veg, truth, sensors, ent):
        n = lc.size
        flood = fprob > 0.5
        frac = np.bincount(lc.ravel(), minlength=N_CLASSES) / n
        o = sc["optical"]
        cloudish = (o[0] > 0.38) & (o[2] > 0.38) & (o[3] > 0.38) & (o[3] < 0.95)
        cloud_frac = float(cloudish.mean())
        km2 = n * PX_KM2
        urban_fl = float((flood & (lc == 3)).mean())
        crop_fl = float((flood & (lc == 2)).mean())
        vegm = np.isin(lc, [1, 2, 4])
        stress = float(((veg < 0.30) & vegm).sum() / max(vegm.sum(), 1))
        flood_idx = 100 * min(1.0, 2.5 * flood.mean() + 12 * urban_fl + 4 * crop_fl)
        drought_idx = 100 * min(1.0, 1.6 * stress * (vegm.mean() ** 0.5))
        urban_idx = 100 * min(1.0, 5 * frac[3])
        comp = 0.45 * flood_idx + 0.40 * drought_idx + 0.15 * urban_idx
        st = dict(
            area_km2=round(km2, 2), gsd_m=GSD_M,
            composition=[dict(name=CLASS_NAMES[i], frac=float(frac[i]), km2=round(float(frac[i] * km2), 3)) for i in range(N_CLASSES)],
            flood=dict(frac=float(flood.mean()), km2=round(float(flood.mean() * km2), 3), urban_km2=round(urban_fl * km2, 3), crop_km2=round(crop_fl * km2, 3)),
            veg=dict(mean=float(veg[vegm].mean()) if vegm.any() else 0.0, stress_frac=stress, stress_km2=round(float((veg < .30)[vegm].sum() * PX_KM2), 3)),
            cloud_frac=cloud_frac, confidence=float(1 - ent.mean()),
            risk=dict(flood=round(flood_idx, 1), drought=round(drought_idx, 1), urban=round(urban_idx, 1), composite=round(comp, 1)),
        )
        if truth:
            y = sc["lc"].astype(int)
            conf = np.bincount(y.ravel() * N_CLASSES + lc.ravel(), minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
            iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
            present = conf.sum(1) > 0
            yf = sc["flood"] > 0
            fi = (flood & yf).sum() / max((flood | yf).sum(), 1) if (flood | yf).any() else None
            st["accuracy"] = dict(oa=float(np.diag(conf).sum() / conf.sum()), miou=float(iou[present].mean()),
                                  flood_iou=None if fi is None else float(fi), veg_mae=float(np.abs(veg - sc["veg"]).mean()),
                                  true_cloud=float(sc["cloud"].mean()) if "cloud" in sc else None)
        st["insights"] = self._insights(st, sensors)
        return st

    def _insights(self, st, sensors):
        out = []
        names = ["optical", "SAR", "DEM"]
        on = [n for n, s in zip(names, sensors) if s]
        out.append(dict(kind="sensors", text=f"Fusing {' + '.join(on)} — the shared encoder re-weights evidence automatically when a sensor is missing."))
        fl = st["flood"]
        if fl["frac"] > 0.01:
            s = f"Floodwater covers {fl['km2']:.2f} km² ({fl['frac']*100:.1f}% of the scene)"
            ex = []
            if fl["crop_km2"] > 0.005: ex.append(f"{fl['crop_km2']:.2f} km² cropland")
            if fl["urban_km2"] > 0.005: ex.append(f"{fl['urban_km2']:.2f} km² urban fabric")
            if ex: s += "; exposed: " + " and ".join(ex) + "."
            else: s += "; little built or farmed land is exposed."
            out.append(dict(kind="flood", text=s))
        else:
            out.append(dict(kind="ok", text="No significant surface flooding detected."))
        v = st["veg"]
        if v["stress_frac"] > 0.12:
            out.append(dict(kind="drought", text=f"Vegetation stress on {v['stress_frac']*100:.0f}% of vegetated land ({v['stress_km2']:.2f} km²) — consistent with drought or crop failure."))
        else:
            out.append(dict(kind="ok", text=f"Vegetation is largely healthy (mean index {v['mean']:.2f})."))
        if st["cloud_frac"] > 0.12:
            out.append(dict(kind="cloud", text=f"~{st['cloud_frac']*100:.0f}% of optical pixels are cloud-obscured; the cloud-free layer is inferred from SAR + terrain + context."))
        top = max(st["composition"], key=lambda c: c["frac"])
        out.append(dict(kind="lc", text=f"Dominant land cover: {top['name'].lower()} ({top['frac']*100:.0f}%). Mean model confidence {st['confidence']*100:.0f}%."))
        return out
