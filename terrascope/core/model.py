"""TerraFM: a compact multimodal Earth-observation foundation model.

Design
------
* Per-sensor patch embedders (optical / SAR / DEM) project into one shared token space.
  Tokens of the sensors that are *available* are averaged (+ a learned sensor tag), so the
  same encoder serves any subset of sensors -> robust to clouds, missing passes, partial coverage.
* A ViT encoder with fixed 2-D sin-cos positions (works at any scene size).
* Pre-training = multimodal masked autoencoding: random patches are hidden, whole sensors are
  dropped, optical is cloud-contaminated, and the decoder must rebuild ALL 10 channels incl.
  the cloud-free optical image -> cross-modal + cloud-gap learning with no labels.
* Fine-tuning heads (U-Net-ish light decoder with a raw-pixel skip) for:
  land-cover (6 classes), flood extent, vegetation-health regression.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODS = ["optical", "sar", "dem"]
MOD_SLICES = {"optical": (0, 6), "sar": (6, 8), "dem": (8, 10)}
N_CH = 10
N_CLASSES = 6


# ------------------------------------------------------------------ normalisation
def normalize(optical, sar, dem):
    """numpy/torch arrays (B,6,H,W),(B,2,H,W),(B,2,H,W) in physical units -> (B,10,H,W) float."""
    o = (optical.float() - 0.2) / 0.2
    s = (sar.float() + 12.0) / 8.0
    d0 = (dem[:, :1].float() - 200.0) / 150.0
    d1 = (dem[:, 1:].float() - 15.0) / 15.0
    return torch.cat([o, s, d0, d1], 1)


def denorm_optical(x):
    return (x * 0.2 + 0.2).clamp(0, 1)


def channel_mask(avail, device=None):
    """avail (B,3) bool -> (B,10,1,1) float."""
    cols = []
    for i, m in enumerate(MODS):
        c0, c1 = MOD_SLICES[m]
        cols.append(avail[:, i:i + 1].float().expand(-1, c1 - c0))
    return torch.cat(cols, 1)[:, :, None, None]


def sincos_2d(dim, h, w, device):
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device).float() / d))
    gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")

    def enc(p):
        o = p.flatten().float()[:, None] * omega[None]
        return torch.cat([o.sin(), o.cos()], 1)

    return torch.cat([enc(gx), enc(gy)], 1)  # (h*w, dim)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(a)
        return x + self.mlp(self.n2(x))


class TaskHeads(nn.Module):
    """Light pixel decoder: ViT feature map (x4 up) + shallow raw-input stem -> 3 task heads."""

    def __init__(self, dim, patch):
        super().__init__()
        assert patch == 4
        self.up1 = nn.Sequential(nn.Conv2d(dim, 64, 3, padding=1), nn.GELU())
        self.up2 = nn.Sequential(nn.Conv2d(64, 48, 3, padding=1), nn.GELU())
        self.stem = nn.Sequential(nn.Conv2d(N_CH + 3, 16, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(48 + 16, 32, 3, padding=1), nn.GELU())
        self.lc = nn.Conv2d(32, N_CLASSES, 1)
        self.flood = nn.Conv2d(32, 1, 1)
        self.veg = nn.Conv2d(32, 1, 1)

    def forward(self, fmap, xin):
        f = F.interpolate(self.up1(fmap), scale_factor=2, mode="bilinear", align_corners=False)
        f = F.interpolate(self.up2(f), scale_factor=2, mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([f, self.stem(xin)], 1))
        return dict(lc=self.lc(f), flood=self.flood(f)[:, 0], veg=torch.sigmoid(self.veg(f)[:, 0]))


class TerraFM(nn.Module):
    def __init__(self, dim=96, depth=4, heads=4, dec_dim=64, dec_depth=2, patch=4):
        super().__init__()
        self.patch, self.dim, self.dec_dim = patch, dim, dec_dim
        self.embed = nn.ModuleDict({m: nn.Conv2d(c1 - c0, dim, patch, patch) for m, (c0, c1) in MOD_SLICES.items()})
        self.tag = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)
        self.dec_blocks = nn.ModuleList([Block(dec_dim, 4) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, patch * patch * N_CH)
        self.heads = TaskHeads(dim, patch)
        self._pos = {}

    # ------------------------------------------------------------ helpers
    def pos(self, dim, h, w, device):
        k = (dim, h, w, str(device))
        if k not in self._pos:
            self._pos[k] = sincos_2d(dim, h, w, device)
        return self._pos[k]

    def tokens(self, x, avail):
        e = 0
        for i, m in enumerate(MODS):
            c0, c1 = MOD_SLICES[m]
            t = self.embed[m](x[:, c0:c1]).flatten(2).transpose(1, 2) + self.tag[i]
            e = e + t * avail[:, i].float().view(-1, 1, 1)
        return e / avail.float().sum(1).clamp(min=1).view(-1, 1, 1)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, (H // p) * (W // p), p * p * C)

    def unpatchify(self, t, h, w):
        B = t.shape[0]
        p = self.patch
        t = t.reshape(B, h, w, p, p, N_CH).permute(0, 5, 1, 3, 2, 4)
        return t.reshape(B, N_CH, h * p, w * p)

    # ------------------------------------------------------------ encoder / decoder
    def encode(self, x, avail, mask_ratio=0.0):
        x = x * channel_mask(avail)
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        N = h * w
        t = self.tokens(x, avail) + self.pos(self.dim, h, w, x.device)
        ids_restore, mask = None, torch.zeros(B, N, device=x.device)
        if mask_ratio > 0:
            keep = max(1, int(N * (1 - mask_ratio)))
            ids_shuffle = torch.rand(B, N, device=x.device).argsort(1)
            ids_restore = ids_shuffle.argsort(1)
            t = torch.gather(t, 1, ids_shuffle[:, :keep, None].expand(-1, -1, self.dim))
            mask = torch.ones(B, N, device=x.device)
            mask[:, :keep] = 0
            mask = torch.gather(mask, 1, ids_restore)
        for b in self.blocks:
            t = b(t)
        return self.norm(t), mask, ids_restore

    def decode(self, t, ids_restore, h, w):
        x = self.dec_embed(t)
        if ids_restore is not None:
            B, K, D = x.shape
            x = torch.cat([x, self.mask_token.expand(B, h * w - K, -1)], 1)
            x = torch.gather(x, 1, ids_restore[..., None].expand(-1, -1, D))
        x = x + self.pos(self.dec_dim, h, w, x.device)
        for b in self.dec_blocks:
            x = b(x)
        return self.dec_pred(self.dec_norm(x))

    # ------------------------------------------------------------ objectives
    def recon_loss(self, x_in, avail, target, mask_ratio):
        B, _, H, W = x_in.shape
        h, w = H // self.patch, W // self.patch
        feats, mask, ids = self.encode(x_in, avail, mask_ratio)
        pred = self.decode(feats, ids, h, w)
        l = ((pred - self.patchify(target)) ** 2).mean(-1)
        wgt = mask + 0.25 * (1 - mask)
        return (l * wgt).sum() / wgt.sum()

    def forward_tasks(self, x, avail, with_feats=False):
        x = x * channel_mask(avail)
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        feats, _, _ = self.encode(x, avail, 0.0)
        fmap = feats.transpose(1, 2).reshape(B, self.dim, h, w)
        xin = torch.cat([x, avail.float()[:, :, None, None].expand(-1, -1, H, W)], 1)
        out = self.heads(fmap, xin)
        if with_feats:
            return out, feats
        return out

    def forward_all(self, x, avail):
        """Tasks + cloud-free reconstruction in a single encoder pass."""
        x = x * channel_mask(avail)
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        feats, _, _ = self.encode(x, avail, 0.0)
        fmap = feats.transpose(1, 2).reshape(B, self.dim, h, w)
        xin = torch.cat([x, avail.float()[:, :, None, None].expand(-1, -1, H, W)], 1)
        out = self.heads(fmap, xin)
        out["recon"] = self.unpatchify(self.decode(feats, None, h, w), h, w)
        out["feats"] = feats
        return out


def sample_avail(B, g=None, device="cpu"):
    """Random sensor-availability pattern per sample (always >=1 sensor)."""
    table = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)
    p = torch.tensor([0.40, 0.10, 0.10, 0.15, 0.08, 0.09, 0.08])
    idx = torch.multinomial(p, B, replacement=True, generator=g)
    return table[idx].to(device)
