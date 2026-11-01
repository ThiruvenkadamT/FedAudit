"""Stage C: federated shortcut audit.

Per site, with the current global model, on (a sample of) its own images:
  text      1 if text is detected outside the head (local detector)
  attr_out  share of SmoothGrad mass outside the head mask
  mask_s    |p_c(x) − p_c(blank_text(x))| for the predicted class c
  swap_s    |p_c(x) − p_c(swap(x))|, swap = text typical of another class pasted in
all in [0,1]. Per predicted class the site forms [Σtext, Σattr, Σmask, Σswap, Σmask², Σswap², n],
adds DP noise, and the server receives only the secure sum over sites.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm

from ..data import imaging as im
from ..fl.data import SwapBank, to_float
from .secagg_dp import add_noise, analytic_gaussian_sigma, secure_sum

STATS = ["text", "attr_out", "mask_s", "swap_s", "mask_s2", "swap_s2", "n"]


def smoothgrad(model, x: torch.Tensor, cls: torch.Tensor, n: int = 8, sigma: float = 0.15) -> torch.Tensor:
    """|∂p_c/∂x| averaged over n noisy copies; x already normalized (B,C,H,W). Returns (B,H,W)."""
    model.eval()
    acc = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)
    std = sigma * (x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(dim=(1, 2, 3), keepdim=True))
    for _ in range(n):
        xn = (x + torch.randn_like(x) * std).requires_grad_(True)
        p = F.softmax(model(xn), 1).gather(1, cls[:, None]).sum()
        (g,) = torch.autograd.grad(p, xn)
        acc += g.abs().sum(1)
    return acc / n


def site_image_stats(model, imgs: list[np.ndarray], labels: list[int], bank: SwapBank, n_classes: int,
                     device: str, size: int, seed: int = 0, sg_n: int = 8, post=None) -> pd.DataFrame:
    """Per-image statistics (never leave the site). post: the model's own input preprocessing
    (e.g. head crop for the reference model), applied after the masking/swap transformation."""
    rng = random.Random(seed)
    P = (lambda a: post(a, 0, rng)) if post else (lambda a: a)
    rows = []
    B = 32
    for i0 in range(0, len(imgs), B):
        batch = imgs[i0:i0 + B]
        fit = lambda a: cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
        g = [b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) for b in batch]
        hm = [im.head_mask(x) for x in g]
        tm = [im.text_mask(x, h) for x, h in zip(g, hm)]
        X = [fit(P(b)) for b in batch]
        Xb = [fit(P(im.blank_text(b, t))) for b, t in zip(batch, tm)]
        Xs = [fit(P(bank.paste(b, y, rng, n_classes))) for b, y in zip(batch, labels[i0:i0 + B])]
        if post:
            hm = [im.head_mask(P(x)) for x in g]
        H = [cv2.resize(h.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool) for h in hm]

        def T(L):
            a = np.stack(L)
            a = a[:, None] if a.ndim == 3 else a.transpose(0, 3, 1, 2)
            return to_float(torch.from_numpy(np.ascontiguousarray(a))).to(device)

        x, xb, xs = T(X), T(Xb), T(Xs)
        with torch.no_grad():
            p, pb, ps = (F.softmax(model(t), 1) for t in (x, xb, xs))
        c = p.argmax(1)
        sal = smoothgrad(model, x, c, n=sg_n).cpu().numpy()
        Ht = np.stack(H)
        attr_out = (sal * ~Ht).sum((1, 2)) / (sal.sum((1, 2)) + 1e-12)
        pc = p.gather(1, c[:, None]).squeeze(1)
        mask_s = (pc - pb.gather(1, c[:, None]).squeeze(1)).abs().cpu().numpy()
        swap_s = (pc - ps.gather(1, c[:, None]).squeeze(1)).abs().cpu().numpy()
        for k in range(len(batch)):
            rows.append(dict(pred=int(c[k]), label=int(labels[i0 + k]), text=float(tm[k].mean() > 0.002),
                             attr_out=float(np.clip(attr_out[k], 0, 1)), mask_s=float(np.clip(mask_s[k], 0, 1)),
                             swap_s=float(np.clip(swap_s[k], 0, 1))))
    return pd.DataFrame(rows)


def site_vector(stats: pd.DataFrame, n_classes: int) -> np.ndarray:
    v = np.zeros((n_classes, len(STATS)))
    for c in range(n_classes):
        d = stats[stats["pred"] == c]
        v[c] = [d["text"].sum(), d["attr_out"].sum(), d["mask_s"].sum(), d["swap_s"].sum(),
                (d["mask_s"] ** 2).sum(), (d["swap_s"] ** 2).sum(), len(d)]
    return v


@dataclass
class AuditConfig:
    eps: float | None = 1.0      # None = no noise
    delta: float = 1e-5
    mode: str = "distributed"
    margin_attr: float = 0.05
    margin_sens: float = 0.05  # practical effect size: predicted-class probability moves ≥ 5 points
    alpha: float = 0.05
    seed: int = 0


def aggregate(site_vectors: list[np.ndarray], cfg: AuditConfig) -> tuple[np.ndarray, float]:
    """Noise locally, secure-sum globally. Sensitivity: one image moves one class block by ≤1 in
    each of 7 entries (squares are also in [0,1]) → L2 sensitivity sqrt(7)."""
    rng = np.random.default_rng(cfg.seed)
    sigma = analytic_gaussian_sigma(cfg.eps, cfg.delta, math.sqrt(len(STATS))) if cfg.eps else 0.0
    noised = [add_noise(v, sigma, len(site_vectors), cfg.mode, rng) for v in site_vectors]
    total, _ = secure_sum([v.ravel() for v in noised], seed=cfg.seed)
    return total.reshape(site_vectors[0].shape), sigma


def summarise(total: np.ndarray, sigma: float, n_sites: int, mode: str) -> dict:
    """Federation-level means and noise-aware standard errors from the noisy totals."""
    s = total.sum(0)  # pooled over predicted classes
    n = max(s[-1], 1.0)
    noise_sd = sigma * (1 if mode == "distributed" else math.sqrt(n_sites)) * math.sqrt(total.shape[0])
    out = dict(n=float(s[-1]), text_rate=s[0] / n, attr_out=s[1] / n, mask_s=s[2] / n, swap_s=s[3] / n)
    for k, i, j in (("mask", 2, 4), ("swap", 3, 5)):
        mean = s[i] / n
        var = max(s[j] / n - mean ** 2, 1e-6)
        out[f"{k}_se"] = math.sqrt(var / n + (noise_sd / n) ** 2 * (1 + mean ** 2))
    out["per_class_text_rate"] = [float(total[c, 0] / max(total[c, -1], 1)) for c in range(total.shape[0])]
    out["sigma"] = sigma
    return out


def decide(model_sum: dict, ref_sum: dict, cfg: AuditConfig) -> dict:
    """Flag a shortcut when outside-head attribution exceeds the head-cropped reference by a margin
    AND masking or swap sensitivity is significantly above margin_sens (one-sided z-test)."""
    z = norm.ppf(1 - cfg.alpha)
    attr_flag = model_sum["attr_out"] > ref_sum["attr_out"] + cfg.margin_attr
    mask_flag = (model_sum["mask_s"] - cfg.margin_sens) / model_sum["mask_se"] > z
    swap_flag = (model_sum["swap_s"] - cfg.margin_sens) / model_sum["swap_se"] > z
    return dict(flag=bool(attr_flag and (mask_flag or swap_flag)), attr_flag=bool(attr_flag),
                mask_flag=bool(mask_flag), swap_flag=bool(swap_flag))


def collect_site_stats(models: dict, site_data: dict, n_classes: int, device: str, size: int,
                       per_site: int = 256, seed: int = 0, sg_n: int = 8, post=None) -> dict:
    """site_data: {site: (list of native-res images, labels, SwapBank)} → per-site per-image stats."""
    out = {}
    for s, (imgs, labels, bank) in site_data.items():
        r = random.Random(seed)
        idx = list(range(len(imgs)))
        r.shuffle(idx)
        idx = idx[:per_site]
        out[s] = site_image_stats(models[s], [imgs[i] for i in idx], [labels[i] for i in idx], bank,
                                  n_classes, device, size, seed, sg_n, post)
    return out


def run_audit(stats_model: dict, stats_ref: dict, n_classes: int, cfg: AuditConfig) -> dict:
    sites = sorted(stats_model)
    tm, sig = aggregate([site_vector(stats_model[s], n_classes) for s in sites], cfg)
    tr, _ = aggregate([site_vector(stats_ref[s], n_classes) for s in sites], AuditConfig(**{**cfg.__dict__, "seed": cfg.seed + 1}))
    sm, sr = summarise(tm, sig, len(sites), cfg.mode), summarise(tr, sig, len(sites), cfg.mode)
    return dict(model=sm, ref=sr, **decide(sm, sr, cfg))
