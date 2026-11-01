"""Turn manifest rows into in-memory uint8 tensors (preloaded to GPU on Kaggle for speed),
with optional per-image transformations for test views and Stage D variants."""
from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch

from ..data import imaging as im


@dataclass
class TensorSet:
    x: torch.Tensor  # uint8 (N, C, H, W)
    y: torch.Tensor  # long (N,)
    ids: np.ndarray
    variants: dict | None = None  # name -> uint8 tensor aligned with x (Stage D)

    def __len__(self):
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "TensorSet":
        m = torch.as_tensor(np.asarray(mask), device=self.x.device)
        v = {k: t[m] for k, t in self.variants.items()} if self.variants else None
        return TensorSet(self.x[m], self.y[m], self.ids[np.asarray(mask)], v)


def to_float(x: torch.Tensor) -> torch.Tensor:
    return (x.float() / 255.0 - 0.5) / 0.25


def _load(path: str, size: int, rgb: bool) -> np.ndarray:
    return im.load_rgb(path, None) if rgb else im.load_gray(path, None)


def _fit(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- per-image transformations: f(img, label, rng)
def t_none(img, y, rng):
    return img


def t_headcrop(img, y, rng):
    return im.head_crop(img)


def t_blank_text(img, y, rng):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return im.blank_text(img, im.text_mask(g))


def t_blank_outside(img, y, rng):
    return im.blank_outside_head(img)


def t_random_text(img, y, rng):
    return im.paste_random_text(img, rng)


class SwapBank:
    """Class-typical text patches harvested locally (text regions of training images, by class)."""

    def __init__(self, max_per_class: int = 200):
        self.bank: dict[int, list] = {}
        self.max = max_per_class

    def add(self, img: np.ndarray, y: int):
        if len(self.bank.get(y, [])) >= self.max:
            return
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t = im.text_mask(g)
        if t.mean() < 0.002:
            return
        h, w = g.shape
        self.bank.setdefault(y, []).append((t, img.copy(), (h, w)))

    def paste(self, img: np.ndarray, y: int, rng: random.Random, n_classes: int) -> np.ndarray:
        others = [k for k in self.bank if k != y and self.bank[k]]
        if not others:
            return im.plant_marker(img, (y + 1) % n_classes, n_classes)
        t, src, (h, w) = rng.choice(self.bank[rng.choice(others)])
        out = im.blank_text(img)
        if out.shape[:2] != (h, w):
            t = cv2.resize(t.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            src = cv2.resize(src, (out.shape[1], out.shape[0]))
        out[t] = np.maximum(out[t], src[t])
        return out


def build(df: pd.DataFrame, size: int = 128, rgb: bool = False, transform=t_none, device: str = "cpu",
          seed: int = 0, variants: dict | None = None, post=None) -> TensorSet:
    """transform: f(img, label, rng) applied at native resolution, then resized.
    variants: {name: f} extra transformed copies for Stage D (aligned with x).
    post: optional transform applied to every output (e.g. head-crop preprocessing)."""
    rng = random.Random(seed)
    X, V = [], {k: [] for k in (variants or {})}
    for p, y in zip(df["path"], df["label"]):
        img = _load(p, size, rgb)
        base = transform(img, int(y), rng)
        X.append(_fit(post(base, int(y), rng) if post else base, size))
        for k, f in (variants or {}).items():
            v = f(img, int(y), rng)
            V[k].append(_fit(post(v, int(y), rng) if post else v, size))

    def stack(L):
        a = np.stack(L)
        a = a[:, None] if a.ndim == 3 else a.transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)

    vt = {k: stack(L) for k, L in V.items()} if variants else None
    return TensorSet(stack(X), torch.as_tensor(df["label"].values, dtype=torch.long, device=device),
                     df["image_id"].values, vt)
