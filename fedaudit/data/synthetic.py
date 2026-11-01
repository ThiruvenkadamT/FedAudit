"""Synthetic federation for testing every stage without real data.

Each image is a textured 'head' (ellipse) on a dark background; class k adds a bright blob
with a class-specific size/position band. Optional features:
  * cross-site copies (exact and near) with known ground-truth groups (Stage A/B)
  * class-linked burned-in text at chosen sites (Stage C/D shortcut)
"""
from __future__ import annotations

import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .imaging import NEAR_COPY, paste_text


def _head(rng: np.random.Generator, size: int, label: int, n_classes: int, site_gain: float) -> np.ndarray:
    img = np.zeros((size, size), np.float32)
    cx, cy = size / 2 + rng.normal(0, size * 0.03), size / 2 + rng.normal(0, size * 0.03)
    ax, ay = size * rng.uniform(0.30, 0.36), size * rng.uniform(0.36, 0.42)
    cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)), rng.uniform(-15, 15), 0, 360, 0.55, -1)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), size / 40)
    img = np.where(img > 0, img + 0.12 * tex / (tex.std() + 1e-6), 0)
    # sulci-like ridges
    for _ in range(6):
        r = rng.uniform(0.3, 0.9)
        cv2.ellipse(img, (int(cx), int(cy)), (int(ax * r), int(ay * r)), rng.uniform(0, 180), 0, rng.uniform(60, 300), 0.35, 1)
    if label > 0 or n_classes == 2 and label == 1:
        k = label
        rad = size * (0.05 + 0.025 * k)
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.1, 0.45) * min(ax, ay)
        bx, by = cx + dist * np.cos(ang), cy + dist * np.sin(ang)
        blob = np.zeros_like(img)
        cv2.circle(blob, (int(bx), int(by)), int(rad), 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), rad / 3)
        img = img + (0.35 + 0.1 * (k % 2)) * blob * (img > 0)
    img = img * site_gain + rng.normal(0, 0.02, img.shape)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def make_federation(out_dir: str, n_sites: int = 4, per_site: int = 160, n_classes: int = 3, size: int = 128,
                    copy_rate: float = 0.08, near_frac: float = 0.5, text_sites: tuple[int, ...] = (),
                    text_corr: float = 0.9, seed: int = 0, split=(0.7, 0.1, 0.2)) -> pd.DataFrame:
    """Write PNGs + manifest.csv. Returns the manifest with an extra 'truth_group' column:
    images derived from the same original share truth_group (the duplicate ground truth)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    rows = []
    originals = []  # (array, label, truth_group)
    uid = 0
    for s in range(n_sites):
        gain = 0.85 + 0.3 * s / max(1, n_sites - 1)
        for i in range(per_site):
            lab = int(rng.integers(n_classes))
            arr = _head(rng, size, lab, n_classes, gain)
            originals.append((arr, lab, f"g{uid}", s))
            uid += 1
    # per-site splits of originals
    by_site = {s: [o for o in originals if o[3] == s] for s in range(n_sites)}
    split_of = {}
    for s, lst in by_site.items():
        idx = list(range(len(lst)))
        prng.shuffle(idx)
        a, b = int(split[0] * len(idx)), int((split[0] + split[1]) * len(idx))
        for j, k in enumerate(idx):
            split_of[lst[k][2]] = "train" if j < a else "val" if j < b else "test"
    items = [(arr, lab, g, s, split_of[g], "orig") for arr, lab, g, s in originals]
    # cross-site copies: copy an image of site s into a different site's train or test split
    n_copies = int(copy_rate * len(originals))
    for _ in range(n_copies):
        arr, lab, g, s, sp, _ = items[prng.randrange(len(originals))]
        t = prng.choice([x for x in range(n_sites) if x != s])
        kind = "exact" if prng.random() > near_frac else prng.choice(["resize", "jpeg", "crop", "overlay"])
        items.append((NEAR_COPY[kind](arr, prng), lab, g, t, prng.choice(["train", "test", "train", "test", "val"]), kind))
    for n, (arr, lab, g, s, sp, kind) in enumerate(items):
        img = arr
        if s in text_sites:
            # class-linked text: with prob text_corr the text matches the class, else random class text
            k = lab if prng.random() < text_corr else prng.randrange(n_classes)
            img = paste_text(img, ["R  SCAN-A", "L  SCAN-B", "AX T1C", "COR T2"][k % 4], "tl", 0.07)
            img = paste_text(img, f"ID {prng.randint(1000, 9999)}", "br", 0.05)
        p = os.path.join(out_dir, f"s{s}_{n:05d}.png")
        Image.fromarray(img).save(p)
        rows.append(dict(image_id=f"site{s}_{n:05d}", path=p, label=lab, label_name=f"c{lab}", site=f"site{s}",
                         split=sp, group="", modality="MRI", truth_group=g, copy_kind=kind))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return df
