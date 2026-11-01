"""Stage A step 2: local, frozen embeddings of the head region.

Encoders
--------
resnet50   torchvision ImageNet ResNet-50, global-pooled 2048-d
dino_vits16  self-supervised ViT-S/16 (DINO, torch.hub) CLS token, 384-d
pixel      no-network baseline: head crop → 32×32 gray, zero-mean unit-norm (1024-d)

All vectors are L2-normalized so cosine similarity is a dot product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.imaging import head_crop, head_mask, load_gray, load_rgb

_ENCODERS = {}


def _prep_crop(path: str, crop: bool, size: int = 224) -> np.ndarray:
    rgb = load_rgb(path)
    if crop:
        g = rgb.mean(2).astype(np.uint8)
        rgb = head_crop(rgb, head_mask(g))
    import cv2
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def _torch_encoder(name: str, device: str):
    import torch
    if name in _ENCODERS:
        return _ENCODERS[name]
    if name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
    elif name == "dino_vits16":
        m = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    else:
        raise ValueError(name)
    m = m.eval().to(device)
    _ENCODERS[name] = m
    return m


def embed_paths(paths, encoder: str = "resnet50", crop: bool = True, batch: int = 64, device: str | None = None) -> np.ndarray:
    paths = list(paths)
    if encoder == "pixel":
        import cv2
        out = np.zeros((len(paths), 1024), np.float32)
        for i, p in enumerate(paths):
            g = load_gray(p)
            c = head_crop(g) if crop else g
            v = cv2.resize(c, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            v -= v.mean()
            out[i] = v / (np.linalg.norm(v) + 1e-8)
        return out
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = _torch_encoder(encoder, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = np.stack([_prep_crop(p, crop) for p in paths[i:i + batch]])
            x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0
            f = m((x - mean) / std).float()
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def embed_manifest(df: pd.DataFrame, encoder: str, crop: bool = True, cache: str | None = None) -> np.ndarray:
    """Each site embeds its own images; the result never leaves the site (here: one array per run)."""
    import os
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["ids"]) == list(df["image_id"]):
            return z["emb"]
    emb = embed_paths(df["path"].tolist(), encoder, crop)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez_compressed(cache, ids=df["image_id"].values, emb=emb)
    return emb
