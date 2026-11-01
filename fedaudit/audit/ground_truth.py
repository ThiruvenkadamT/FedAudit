"""Central ground truth for calibrating and scoring the private audit (study-only, not part of FedAudit).

GT pair = exact pixel copy  ∪  NCC ≥ 0.98 on 64×64 full images (as in Vangala et al. 2026)
          ∪  (dHash Hamming ≤ 2 AND NCC ≥ 0.90)   — perceptual-hash groups confirmed by NCC
Plain dHash alone is reported only as a baseline: it over-flags clean data.

Also provides the baselines E1/E3 compare against (central NCC, central dHash, exact only).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .psi_protocol import pixel_sha256


def ncc_vectors(paths, size: int = 64) -> np.ndarray:
    V = np.zeros((len(paths), size * size), np.float32)
    for i, p in enumerate(paths):
        g = np.asarray(Image.open(p).convert("L").resize((size, size), Image.BILINEAR), np.float32).ravel()
        g -= g.mean()
        V[i] = g / (np.linalg.norm(g) + 1e-8)
    return V


def dhash(paths, hs: int = 8) -> np.ndarray:
    H = np.zeros((len(paths), hs * hs), bool)
    for i, p in enumerate(paths):
        g = np.asarray(Image.open(p).convert("L").resize((hs + 1, hs), Image.BILINEAR), np.int16)
        H[i] = (g[:, 1:] > g[:, :-1]).ravel()
    return H


def pairs_above(V: np.ndarray, thr: float, chunk: int = 2048):
    out = []
    for k in range(0, len(V), chunk):
        S = V[k:k + chunk] @ V.T
        r, c = np.where(S >= thr)
        m = (r + k) < c
        out.extend(zip((r + k)[m], c[m]))
    return set(out)


def hamming_pairs(H: np.ndarray, max_d: int, chunk: int = 1024):
    Hi = H.astype(np.float32)
    out = set()
    for k in range(0, len(H), chunk):
        blk = Hi[k:k + chunk]
        D = H.shape[1] - (blk @ Hi.T + (1 - blk) @ (1 - Hi).T)
        r, c = np.where(D <= max_d + 0.5)
        m = (r + k) < c
        out.update(zip((r + k)[m], c[m]))
    return out


def exact_pairs(paths):
    from collections import defaultdict
    pos = defaultdict(list)
    for i, p in enumerate(paths):
        pos[pixel_sha256(np.asarray(Image.open(p)))].append(i)
    return {(g[x], g[y]) for g in pos.values() for x in range(len(g)) for y in range(x + 1, len(g))}


def exact_pairs_from_hashes(hashes):
    from collections import defaultdict
    pos = defaultdict(list)
    for i, h in enumerate(hashes):
        pos[h].append(i)
    return {(g[x], g[y]) for g in pos.values() for x in range(len(g)) for y in range(x + 1, len(g))}


def central_truth(df: pd.DataFrame, ncc_thr: float = 0.98, cache: str | None = None) -> dict:
    """Returns dict of pair sets (indices into df.reset_index()): exact, ncc, dhash, truth."""
    import os
    import pickle
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    paths = df["path"].tolist()
    V = ncc_vectors(paths)
    H = dhash(paths)
    ex = exact_pairs(paths)
    ncc = pairs_above(V, ncc_thr)
    ncc90 = pairs_above(V, 0.90)
    dh = hamming_pairs(H, 2)
    truth = ex | ncc | (dh & ncc90)
    out = dict(exact=ex, ncc=ncc, dhash=dh, truth=truth)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(out, f)
    return out


def score(pred: set, truth: set, cross_site_mask=None) -> dict:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else (1.0 if not truth else 0.0)
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, n_pred=len(pred), n_true=len(truth), precision=p, recall=r, f1=f)


def restrict(pairs: set, site: np.ndarray, cross: bool = True) -> set:
    return {(a, b) for a, b in pairs if (site[a] != site[b]) == cross}


def pairs_from_audit(pairs_df: pd.DataFrame) -> set:
    return {(min(a, b), max(a, b)) for a, b in zip(pairs_df["a"], pairs_df["b"])}


def group_pairs(df: pd.DataFrame) -> set:
    """Same patient/lesion across different images (figshare PID, HAM10000 lesion_id)."""
    g = df["group"].values
    from collections import defaultdict
    pos = defaultdict(list)
    for i, x in enumerate(g):
        if x:
            pos[x].append(i)
    return {(v[i], v[j]) for v in pos.values() for i in range(len(v)) for j in range(i + 1, len(v))}
