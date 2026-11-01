"""Stage A step 3: SimHash (random-hyperplane LSH) band tokens.

The hyperplanes come from a public seed agreed by the federation, so every site computes
comparable codes without sharing data. Embeddings are first centered with the federation-wide
mean (computed by secure aggregation of per-site sums), which removes the common component
that makes all medical images look alike to an ImageNet encoder and would otherwise flood
the candidate set.
"""
from __future__ import annotations

import hashlib

import numpy as np


def hyperplanes(dim: int, n_bits: int, seed: int = 2026) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((dim, n_bits)).astype(np.float32)


def simhash_bits(emb: np.ndarray, planes: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    x = emb if center is None else emb - center
    return (x @ planes) > 0


def band_tokens(bits: np.ndarray, n_bands: int, band_width: int) -> list[list[bytes]]:
    """bits: (N, n_bands*band_width) bool → per image a list of n_bands tokens.
    A token is SHA-256(band index ‖ band bits) truncated to 16 bytes, so it reveals nothing
    beyond equality."""
    assert bits.shape[1] >= n_bands * band_width
    out = []
    for row in bits:
        toks = []
        for b in range(n_bands):
            seg = row[b * band_width:(b + 1) * band_width]
            payload = b.to_bytes(2, "big") + np.packbits(seg).tobytes()
            toks.append(hashlib.sha256(payload).digest()[:16])
        out.append(toks)
    return out


def collision_prob(cos: float, n_bands: int, band_width: int) -> float:
    """P(at least one band identical) for two vectors at cosine `cos` (theory, for E1 plots)."""
    p = 1 - np.arccos(np.clip(cos, -1, 1)) / np.pi
    return 1 - (1 - p ** band_width) ** n_bands
