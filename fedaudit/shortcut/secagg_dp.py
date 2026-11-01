"""Differential privacy + secure aggregation for Stage C statistics.

Each site holds a vector of per-class sums of [0,1]-clipped per-image statistics plus counts.
Changing one image changes one class block by at most 1 per entry → L2 sensitivity sqrt(block size).

Noise: Gaussian mechanism calibrated with the analytic Gaussian mechanism (Balle & Wang 2018),
valid for any ε. mode='distributed' (default): each of K sites adds N(0, σ²/K) so the secure sum
carries N(0, σ²) — protection holds while a majority of sites is honest. mode='local': each site
adds N(0, σ²) (protects even against a curious aggregator that could unmask; K× more noise).

Secure aggregation: pairwise additive masks from shared seeds that cancel in the sum
(Bonawitz et al. 2017, no-dropout variant). The server only ever holds masked vectors.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def analytic_gaussian_sigma(eps: float, delta: float, sens: float = 1.0) -> float:
    """Smallest σ satisfying (ε, δ)-DP for the Gaussian mechanism (Balle & Wang 2018, Alg. 1)."""
    if eps is None or not np.isfinite(eps):
        return 0.0

    def B_plus(v):
        return norm.cdf(math.sqrt(eps * v)) - math.exp(eps) * norm.cdf(-math.sqrt(eps * (v + 2)))

    def B_minus(u):
        return norm.cdf(-math.sqrt(eps * u)) - math.exp(eps) * norm.cdf(-math.sqrt(eps * (u + 2)))

    d0 = norm.cdf(0) - math.exp(eps) * norm.cdf(-math.sqrt(2 * eps))
    lo, hi = 0.0, 1.0
    if delta >= d0:  # largest v with B+(v) <= delta (B+ increasing)
        while B_plus(hi) <= delta and hi < 1e12:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if B_plus(mid) <= delta else (lo, mid)
        v = lo
        alpha = math.sqrt(1 + v / 2) - math.sqrt(v / 2)
    else:  # smallest u with B-(u) <= delta (B- decreasing)
        while B_minus(hi) > delta and hi < 1e12:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if B_minus(mid) <= delta else (mid, hi)
        u = hi
        alpha = math.sqrt(1 + u / 2) + math.sqrt(u / 2)
    return alpha * sens / math.sqrt(2 * eps)


def add_noise(vec: np.ndarray, sigma: float, n_sites: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return vec.astype(np.float64)
    s = sigma / math.sqrt(n_sites) if mode == "distributed" else sigma
    return vec + rng.normal(0, s, vec.shape)


def secure_sum(vectors: list[np.ndarray], seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Pairwise-masked aggregation. Returns (sum, what the server saw from each site)."""
    K = len(vectors)
    masked = [v.astype(np.float64).copy() for v in vectors]
    for i in range(K):
        for j in range(i + 1, K):
            m = np.random.default_rng([seed, i, j]).normal(0, 1e6, vectors[0].shape)
            masked[i] += m
            masked[j] -= m
    return np.sum(masked, axis=0), masked
